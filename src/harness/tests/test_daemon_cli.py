"""The detached path: config round-trips, services assemble, the CLI drives it.

Covers the seam that only existed in manual testing -- `harness start` writes a
config, spawns a daemon, and everything after talks to the store.
"""
import json

import pytest

from harness.cli import main as cli_main
from harness.daemon import FleetConfig, build
from harness.session import SqliteSessionStore


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    """The daemon opens the shared skill bank; tests must not touch ~/."""
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))


def test_config_round_trips(tmp_path):
    """A daemon must be restartable from its config alone."""
    cfg = FleetConfig(session_id="s1", root=str(tmp_path), agents=7,
                      levels=(1, 2, 4, 8), seeds=("try a thing",), dry_run=True)
    p = tmp_path / "fleet.json"
    cfg.save(p)
    # A field an older harness wrote must not stop a restart.
    p.write_text(p.read_text().replace('"agents": 7', '"agents": 7, "mode": "build"'))
    back = FleetConfig.load(p)
    assert back.agents == 7 and back.levels == (1, 2, 4, 8)
    assert back.seeds == ("try a thing",) and back.dry_run


def test_build_assembles_every_service(tmp_path):
    """The one place implementations are chosen; if it drifts, nothing runs."""
    store = SqliteSessionStore(tmp_path / "s.db")
    cfg = FleetConfig(session_id="s1", root=str(tmp_path / "agents"),
                      agents=2, eval_capacity=2, dry_run=True, fake_agents=True)
    fleet, broker = build(cfg, store=store)
    try:
        assert broker.capacity == 2
        assert fleet.session_id == "s1" and fleet.root
        agent = fleet.make_agent("a00", fleet)
        assert agent.agent_id == "a00"
        assert agent.evals is broker
        assert agent.control is fleet
    finally:
        broker.shutdown()


def test_quality_baseline_reaches_the_evaluator():
    """Without it the gate is not weak, it is absent (see `evaluator_for`)."""
    from harness.daemon import evaluator_for

    cfg = FleetConfig(session_id="s1", baseline={"bill_per_1k": 12.0,
                                                 "quality": {"gsm8k": 0.62}})
    full = evaluator_for(cfg, "full")
    assert full.extra["quality_baseline"] == {"gsm8k": 0.62}
    assert full.levels == cfg.levels and full.seconds_per_level == cfg.seconds_per_level
    screen = evaluator_for(cfg, "screen")
    assert screen.levels == cfg.screen_levels
    assert screen.seconds_per_level == cfg.screen_seconds
    assert evaluator_for(FleetConfig(session_id="s2"), "full").extra == {
        "quality_baseline": {}}


def test_a_real_fleet_refuses_a_baseline_it_cannot_score_against():
    """Each missing piece is a way the fleet runs all night and learns nothing."""
    from harness.daemon import check

    check(FleetConfig(session_id="s", dry_run=True, fake_agents=True))   # fakes need nothing
    with pytest.raises(SystemExit, match="bank"):
        check(FleetConfig(session_id="s", dry_run=True))     # real agents cannot self-seed
    full = {"bill_per_1k": 14.96, "quality": {"gsm8k": 0.66},
            "screen": {"bill_per_1k": 17.3}}
    check(FleetConfig(session_id="s", bank="ideas.db", baseline=full))
    for missing in ("bill_per_1k", "quality", "screen"):
        with pytest.raises(SystemExit):
            check(FleetConfig(session_id="s", bank="ideas.db",
                              baseline={k: v for k, v in full.items() if k != missing}))
    check(FleetConfig(session_id="s", bank="ideas.db", screen_first=False,
                      baseline={k: v for k, v in full.items() if k != "screen"}))


def test_the_bank_and_the_manager_reach_the_proposer(tmp_path):
    """`--bank --manager`: the proposer reads the manager's tool index;
    nothing shells out at construction."""
    from harness.ideas import SqliteIdeaBank

    bank = tmp_path / "ideas.db"
    SqliteIdeaBank(bank)
    cfg = FleetConfig(session_id="s1", root=str(tmp_path / "agents"), agents=1,
                      dry_run=True, bank=str(bank), manager=True,
                      baseline={"bill_per_1k": 12.23, "quality": {"gsm8k": 0.69},
                                "screen": {"bill_per_1k": 17.3}})
    fleet, broker = build(cfg, store=SqliteSessionStore(tmp_path / "s.db"))
    try:
        assert fleet.bank is not None and fleet.manager is not None
        agent = fleet.make_agent("a00", fleet)
        assert not hasattr(agent.proposer, "seed"), "real agents claim, never invent"
        assert agent.proposer.session_tools is not None
        fleet.manager.stash.add("bench", "bench a kernel", "python tools/bench.py", "print(1)", 2)
        from harness.contracts import Brief
        assert "bench.py" in agent.proposer._brief_text(Brief(text="known"), "")
    finally:
        broker.shutdown()
    import pytest as _pt

    from harness.daemon import check
    with _pt.raises(SystemExit):
        check(FleetConfig(session_id="s2",
                          baseline={"bill_per_1k": 1, "quality": {"gsm8k": 0.6},
                                    "screen": {"bill_per_1k": 1}}))


def test_fake_agents_never_shell_out(tmp_path, monkeypatch):
    """`--fake-agents` must not spend subscription usage; the offline socket
    guard would catch a network call, but not a subprocess."""
    import subprocess

    def boom(*a, **k):
        raise AssertionError("fake agents must not run claude")

    monkeypatch.setattr(subprocess, "run", boom)
    cfg = FleetConfig(session_id="s1", root=str(tmp_path / "agents"),
                      agents=1, dry_run=True, fake_agents=True)
    fleet, broker = build(cfg, store=SqliteSessionStore(tmp_path / "s.db"))
    try:
        from harness.contracts import Brief, Idea
        agent = fleet.make_agent("a00", fleet)
        idea = agent.proposer.seed((), Brief(text=""))
        assert isinstance(idea, Idea) and idea.title
    finally:
        broker.shutdown()


def test_cli_reports_no_session_cleanly(tmp_path, capsys):
    rc = cli_main(["--store", str(tmp_path / "s.db"), "status"])
    assert rc == 1
    assert "no session" in capsys.readouterr().err.lower()


def test_cli_status_renders_a_snapshot(tmp_path, capsys):
    from harness.contracts.session import AgentView, SessionView, TokenUse

    db = tmp_path / "s.db"
    store = SqliteSessionStore(db)
    v = SessionView(session_id="s1", phase="running", started_at=1.0,
                    target_agents=2, cost_usd=3.5, budget_usd=100.0,
                    tokens=TokenUse(1, 2, 3, 4),
                    agents=(AgentView("a00", status="thinking",
                                      idea_title="lfu eviction",
                                      activity="writing a diff"),))
    store.create(v)
    store.publish(v)
    assert cli_main(["--store", str(db), "status"]) == 0
    out = capsys.readouterr().out
    assert "s1" in out and "lfu eviction" in out and "writing a diff" in out


def test_cli_calls_summarises_a_call_log(tmp_path, capsys):
    """`harness calls` reads the per-call JSONL an agent's proposer wrote.

    The only view of where an agent-hour went, and it is parsed rather than
    stored: a row shape that drifts shows up here first.
    """
    root = tmp_path / "run"
    d = root / "a01" / "calls"
    d.mkdir(parents=True)
    rows = [
        {"ts": 100.0, "since_prev_s": 1.0, "type": "assistant", "input": 10,
         "output": 5, "cache_read": 700, "cache_write": 0, "tools": ["Bash"]},
        {"ts": 400.0, "since_prev_s": 300.0, "type": "assistant", "input": 3,
         "output": 7, "cache_read": 300, "cache_write": 0,
         "tools": ["Edit", "Bash"]},
        {"ts": 460.0, "since_prev_s": 60.0, "type": "result", "num_turns": 9},
    ]
    (d / "edit-1700000000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    assert cli_main(["--store", str(tmp_path / "s.db"), "calls",
                     "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "a01" in out and "edit-1700000000" in out
    assert "2 msgs" in out and "6.0 min" in out      # 100s -> 460s
    assert "1,000" in out                            # cache reads, both messages
    assert "turns 9" in out and "Bashx2" in out and "Editx1" in out

    assert cli_main(["--store", str(tmp_path / "s.db"), "calls",
                     "--root", str(root), "-v"]) == 0
    assert "300.0s" in capsys.readouterr().out       # the per-message rows


def test_cli_status_json_is_machine_readable(tmp_path, capsys):
    from harness.contracts.session import SessionView

    db = tmp_path / "s.db"
    store = SqliteSessionStore(db)
    store.create(SessionView(session_id="s1", phase="running"))
    store.publish(SessionView(session_id="s1", phase="running"))
    assert cli_main(["--store", str(db), "status", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["session_id"] == "s1" and d["phase"] == "running"


@pytest.mark.parametrize("args,kind", [
    (["scale", "3"], "scale"),
    (["agent", "pause", "a01"], "pause"),
    (["stop"], "stop"),
])
def test_cli_commands_reach_the_store(tmp_path, args, kind, capsys):
    """The CLI never touches the fleet directly -- it writes a command row."""
    import time

    from harness.contracts.session import SessionView

    db = tmp_path / "s.db"
    store = SqliteSessionStore(db)
    # A session that stopped publishing: the realistic case for a CLI command
    # that nobody will acknowledge.
    dead = SessionView(session_id="s1", phase="running",
                       updated_at=time.time() - 120)
    store.create(dead)
    store.publish(dead)

    t0 = time.time()
    assert cli_main(["--store", str(db), *args]) == 1
    assert time.time() - t0 < 2.0, "waited on a session that is not running"
    assert "last published" in capsys.readouterr().err
    assert [c.kind for c in store.take_commands("s1")] == [kind]


def test_cli_waits_for_a_live_session(tmp_path, capsys):
    """When something *is* ticking, the CLI reports what the fleet decided."""
    import threading
    import time

    from harness.contracts.session import SessionView

    db = tmp_path / "s.db"
    store = SqliteSessionStore(db)
    live = SessionView(session_id="s1", phase="running")
    store.create(live)
    store.publish(live)

    def acknowledge():
        for _ in range(100):
            time.sleep(0.02)
            pend = store.take_commands("s1")
            if pend:
                store.acknowledge(pend[0].id, "target 3")
                return

    t = threading.Thread(target=acknowledge)
    t.start()
    try:
        assert cli_main(["--store", str(db), "--wait", "5", "scale", "3"]) == 0
        assert "target 3" in capsys.readouterr().out
    finally:
        t.join()


def test_agents_use_the_subscription_not_an_api_key(monkeypatch):
    """Claude Code prefers an API key over the subscription when both are set.

    A real fleet run printed "claude.ai connectors are disabled because
    ANTHROPIC_API_KEY ... takes precedence" before every call -- i.e. it was
    quietly billing the API. The key is stripped unless asked for.
    """
    from harness.agent.claude_code import ClaudeCodeProposer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    assert "ANTHROPIC_API_KEY" not in ClaudeCodeProposer()._env()
    assert "ANTHROPIC_BASE_URL" not in ClaudeCodeProposer()._env()
    assert ClaudeCodeProposer(use_api_key=True)._env()["ANTHROPIC_API_KEY"] \
        == "sk-should-not-leak"


def test_start_refuses_a_second_daemon_on_the_same_root(tmp_path):
    """Two daemons on one root reset each other's agent workspaces."""
    import os

    from harness.cli import running_daemon

    assert running_daemon(tmp_path) == 0
    (tmp_path / "daemon.pid").write_text(str(os.getpid()))
    assert running_daemon(tmp_path) == os.getpid()
    (tmp_path / "daemon.pid").write_text("999999999")
    assert running_daemon(tmp_path) == 0


def test_status_marks_a_dead_daemon(tmp_path, capsys):
    from harness.cli import _mark_dead, daemon_alive
    from harness.contracts.session import AgentView, SessionView

    assert not daemon_alive(999_999_999) and not daemon_alive(0)
    v = SessionView(session_id="x", phase="running", pid=999_999_999,
                    agents=(AgentView("a00", status="thinking", activity="editing"),))
    d = _mark_dead(v)
    assert d.phase == "dead" and d.agents[0].status == "lost"
    assert "daemon exited" in d.agents[0].activity
    assert _mark_dead(SessionView(session_id="y", phase="stopped", pid=0)).phase == "stopped"


# ── compounding: --base ─────────────────────────────────────────────────

def _saved_stack(tmp_path):
    """A run directory as `simulate run` / an attempt leaves it."""
    from simulator import InferenceStack

    run = tmp_path / "fleet-1" / "a02" / "runs" / "attempt-001"
    run.mkdir(parents=True)
    st = InferenceStack(files={"srt/managers/schedule_policy.py": "CHUNK = 4096\n"},
                        serving={"chunked_prefill_size": 4096}, label="a02: ISRTF")
    (run / "stack.json").write_text(json.dumps(st.as_dict()))
    return run, st


def test_a_base_that_does_not_load_is_refused(tmp_path):
    from harness.daemon import check

    ok = _saved_stack(tmp_path)[0]
    cfg = FleetConfig(session_id="s", dry_run=True, fake_agents=True)
    check(FleetConfig(session_id="s", dry_run=True, fake_agents=True, base=str(ok)))
    with pytest.raises(SystemExit, match="stock"):          # nothing there to build on
        check(FleetConfig(session_id="s", dry_run=True, fake_agents=True,
                          base=str(tmp_path / "nowhere")))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(SystemExit, match="does not load"):
        check(FleetConfig(session_id="s", dry_run=True, fake_agents=True, base=str(bad)))
    with pytest.raises(SystemExit, match="does not load"):
        cfg.with_base(str(bad))


def test_start_records_the_base_and_its_idea(tmp_path, monkeypatch):
    """fleet.json says what the fleet compounds on -- path, digest, label --
    and the hypothesis that produced it, read from the memory beside it."""
    import harness.cli as cli
    from harness.contracts import Experiment
    from harness.memory import SqliteMemory

    run, st = _saved_stack(tmp_path)
    SqliteMemory(tmp_path / "fleet-1" / "memory.db").record(Experiment(
        agent_id="a02", hypothesis="serve shortest remaining prefill first",
        stack_digest=st.digest, verdict="win"))

    class Proc:
        pid = 4242

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: Proc())
    root = tmp_path / "fleet-2"
    assert cli_main(["--session", "s2", "start", "--root", str(root), "--dry-run",
                     "--fake-agents", "--base", str(run)]) == 0
    cfg = FleetConfig.load(root / "fleet.json")
    assert cfg.base == str(run) and cfg.base_digest == st.digest
    assert cfg.base_label == "a02: ISRTF"
    assert cfg.base_idea == "serve shortest remaining prefill first"
    assert cfg.base_seed == "a02: ISRTF serve shortest remaining prefill first"
    # Refused in the terminal, before a daemon is spawned to fail in its log.
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(SystemExit, match="does not load"):
        cli_main(["--session", "s3", "start", "--root", str(tmp_path / "f3"),
                  "--dry-run", "--fake-agents", "--base", str(tmp_path / "bad.json")])
    assert not (tmp_path / "f3" / "fleet.json").exists()


def test_every_agent_starts_from_the_base(tmp_path):
    """The daemon hands the base to each workspace, and takes a stale one
    away when the next fleet on the same root starts from stock."""
    from harness.tests.test_workspace import FakeStock  # noqa: F401  (import check only)

    run, st = _saved_stack(tmp_path)
    root = tmp_path / "agents"
    cfg = FleetConfig(session_id="s1", root=str(root), dry_run=True, fake_agents=True,
                      base=str(run), base_digest=st.digest)
    fleet, broker = build(cfg)
    try:
        agent = fleet.make_agent("a01", fleet)
        assert agent.workspace.base is not None and agent.workspace.base.digest == st.digest
        assert (root / "a01" / "base.json").is_file()
        assert "CHUNK = 4096" in agent.workspace.read("srt/managers/schedule_policy.py")
    finally:
        broker.shutdown(wait=False)
    fleet, broker = build(FleetConfig(session_id="s2", root=str(root), dry_run=True,
                                      fake_agents=True))
    try:
        assert fleet.make_agent("a01", fleet).workspace.base is None
        assert not (root / "a01" / "base.json").exists()
    finally:
        broker.shutdown(wait=False)


def test_status_and_sessions_name_the_base(tmp_path, capsys):
    from harness.contracts.session import SessionView

    run, st = _saved_stack(tmp_path)
    root = tmp_path / "agents"
    root.mkdir()
    FleetConfig(session_id="s1", root=str(root), base=str(run), base_digest=st.digest,
                base_label="a02: ISRTF").save(root / "fleet.json")
    store = SqliteSessionStore(tmp_path / "s.db")
    store.create(SessionView(session_id="s1", phase="stopped", root=str(root),
                             note=f"base {st.digest}"))
    cli_main(["--store", str(tmp_path / "s.db"), "--session", "s1", "sessions"])
    assert f"base {st.digest}" in capsys.readouterr().out
    cli_main(["--store", str(tmp_path / "s.db"), "--session", "s1", "status"])
    assert f"base:  {st.digest} a02: ISRTF" in capsys.readouterr().out


def test_tool_ablate_dispatches_or_says_it_is_missing(monkeypatch, capsys):
    import harness.tools as tools

    monkeypatch.delattr(tools, "ablate", raising=False)
    assert cli_main(["tool", "ablate", "--env", "SGLANG_X=1"]) == 2
    assert "ablate is not available" in capsys.readouterr().err
    seen = {}

    def fake(workspace, env, tier):
        seen.update(workspace=workspace, env=env, tier=tier)
        return {"ok": True, "delta_pct": -3.2}

    monkeypatch.setattr(tools, "ablate", fake, raising=False)
    assert cli_main(["tool", "ablate", "--workspace", "agents/s/a01", "--env", "SGLANG_X=1",
                     "--env", "SGLANG_Y=a=b", "--tier", "full", "--json"]) == 0
    assert seen == {"workspace": "agents/s/a01", "env": {"SGLANG_X": "1", "SGLANG_Y": "a=b"},
                    "tier": "full"}
    assert json.loads(capsys.readouterr().out)["delta_pct"] == -3.2
    assert cli_main(["tool", "ablate", "--env", "NOEQUALS"]) == 2


def test_a_fleet_starts_with_the_shared_stock_profile(tmp_path, monkeypatch):
    from harness.daemon import seed_stock_profile

    home = tmp_path / "home"
    (home / "profiles").mkdir(parents=True)
    (home / "profiles" / "stock.sqlite").write_bytes(b"sqlite")
    monkeypatch.setenv("HARNESS_HOME", str(home))
    root = tmp_path / "agents" / "s"
    root.mkdir(parents=True)
    assert seed_stock_profile(root) is True
    assert (root / "profiles" / "stock.sqlite").read_bytes() == b"sqlite"
    assert seed_stock_profile(root) is False          # never overwrites
    (home / "profiles" / "stock.sqlite").unlink()
    assert seed_stock_profile(tmp_path / "agents" / "t") is False
