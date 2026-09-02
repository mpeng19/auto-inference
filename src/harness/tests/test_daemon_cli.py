"""The detached path: config round-trips, services assemble, the CLI drives it.

Covers the seam that only existed in manual testing -- `harness start` writes a
config, spawns a daemon, and everything after talks to the store.
"""
import json

import pytest

from harness.cli import main as cli_main
from harness.daemon import FleetConfig, build
from harness.session import SqliteSessionStore


def test_config_round_trips(tmp_path):
    """A daemon must be restartable from its config alone."""
    cfg = FleetConfig(session_id="s1", root=str(tmp_path), agents=7,
                      levels=(1, 2, 4, 8), seeds=("try a thing",), dry_run=True)
    p = tmp_path / "fleet.json"
    cfg.save(p)
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

    check(FleetConfig(session_id="s", dry_run=True))            # fakes need nothing
    full = {"bill_per_1k": 14.96, "quality": {"gsm8k": 0.66},
            "screen": {"bill_per_1k": 17.3}}
    check(FleetConfig(session_id="s", baseline=full))
    for missing in ("bill_per_1k", "quality", "screen"):
        with pytest.raises(SystemExit):
            check(FleetConfig(session_id="s",
                              baseline={k: v for k, v in full.items() if k != missing}))
    check(FleetConfig(session_id="s", screen_first=False,
                      baseline={k: v for k, v in full.items() if k != "screen"}))


def test_build_mode_reaches_the_proposer_with_the_tool_index(tmp_path):
    """`--mode build --bank --manager`: the proposer is built in build mode
    and reads the manager's tool index; nothing shells out at construction."""
    from harness.ideas import SqliteIdeaBank

    bank = tmp_path / "ideas.db"
    SqliteIdeaBank(bank)
    cfg = FleetConfig(session_id="s1", root=str(tmp_path / "agents"), agents=1,
                      dry_run=True, mode="build", bank=str(bank), manager=True,
                      baseline={"bill_per_1k": 12.23, "quality": {"gsm8k": 0.69},
                                "screen": {"bill_per_1k": 17.3}})
    fleet, broker = build(cfg, store=SqliteSessionStore(tmp_path / "s.db"))
    try:
        assert fleet.bank is not None and fleet.manager is not None
        agent = fleet.make_agent("a00", fleet)
        assert agent.proposer.mode == "build"
        assert agent.proposer.session_tools is not None
        fleet.manager.stash.add("bench", "bench a kernel", "python tools/bench.py", "print(1)", 2)
        from harness.contracts import Brief
        assert "bench.py" in agent.proposer._brief_text(Brief(text="known"), "")
    finally:
        broker.shutdown()
    import pytest as _pt

    from harness.daemon import check
    with _pt.raises(SystemExit):
        check(FleetConfig(session_id="s2", mode="build",
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
