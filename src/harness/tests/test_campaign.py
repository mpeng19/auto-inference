"""Fleets in sequence: each round's best publishable result is the next
round's base, its report the next baseline, and the chain stops at the
target, when the rounds run out, when a round finds nothing, or when the
operator says so. The fleets are faked; the selection code is real.
"""
import json
import pathlib
from dataclasses import dataclass, field

import pytest

from harness import campaign as cp
from harness.context import JsonlContext
from harness.contracts import Experiment
from harness.contracts.context import TraceMeta, Turn
from harness.daemon import FleetConfig
from harness.memory import SqliteMemory

BASE = {"bill_per_1k": 12.0, "quality": {"gsm8k": 0.70}, "screen": {"bill_per_1k": 15.0}}


@dataclass
class FakeRound:
    """Stands in for `daemon.run`: leaves behind what a fleet leaves behind.

    `factor` scales the bill per round (0.9: a 10% win each time); `pub`
    says whether an ablation is written (publishable) -- a list is one
    entry per round; `wins` False makes a round find nothing; `ended` is
    what the fake reports as the reason the fleet stopped.
    """
    factor: float = 0.9
    pub: object = True
    wins: bool = True
    ended: str = "finished"
    screen: bool = True
    configs: list = field(default_factory=list)

    def __call__(self, cfg: FleetConfig) -> str:
        from simulator import InferenceStack

        self.configs.append(cfg)
        n = len(self.configs)
        root = pathlib.Path(cfg.root)
        pub = self.pub[n - 1] if isinstance(self.pub, (list, tuple)) else self.pub
        overlay = InferenceStack(files={f"srt/round{n}.py": f"# round {n}\n"},
                                 label=f"a00: round {n}")
        stack = (InferenceStack.compose(InferenceStack.load(cfg.base), overlay)
                 if cfg.base else overlay)
        bill = round(cfg.baseline["bill_per_1k"] * self.factor, 4)
        hyp = f"idea {n}: a mechanism tried in round {n}"
        # memory: the experiment the loop would record
        SqliteMemory(root / "memory.db").record(Experiment(
            agent_id="a00", idea_id=f"idea_{n}", hypothesis=hyp,
            stack_digest=stack.digest, verdict="win" if self.wins else "neutral",
            metrics={"bill_per_1k": bill, "tier": "full", "n_star": 12,
                     "quality": [{"suite": "gsm8k", "accuracy": 0.7 + n / 100}]},
            baseline_metrics=dict(cfg.baseline)))
        # traces: a screen, then two full prices (the replicate)
        ctx = JsonlContext(root / "traces", session_id=cfg.session_id)
        tid = ctx.open(TraceMeta(agent_id="a00", idea_id=f"idea_{n}", model="fake"))
        if self.screen:
            ctx.append(tid, Turn(kind="eval_result", name=stack.digest,
                                 data={"tier": "screen", "bill_per_1k": bill * 1.3}))
        for _ in range(2 if self.wins else 1):
            ctx.append(tid, Turn(kind="eval_result", name=stack.digest,
                                 data={"tier": "full", "bill_per_1k": bill}))
        ctx.close(tid, outcome="won")
        # the run directory the next round would load
        for name in ("attempt-000", "attempt-000-rep1"):
            d = root / "a00" / "runs" / name
            d.mkdir(parents=True)
            (d / "stack.json").write_text(json.dumps(stack.as_dict()))
            (d / "result.json").write_text(json.dumps({"ok": True, "bill_per_1k": bill}))
        if pub and self.wins:
            a = root / "a00" / "ablations" / "0"
            a.mkdir(parents=True)
            (a / "ablation.json").write_text(json.dumps({
                "stack_digest": stack.digest, "explains": True, "ts": 1.0,
                "as_is": {"bill_per_1k": bill}, "disabled": {"bill_per_1k": bill * 1.1}}))
        (root / "summary.json").write_text(json.dumps({"outcomes": [
            {"idea_id": f"idea_{n}", "title": f"idea {n}", "hypothesis": hyp, "stop": "won"},
            {"idea_id": f"idea_{n}b", "title": "dud", "hypothesis": f"dud {n}", "stop": "no_progress"}]}))
        return self.ended


def _cfg(tmp_path, rounds=3, target=10.0, **fleet):
    from dataclasses import asdict

    tmpl = FleetConfig(session_id="camp", dry_run=True, fake_agents=True,
                       baseline=dict(BASE), **fleet)
    return cp.CampaignConfig(name="camp", root=str(tmp_path / "camp"), rounds=rounds,
                             target=target, fleet=asdict(tmpl))


def test_three_rounds_chain_bases_and_baselines(tmp_path):
    fake = FakeRound()
    state = cp.drive(_cfg(tmp_path), run_round=fake, log=lambda *a, **k: None)
    assert [c.session_id for c in fake.configs] == ["camp-r1", "camp-r2", "camp-r3"]
    assert [c.root for c in fake.configs] == [str(tmp_path / "camp" / f"r{n}") for n in (1, 2, 3)]
    r1, r2, r3 = fake.configs
    assert r1.base == "" and r1.baseline == BASE
    # round 2 builds on round 1's winning run and is judged against its report
    assert r2.base == str(tmp_path / "camp" / "r1" / "a00" / "runs" / "attempt-000")
    assert r2.base_digest and r2.base_label == "a00: round 1"
    assert r2.base_idea == "idea 1: a mechanism tried in round 1"
    assert r2.baseline["bill_per_1k"] == pytest.approx(10.8)
    assert r2.baseline["screen"]["bill_per_1k"] == pytest.approx(10.8 * 1.3)   # the screen attempt
    assert r2.baseline["quality"] == {"gsm8k": 0.71}
    assert r3.base == str(tmp_path / "camp" / "r2" / "a00" / "runs" / "attempt-000")
    assert r3.baseline["bill_per_1k"] == pytest.approx(10.8 * 0.9)
    # every idea tried so far, hypotheses and ids, is kept away from
    assert "idea 1: a mechanism tried in round 1" in r2.avoid and "idea_1" in r2.avoid
    assert "dud 1" in r2.avoid
    assert {"idea 1: a mechanism tried in round 1", "idea 2: a mechanism tried in round 2",
            "idea_2"} <= set(r3.avoid)
    assert r1.avoid == ()
    # the record
    assert state["status"] == "stopped" and state["stop_reason"] == "rounds exhausted"
    assert len(state["rounds"]) == 3
    assert state["cumulative_gain"] == pytest.approx(12.0 / (12.0 * 0.9 ** 3), abs=1e-3)
    assert [c["hypothesis"] for c in state["chain"]] == [
        f"idea {n}: a mechanism tried in round {n}" for n in (1, 2, 3)]
    assert all(not c["fell_back"] for c in state["chain"])
    assert [r["base"] for r in state["rounds"]] == ["", r2.base, r3.base]
    assert [r["baseline"]["bill_per_1k"] for r in state["rounds"]] == [
        12.0, pytest.approx(10.8), pytest.approx(9.72)]
    assert state["rounds"][0]["best"]["pub"] == "yes"
    assert state["rounds"][0]["best"]["run_dir"] == r2.base
    on_disk = json.loads((tmp_path / "camp" / "campaign.json").read_text())
    assert on_disk["cumulative_gain"] == state["cumulative_gain"]
    assert on_disk["current_session"] == ""
    # each round's directory names the process running it
    assert (tmp_path / "camp" / "r1" / "daemon.pid").is_file()
    assert (tmp_path / "camp" / "r1" / "fleet.json").is_file()


def test_the_target_stops_it_early(tmp_path):
    fake = FakeRound(factor=0.8)
    state = cp.drive(_cfg(tmp_path, rounds=5, target=1.5), run_round=fake,
                     log=lambda *a, **k: None)
    assert len(fake.configs) == 2                # 1.25x after one, 1.5625x after two
    assert state["cumulative_gain"] == pytest.approx(1.5625)
    assert state["stop_reason"].startswith("target 1.5x reached")


def test_nothing_publishable_falls_back_and_says_so(tmp_path):
    fake = FakeRound(pub=[False, True])
    state = cp.drive(_cfg(tmp_path, rounds=2), run_round=fake, log=lambda *a, **k: None)
    r1 = state["rounds"][0]["best"]
    assert r1["fell_back"] is True and r1["pub"] == "no-ablation"
    assert "fell back to the best replicated win" in r1["note"]
    assert state["chain"][0]["fell_back"] is True
    # it still chains: round 2 starts from that win
    assert fake.configs[1].base.endswith("r1/a00/runs/attempt-000")
    assert state["rounds"][1]["best"]["fell_back"] is False
    assert "fallback" in cp.status_text(state)


def test_a_round_that_finds_nothing_ends_the_campaign(tmp_path):
    fake = FakeRound(wins=False)
    state = cp.drive(_cfg(tmp_path, rounds=3), run_round=fake, log=lambda *a, **k: None)
    assert len(fake.configs) == 1
    assert state["rounds"][0]["best"] is None
    assert state["stop_reason"] == "round 1 produced no replicated win"
    assert len(state["rounds"][0]["tried"]) == 2
    assert state["cumulative_gain"] is None
    assert "no replicated win" in cp.status_text(state)


def test_an_operator_stop_on_the_round_ends_the_campaign_after_it(tmp_path):
    fake = FakeRound(ended="operator")
    state = cp.drive(_cfg(tmp_path, rounds=3), run_round=fake, log=lambda *a, **k: None)
    assert len(fake.configs) == 1
    assert state["stop_reason"] == "operator stopped round 1"
    assert state["rounds"][0]["best"] is not None      # the round's result is still recorded


def test_campaign_stop_marker_ends_it_between_rounds(tmp_path):
    fake = FakeRound()
    marker = tmp_path / "camp" / cp.STOP_FILE

    def run_then_mark(cfg):
        why = fake(cfg)
        marker.write_text("1")
        return why

    state = cp.drive(_cfg(tmp_path, rounds=3), run_round=run_then_mark,
                     log=lambda *a, **k: None)
    assert len(fake.configs) == 1
    assert state["stop_reason"] == "campaign stop after round 1"


def test_screen_baseline_is_scaled_when_no_screen_attempt_exists(tmp_path):
    fake = FakeRound(screen=False)
    cp.drive(_cfg(tmp_path, rounds=2), run_round=fake, log=lambda *a, **k: None)
    r2 = fake.configs[1]
    # full 10.8, scaled by the fleet's 15/12 screen/full ratio
    assert r2.baseline["screen"]["bill_per_1k"] == pytest.approx(10.8 * 15 / 12)


def test_config_round_trips(tmp_path):
    cfg = _cfg(tmp_path, rounds=4, target=3.0, agents=2)
    p = tmp_path / "c.json"
    cfg.save(p)
    back = cp.CampaignConfig.load(p)
    assert back.rounds == 4 and back.target == 3.0 and back.name == "camp"
    assert back.template().agents == 2 and back.template().levels == (4, 8, 12, 16, 24)


# ── the CLI ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))


def test_cli_start_writes_the_config_and_spawns_the_driver(tmp_path, monkeypatch, capsys):
    import harness.cli as cli
    from harness.cli import main as cli_main

    spawned = []

    class Proc:
        pid = 4242

    monkeypatch.setattr(cli.subprocess, "Popen", lambda argv, **k: spawned.append(argv) or Proc())
    root = tmp_path / "camp"
    assert cli_main(["--store", str(tmp_path / "s.db"), "--session", "camp", "campaign", "start",
                     "--root", str(root), "--rounds", "2", "--target", "1.5",
                     "--dry-run", "--fake-agents", "--agents", "2", "--stall-minutes", "5",
                     "--no-auto-ablate"]) == 0
    out = capsys.readouterr().out
    assert "camp-r1 .. camp-r2" in out and "4242" in out
    assert "harness.campaign" in " ".join(map(str, spawned[0]))
    assert (root / "daemon.pid").read_text() == "4242"
    cfg = cp.CampaignConfig.load(root / cp.CONFIG_FILE)
    assert cfg.rounds == 2 and cfg.target == 1.5 and cfg.name == "camp"
    t = cfg.template()
    assert t.agents == 2 and t.stall_minutes == 5 and t.auto_ablate is False
    assert t.dry_run and t.fake_agents
    # a second start on the same root is refused while the pid is alive
    import os
    (root / "daemon.pid").write_text(str(os.getpid()))
    with pytest.raises(SystemExit, match="already running"):
        cli_main(["--store", str(tmp_path / "s.db"), "--session", "camp", "campaign", "start",
                  "--root", str(root), "--dry-run", "--fake-agents"])


def test_cli_status_and_stop_read_campaign_json(tmp_path, capsys):
    from harness.cli import main as cli_main

    root = tmp_path / "camp"
    fake = FakeRound()
    state = cp.drive(_cfg(tmp_path, rounds=2), run_round=fake, log=lambda *a, **k: None)
    db = str(tmp_path / "s.db")
    assert cli_main(["--store", db, "campaign", "status", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "camp" in out and "r1" in out and "r2" in out and "rounds exhausted" in out
    assert cli_main(["--store", db, "campaign", "status", "--root", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cumulative_gain"] == state["cumulative_gain"]
    # stop on a finished campaign: the marker is written, nothing is sent
    assert cli_main(["--store", db, "campaign", "stop", "--root", str(root)]) == 0
    assert (root / cp.STOP_FILE).is_file()
    assert "marked stopped" in capsys.readouterr().out
    # a running one: the current round's session gets the stop command
    from harness.contracts.session import SessionView
    from harness.session import SqliteSessionStore

    store = SqliteSessionStore(db)
    live = SessionView(session_id="camp-r2", phase="running")
    store.create(live)
    store.publish(live)
    state["status"], state["current_session"] = "running", "camp-r2"
    (root / cp.STATE_FILE).write_text(json.dumps(state))
    cli_main(["--store", db, "--wait", "0.1", "campaign", "stop", "--root", str(root)])
    assert "stopping round session camp-r2" in capsys.readouterr().out
    assert [c.kind for c in store.take_commands("camp-r2")] == ["stop"]
    # no campaign here
    assert cli_main(["--store", db, "campaign", "status", "--root", str(tmp_path / "nope")]) == 1
