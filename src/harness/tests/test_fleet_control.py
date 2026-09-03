"""Operator control, end to end through the store: the transitions a
dashboard renders, the acknowledgements it waits on, and the accounting it
shows. These are the rows the TUI reads, so the tests are written against
the store and the fleet rather than the screen."""
import pathlib
import tempfile
import threading
import time

import pytest

from harness import EvalBroker, Fleet
from harness.contracts import AgentOutcome, FleetBudget, FleetSpec, Idea
from harness.contracts.session import Command, TokenUse
from harness.session import SqliteSessionStore


class _Gated:
    """An agent that sits in `run` until released, checking the control
    seam the way the real loop does, so pause and kill have somewhere to
    land between checkpoints."""

    def __init__(self, agent_id, fleet, gate):
        self.agent_id = agent_id
        self.fleet = fleet
        self.gate = gate

    def propose(self, seed=None, live_ideas=()):
        return Idea(title=f"idea {self.agent_id}", hypothesis=f"h {self.agent_id}")

    def run(self, idea, budget):
        self.fleet.report(self.agent_id, status="evaluating", activity="sweep")
        while not self.gate.is_set():
            # the real loop's checkpoint: blocks while paused, returns
            # False when stopped
            if not self.fleet.wait_if_paused(self.agent_id, timeout_s=0.05) \
                    and self.fleet.should_stop(self.agent_id):
                break
            self.gate.wait(0.02)
        return AgentOutcome(agent_id=self.agent_id, idea=idea, stop="no_progress")


@pytest.fixture
def live():
    """A one-agent fleet on a real store, ticking fast."""
    store = SqliteSessionStore(pathlib.Path(tempfile.mkdtemp()) / "s.db")
    gate = threading.Event()
    broker = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    fleet = Fleet(lambda a, f: _Gated(a, f, gate), broker, store=store,
                  session_id="ctl", tick_s=0.05)
    fleet.start(FleetSpec(fleet_budget=FleetBudget(max_agents=1, max_usd_total=100)))
    for _ in range(200):
        v = store.read("ctl")
        if v and v.agents and v.agents[0].status == "evaluating":
            break
        time.sleep(0.01)
    yield store, fleet, gate
    gate.set()
    fleet.stop()
    broker.shutdown()


def _acked(store, cid, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        c = store.command_status(cid)
        if c and c.applied_at:
            return c
        time.sleep(0.01)
    raise AssertionError(f"{cid} never acknowledged")


def test_pause_resume_kill_are_acknowledged_and_the_snapshot_agrees(live):
    """The contract with a watcher: when `applied_at` is set, the published
    snapshot already shows the new state. A dashboard that clears its
    pending mark on the acknowledgement therefore never shows an
    acknowledged pause on a row still reading "evaluating"."""
    store, _fleet, _gate = live
    cid = store.send_to("ctl", Command(kind="pause", agent_id="a00"))
    assert _acked(store, cid).result == "paused"
    assert store.read("ctl").agents[0].status == "paused"
    assert store.read("ctl").phase == "paused"            # the only agent is paused
    assert store.pending("ctl") == ()

    cid = store.send_to("ctl", Command(kind="resume", agent_id="a00"))
    assert _acked(store, cid).result == "resumed"
    # back to what it was doing, not a generic "thinking"
    assert store.read("ctl").agents[0].status == "evaluating"
    assert store.read("ctl").phase == "running"

    cid = store.send_to("ctl", Command(kind="kill", agent_id="a00"))
    assert _acked(store, cid).result == "stopping"
    assert store.read("ctl").agents[0].status == "stopping"
    for _ in range(200):                                  # the thread winds up
        if store.read("ctl").agents[0].status == "done":
            break
        time.sleep(0.01)
    assert store.read("ctl").agents[0].status == "done"


def test_a_kill_reaches_a_paused_agent(live):
    """Kill must not wait behind pause: a paused agent is blocked in
    `wait_if_paused`, and the kill has to release it."""
    store, fleet, _gate = live
    _acked(store, store.send_to("ctl", Command(kind="pause", agent_id="a00")))
    assert fleet._slots["a00"].paused
    _acked(store, store.send_to("ctl", Command(kind="kill", agent_id="a00")))
    for _ in range(200):
        if store.read("ctl").agents[0].status == "done":
            break
        time.sleep(0.01)
    assert store.read("ctl").agents[0].status == "done"


def test_commands_for_a_finished_agent_are_refused_with_a_reason(live):
    """A pause on a done agent used to flip its row to "paused" with no
    thread behind it. Now it is refused, and the acknowledgement says why,
    so the operator reads "already done" rather than retrying."""
    store, _fleet, _gate = live
    _acked(store, store.send_to("ctl", Command(kind="kill", agent_id="a00")))
    for _ in range(200):
        if store.read("ctl").agents[0].status == "done":
            break
        time.sleep(0.01)
    assert store.read("ctl").agents[0].status == "done"
    c = _acked(store, store.send_to("ctl", Command(kind="pause", agent_id="a00")))
    assert c.result == "a00 is already done"
    assert store.read("ctl").agents[0].status == "done"
    c = _acked(store, store.send_to("ctl", Command(kind="kill", agent_id="a00")))
    assert c.result == "a00 is already done"
    c = _acked(store, store.send_to("ctl", Command(kind="resume", agent_id="a00")))
    assert c.result == "a00 was not paused"
    c = _acked(store, store.send_to("ctl", Command(kind="pause", agent_id="zz")))
    assert c.result == "no such agent zz"


def test_pending_is_the_unacknowledged_rows_and_nothing_else(tmp_path):
    """`pending` is defined by `applied_at == 0`: a fresh dashboard on the
    same store sees the same marks, and a dead daemon's rows stay pending
    until someone deletes the session -- the watcher decides they are
    undeliverable, not the store."""
    from harness.contracts.session import SessionView

    store = SqliteSessionStore(tmp_path / "s.db")
    store.create(SessionView(session_id="s"))
    a = store.send_to("s", Command(kind="pause", agent_id="a01"))
    b = store.send_to("s", Command(kind="scale", value="3"))
    assert [(c.id, c.agent_id) for c in store.pending("s")] == [(a, "a01"), (b, "")]
    store.acknowledge(a, "paused")
    assert [c.id for c in store.pending("s")] == [b]
    assert store.pending("other") == ()


def test_delete_session_removes_every_row(tmp_path):
    from harness.contracts.session import SessionView

    store = SqliteSessionStore(tmp_path / "s.db")
    for sid in ("keep", "drop"):
        store.create(SessionView(session_id=sid))
        store.add_tokens(sid, "a00", TokenUse(1, 1, 1, 1))
        store.send_to(sid, Command(kind="stop"))
    removed = store.delete_session("drop")
    assert removed == {"sessions": 1, "tokens": 1, "commands": 1}
    assert store.read("drop") is None and store.tokens("drop") == {}
    assert store.pending("drop") == ()
    assert store.read("keep") is not None and store.tokens("keep")
    assert store.delete_session("drop") == {"sessions": 0, "tokens": 0, "commands": 0}


def test_dollars_are_modal_spend_and_tokens_are_tokens(live):
    """Per-agent cost moves only on `cost_delta` (evaluations and the tool
    ledger, both Modal). Token reports change the token counters -- in the
    snapshot and in the store's per-agent table, identically -- and never
    the dollars; phase time accumulates per bucket."""
    store, fleet, _gate = live
    fleet.report("a00", tokens=TokenUse(100, 20, 5000, 300))
    fleet.report("a00", tokens=TokenUse(50, 10, 1000, 0))
    fleet.report("a00", phase_delta=("edit", 30.0))
    fleet.report("a00", phase_delta=("edit", 12.5))
    fleet.report("a00", phase_delta=("wait", 7.0))
    for _ in range(100):
        v = store.read("ctl")
        if v.agents[0].tokens.input == 150 and v.agents[0].phase_s.get("wait"):
            break
        time.sleep(0.01)
    a = v.agents[0]
    assert a.cost_usd == 0.0 and v.cost_usd == 0.0
    assert a.tokens == TokenUse(150, 30, 6000, 300)
    assert v.tokens == a.tokens                       # the fleet total is the sum
    assert store.tokens("ctl")["a00"] == a.tokens      # and the table agrees
    assert a.phase_s == {"edit": 42.5, "wait": 7.0}

    fleet.report("a00", cost_delta=1.25)
    fleet.report("a00", cost_delta=0.75)
    for _ in range(100):
        v = store.read("ctl")
        if v.agents[0].cost_usd == 2.0:
            break
        time.sleep(0.01)
    assert v.agents[0].cost_usd == 2.0 and v.cost_usd == 2.0
    assert v.agents[0].tokens == TokenUse(150, 30, 6000, 300)   # untouched


def test_the_run_directory_alone_answers_the_morning_question(tmp_path):
    """`summary.json` is written while running and once more on stop, so
    the ask context for a finished run has the fleet's state and every
    idea's outcome without the daemon or the store; while it runs, a live
    snapshot takes the summary's place."""
    from harness.ask import build_context
    from harness.results import load_summary, snapshot_text

    root = tmp_path / "run"
    root.mkdir()
    gate = threading.Event()
    broker = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    fleet = Fleet(lambda a, f: _Gated(a, f, gate), broker, session_id="morn",
                  root=str(root), tick_s=0.05)
    fleet.start(FleetSpec(baseline_metrics={"bill_per_1k": 14.96},
                          fleet_budget=FleetBudget(max_agents=1, max_usd_total=100)))
    try:
        time.sleep(0.2)
        fleet.report("a00", cost_delta=3.5, tokens=TokenUse(10, 5, 1000, 0),
                     phase_delta=("edit", 120.0))
        live_ctx = build_context(root, view=fleet._snapshot())
        assert "## Fleet now (live)" in live_ctx
        assert "a00: evaluating" in live_ctx and "spend $3.50" in live_ctx
        assert "edit 2m" in live_ctx and "cache 1,000" in live_ctx
        gate.set()
    finally:
        fleet.stop("finished")
        broker.shutdown()
    doc = load_summary(root)
    assert doc["snapshot"]["phase"] == "stopped"
    assert doc["baseline"] == {"bill_per_1k": 14.96}
    assert doc["outcomes"] and doc["outcomes"][0]["stop"] == "no_progress"
    ctx = build_context(root)
    assert "## Fleet at last summary" in ctx and "## Idea outcomes" in ctx
    assert "phase stopped" in ctx and "stopped on no_progress" in ctx
    assert "Fleet now (live)" not in ctx
    assert "Modal spend $3.50" in snapshot_text(doc["snapshot"])
