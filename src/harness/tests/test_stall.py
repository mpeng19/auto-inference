"""A model call that goes quiet is cut and the attempt restarted.

build-4: an edit call hung mid-stream and the agent sat in `thinking` for
the full two-hour edit timeout. The stream reports every message, so
silence is the tell; the fleet sets the slot's `cancel` (not its `stop`),
the loop records `stalled`, resets the workspace and tries the same idea
again. Kill and stop still end the agent, and a slow call that keeps
streaming is left alone.
"""
import pathlib
import threading
import time
from dataclasses import dataclass, field

from harness import EvalBroker, Fleet, IterativeAgent, Workspace
from harness.contracts import AgentBudget, FleetBudget, FleetSpec, Idea
from harness.contracts.session import TokenUse

from .test_fleet import FakeRunner
from .test_workspace import FakeStock

P = "srt/managers/schedule_policy.py"
GOOD = "CHUNK = 16384\n\n\nclass SchedulePolicy:\n    pass\n"


@dataclass
class StallingProposer:
    """Streams two tokens, leaves half a diff, then goes silent until
    cancelled. Every later call edits properly -- unless `always` is set,
    in which case it hangs every time (for the kill test)."""
    fleet: object = None
    always: bool = False
    stream_for_s: float = 0.0        # >0: keep streaming this long instead of hanging
    calls: int = 0
    seeds: int = 0
    entered: threading.Event = field(default_factory=threading.Event)
    cancelled: list = field(default_factory=list)
    seen_activity: list = field(default_factory=list)
    clean_on_retry: list = field(default_factory=list)

    def seed(self, live_ideas, brief):
        # One idea: without a bank the fleet would otherwise claim a second
        # one the moment the first ends, and the counts below would blur.
        self.seeds += 1
        if self.seeds > 1:
            raise RuntimeError("one idea only")
        return Idea(title="hang", hypothesis="a call that hangs", targets=(P,))

    def _tok(self):
        self.fleet.report("a00", tokens=TokenUse(output=1))

    def edit(self, ws, idea, brief, attempt, history, cancel=None):
        self.calls += 1
        self.entered.set()
        if self.stream_for_s:
            end = time.time() + self.stream_for_s
            while time.time() < end:
                self._tok()
                time.sleep(0.05)
            self.cancelled.append(cancel.is_set())
            ws.edit(P, GOOD)
            return "slow but alive"
        if self.calls == 1 or self.always:
            ws.edit(P, "PARTIAL = 1\n")
            self._tok()
            time.sleep(0.02)
            self._tok()
            self.cancelled.append(cancel.wait(timeout=20))
            self.seen_activity.append(self.fleet._slots["a00"].view.activity)
            return "(cut)"
        self.clean_on_retry.append("PARTIAL" not in ws.read(P))
        ws.edit(P, GOOD)
        return "real edit"

    def study(self, ws, idea, brief, history, cancel=None):
        return "note"


def _fleet(tmp_path, stock_dir, memory, context, prop, *, stall_s, max_attempts=3):
    broker = EvalBroker(FakeRunner(delay=0.05), capacity=2)

    def factory(agent_id, fleet):
        prop.fleet = fleet
        ws = Workspace(pathlib.Path(tmp_path) / agent_id, agent_id=agent_id,
                       source=FakeStock(stock_dir))
        return IterativeAgent(agent_id=agent_id, workspace=ws, memory=memory,
                              context=context, proposer=prop, evals=fleet.evals,
                              control=fleet, baseline={"bill_per_1k": 12.23})

    fleet = Fleet(factory, broker, store=None, session_id="t1", tick_s=0.05,
                  stall_s=stall_s)
    fleet.root = str(tmp_path)
    fleet.start(FleetSpec(
        agent_budget=AgentBudget(max_attempts=max_attempts, max_usd=50, patience=3,
                                 screen_first=False, replicate_wins=False,
                                 auto_ablate=False),
        fleet_budget=FleetBudget(max_agents=1, max_concurrent_evals=2,
                                 max_usd_total=10_000, max_wall_s=60)))
    return fleet, broker


def _wait_done(fleet, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        done = fleet.state().completed
        if done:
            return done
        time.sleep(0.05)
    raise AssertionError("the agent never finished")


def test_a_silent_call_is_cut_and_the_attempt_restarted(tmp_path, stock_dir, memory, context):
    prop = StallingProposer()
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop, stall_s=0.3)
    try:
        out = _wait_done(fleet)[0]
    finally:
        fleet.stop()
        broker.shutdown()
    assert prop.cancelled == [True], "the stall watch never cancelled the call"
    assert prop.seen_activity and prop.seen_activity[0].startswith("stalled: no output for")
    assert "restarting the attempt" in prop.seen_activity[0]
    # Same idea, next attempt, from a clean workspace: the half diff is not priced.
    assert prop.calls == 2 and prop.clean_on_retry == [True]
    assert [a.failure for a in out.attempts] == ["stalled", ""]
    assert out.attempts[0].n == 0 and out.attempts[1].n == 1 and out.attempts[1].ok
    assert out.stop == "won"
    turns = list(context.read(out.attempts[1].trace_ref))
    stalled = [t for t in turns if t.kind == "error" and t.name == "stalled"]
    assert len(stalled) == 1 and stalled[0].data["waited_s"] >= 0.3
    assert stalled[0].data["phase"] == "propose"
    assert fleet._slots["a00"].view.stalls == 1
    assert fleet._stalls == 1


def test_stalls_count_against_patience(tmp_path, stock_dir, memory, context):
    prop = StallingProposer(always=True)
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop, stall_s=0.2,
                           max_attempts=10)
    try:
        out = _wait_done(fleet)[0]
    finally:
        fleet.stop()
        broker.shutdown()
    assert out.stop == "no_progress"
    assert [a.failure for a in out.attempts] == ["stalled"] * 3
    assert prop.calls == 3


def test_kill_still_ends_a_stalling_agent(tmp_path, stock_dir, memory, context, monkeypatch):
    monkeypatch.setattr(Fleet, "_cancel_calls", staticmethod(lambda root: []))
    prop = StallingProposer(always=True)
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop, stall_s=0.3,
                           max_attempts=10)
    try:
        assert prop.entered.wait(5)
        for _ in range(100):             # let at least one stall restart happen
            if prop.calls >= 2:
                break
            time.sleep(0.05)
        assert prop.calls >= 2
        assert fleet.kill_agent("a00") is True
        out = _wait_done(fleet)[0]
    finally:
        fleet.stop()
        broker.shutdown()
    assert out.stop == "stopped"
    assert "stalled" in [a.failure for a in out.attempts]
    assert prop.cancelled[-1] is True
    assert fleet._slots["a00"].view.status in ("done", "stopping")


def test_a_slow_call_that_keeps_streaming_is_not_cut(tmp_path, stock_dir, memory, context):
    prop = StallingProposer(stream_for_s=1.0)
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop, stall_s=0.3)
    try:
        out = _wait_done(fleet)[0]
    finally:
        fleet.stop()
        broker.shutdown()
    assert prop.cancelled == [False]
    assert prop.calls == 1
    assert [a.failure for a in out.attempts] == [""] and out.attempts[0].ok
    assert fleet._slots["a00"].view.stalls == 0


def test_stop_sets_cancel_and_stop_together(tmp_path, stock_dir, memory, context):
    """Kill, scale-down and stop end the agent: both events. The stall watch
    sets only `cancel`, and `should_stop` stays false."""
    prop = StallingProposer(always=True)
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop, stall_s=0)
    try:
        assert prop.entered.wait(5)
        slot = fleet._slots["a00"]
        assert fleet.cancel_event("a00") is slot.cancel
        assert slot.cancel is not slot.stop
        assert not slot.cancel.is_set() and not fleet.should_stop("a00")
        fleet._watch_stalls(time.time() + 10)      # forced: as if 10 s of silence at stall_s=0 -> off
        assert not slot.cancel.is_set(), "stall_s=0 must disable the watch"
        fleet.stall_s = 5.0
        fleet._watch_stalls(time.time() + 10)
        assert slot.cancel.is_set() and not fleet.should_stop("a00")
        fleet.stop("operator")
        assert slot.stop.is_set() and slot.cancel.is_set()
    finally:
        broker.shutdown()


def test_host_sleep_is_not_a_stall(tmp_path, stock_dir, memory, context):
    prop = StallingProposer(always=True)
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop, stall_s=3600)
    try:
        assert prop.entered.wait(5)
        slot = fleet._slots["a00"]
        slot.last_activity = time.time() - 7200
        fleet.note_host_sleep(7000)
        fleet._watch_stalls(time.time())
        assert not slot.cancel.is_set()
    finally:
        fleet.stop()
        broker.shutdown()
