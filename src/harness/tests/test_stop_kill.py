"""Stop and kill reach a running model call, not just the next checkpoint.

build-4: the operator pressed stop at 16:19 and the agents kept writing for
the rest of their two-hour edits, because stop only set a flag the loop read
between attempts. Now the slot's stop event is the proposer's `cancel`.
"""
import pathlib
import threading
import time
from dataclasses import dataclass, field

from harness import EvalBroker, Fleet, IterativeAgent, Workspace
from harness.contracts import AgentBudget, FleetBudget, FleetSpec, Idea

from .test_workspace import FakeStock

P = "srt/managers/schedule_policy.py"


@dataclass
class BlockingProposer:
    """An edit that runs until cancelled, like a two-hour claude call."""
    entered: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False

    def seed(self, live_ideas, brief):
        return Idea(title="long", hypothesis="edit for hours", targets=(P,))

    def edit(self, ws, idea, brief, attempt, history, cancel=None):
        self.entered.set()
        if cancel is not None:
            self.cancelled = cancel.wait(timeout=20)
        else:
            time.sleep(20)
        return "(cut off)"

    def study(self, ws, idea, brief, history, cancel=None):
        return "note"


def _fleet(tmp_path, stock_dir, memory, context, proposer, agents=1, runner_delay=0.05):
    from .test_fleet import FakeRunner

    broker = EvalBroker(FakeRunner(delay=runner_delay), capacity=2)

    def factory(agent_id, fleet):
        ws = Workspace(pathlib.Path(tmp_path) / agent_id, agent_id=agent_id,
                       source=FakeStock(stock_dir))
        return IterativeAgent(agent_id=agent_id, workspace=ws, memory=memory,
                              context=context, proposer=proposer,
                              evals=fleet.evals if fleet is not None else broker,
                              control=fleet, baseline={"bill_per_1k": 12.23})

    fleet = Fleet(factory, broker, store=None, session_id="t1", tick_s=0.05)
    fleet.root = str(tmp_path)
    fleet.start(FleetSpec(
        agent_budget=AgentBudget(max_attempts=3, max_usd=5, patience=3, screen_first=False),
        fleet_budget=FleetBudget(max_agents=agents, max_concurrent_evals=2,
                                 max_usd_total=10_000, max_wall_s=60)))
    return fleet, broker


def test_stop_cancels_the_running_model_call_within_a_second(tmp_path, stock_dir, memory, context):
    prop = BlockingProposer()
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop)
    try:
        assert prop.entered.wait(5), "the edit never started"
        t0 = time.time()
        fleet.stop("operator")
        assert time.time() - t0 < 5.0, "stop waited for the call to end on its own"
        assert prop.cancelled is True
        assert not fleet.state().running
    finally:
        broker.shutdown()


def test_kill_cancels_the_agents_gpu_calls_too(tmp_path, stock_dir, memory, context, monkeypatch):
    prop = BlockingProposer()
    cancelled_roots = []
    monkeypatch.setattr(Fleet, "_cancel_calls",
                        staticmethod(lambda root: cancelled_roots.append(root) or []))
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop)
    try:
        assert prop.entered.wait(5)
        assert fleet.kill_agent("a00") is True
        for _ in range(100):
            if prop.cancelled:
                break
            time.sleep(0.05)
        assert prop.cancelled is True
        assert cancelled_roots == [str(pathlib.Path(tmp_path) / "a00")]
        assert fleet.should_stop("a00")
    finally:
        fleet.stop()
        broker.shutdown()


def test_a_cancelled_edit_ends_the_idea_as_stopped_not_priced(tmp_path, stock_dir, memory, context):
    """What the model wrote before the cut is not sent to a GPU."""
    prop = BlockingProposer()
    fleet, broker = _fleet(tmp_path, stock_dir, memory, context, prop)
    try:
        assert prop.entered.wait(5)
        fleet.stop("operator")
        done = fleet.state().completed
        assert done and all(o.stop == "stopped" and not o.attempts for o in done)
    finally:
        broker.shutdown()
