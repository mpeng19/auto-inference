"""The whole loop, end to end, with no GPU and no model.

If this passes, the wiring is right: agents seed diverse ideas, propose diffs,
get gated on evaluation, write experiments and edges into memory, read each
other's failures back, and stop for the right reasons.
"""
import pathlib
import threading
from dataclasses import dataclass, field

import pytest

from harness import Fleet, IterativeAgent, Workspace
from harness.contracts import AgentBudget, FleetBudget, FleetSpec, Idea, Recall
from harness.contracts.orchestration import OrchestrationService

from .test_workspace import FakeStock

P = "srt/managers/schedule_policy.py"


@dataclass
class ScriptedProposer:
    """Stands in for the model. Bumps a constant a little further each attempt."""
    titles: list[str] = field(default_factory=lambda: ["chunk", "evict", "batch"])
    seen_briefs: list[str] = field(default_factory=list)
    _n: int = 0

    def seed(self, live_ideas, brief):
        self.seen_briefs.append(brief.text)
        t = self.titles[self._n % len(self.titles)]
        self._n += 1
        return Idea(title=t, hypothesis=f"tune {t} to lower cost per output token",
                    targets=(P,))

    def edit(self, ws, idea, brief, attempt, history):
        self.seen_briefs.append(brief.text)
        ws.edit(P, f"CHUNK = {8192 * (attempt + 2)}\n\n\nclass SchedulePolicy:\n    pass\n")
        return f"raise CHUNK on attempt {attempt}"


@dataclass
class FakeEvaluator:
    """Deterministic 'measurements'. `mode` picks the shape of the run."""
    mode: str = "improving"
    calls: list = field(default_factory=list)
    concurrent: int = 0
    peak: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def evaluate(self, stack, run_dir):
        with self._lock:
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
        try:
            self.calls.append(stack.digest)
            n = len(self.calls)
            if self.mode == "infra_once" and n == 1:
                return False, {"error": "server never started"}, "infra"
            if self.mode == "flat":
                return True, {"bill_per_1k": 12.23, "n_star": 12, "cost_usd": 1.0}, ""
            return True, {"bill_per_1k": 12.23 - n, "n_star": 12 + n,
                          "cost_usd": 1.0}, ""
        finally:
            with self._lock:
                self.concurrent -= 1


def _agent_factory(tmp_path, stock_dir, memory, context, evaluator, proposer=None):
    def make(agent_id, fleet):
        ws = Workspace(pathlib.Path(tmp_path) / agent_id, agent_id=agent_id,
                       source=FakeStock(stock_dir))
        return IterativeAgent(
            agent_id=agent_id, workspace=ws, memory=memory, context=context,
            proposer=proposer or ScriptedProposer(), evaluator=evaluator,
            fleet=fleet, baseline={"bill_per_1k": 12.23})
    return make


def test_fleet_satisfies_the_contract():
    assert isinstance(Fleet(lambda a, f: None), OrchestrationService)


def test_end_to_end_two_agents(tmp_path, stock_dir, memory, context):
    ev = FakeEvaluator()
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, ev))
    fleet.start(FleetSpec(
        agent_budget=AgentBudget(max_attempts=2, max_usd=5, patience=2),
        fleet_budget=FleetBudget(max_agents=2, max_concurrent_evals=1,
                                 max_usd_total=8, max_wall_s=60)))
    import time
    for _ in range(200):
        if fleet.state().cost_usd >= 8 or not fleet.state().running:
            break
        time.sleep(0.05)
    st = fleet.stop()
    assert ev.calls, "no evaluation ever ran"
    assert st.cost_usd > 0
    # every experiment reached memory
    br = memory.recall(Recall(intent="tune", k=20))
    assert br.hits


def test_eval_slots_cap_gpu_concurrency(tmp_path, stock_dir, memory, context):
    """The one thing the orchestrator exists for: attempts rent GPUs."""
    ev = FakeEvaluator()
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, ev))
    fleet.start(FleetSpec(
        agent_budget=AgentBudget(max_attempts=3, max_usd=5, patience=3),
        fleet_budget=FleetBudget(max_agents=6, max_concurrent_evals=2,
                                 max_usd_total=12, max_wall_s=60)))
    import time
    for _ in range(200):
        if fleet.state().cost_usd >= 12 or not fleet.state().running:
            break
        time.sleep(0.05)
    fleet.stop()
    assert ev.peak <= 2, f"ran {ev.peak} concurrent evaluations against a cap of 2"


def test_duplicate_ideas_are_rejected():
    """No agent can tell it is duplicating; only the fleet holding all of them."""
    f = Fleet(lambda a, fl: None)
    f._spec = FleetSpec()
    a = Idea(title="chunk", hypothesis="raise chunked prefill size for throughput")
    b = Idea(title="chunk2", hypothesis="raise chunked prefill size for throughput")
    assert f.claim_idea("a01", a) is not None
    assert f.claim_idea("a02", b) is None
    c = Idea(title="evict", hypothesis="change radix eviction to lfu ordering")
    assert f.claim_idea("a03", c) is not None


def test_agent_stops_when_nothing_improves(tmp_path, stock_dir, memory, context):
    ev = FakeEvaluator(mode="flat")
    make = _agent_factory(tmp_path, stock_dir, memory, context, ev)
    agent = make("a01", None)
    out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                    AgentBudget(max_attempts=6, patience=2))
    assert out.stop == "no_progress"
    assert len(out.attempts) <= 3, "should not spend the whole budget on a dead idea"


def test_infra_failures_retry_and_hypothesis_failures_do_not(
        tmp_path, stock_dir, memory, context):
    ev = FakeEvaluator(mode="infra_once")
    agent = _agent_factory(tmp_path, stock_dir, memory, context, ev)("a01", None)
    out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                    AgentBudget(max_attempts=3, patience=3))
    assert out.attempts[0].failure == "infra"
    # the retry reused the same attempt number rather than consuming one
    assert out.attempts[0].n == out.attempts[1].n


def test_divergence_stops_an_agent_that_wandered(tmp_path, stock_dir, memory, context):
    class Wanderer(ScriptedProposer):
        def edit(self, ws, idea, brief, attempt, history):
            ws.edit("srt/mem_cache/radix_cache.py", "class RadixCache:\n    evict='lfu'\n")
            return "rewrote something else entirely"

    ev = FakeEvaluator()
    agent = _agent_factory(tmp_path, stock_dir, memory, context, ev,
                           proposer=Wanderer())("a01", None)
    out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                    AgentBudget(max_attempts=5, divergence_threshold=0.5))
    assert out.stop == "diverged"


def test_agents_read_each_others_failures(tmp_path, stock_dir, memory, context):
    """The reason memory is shared at all."""
    ev = FakeEvaluator(mode="flat")
    p1 = ScriptedProposer()
    a1 = _agent_factory(tmp_path, stock_dir, memory, context, ev, proposer=p1)("a01", None)
    a1.run(Idea(title="chunk", hypothesis="tune chunk sizing", targets=(P,)),
           AgentBudget(max_attempts=2, patience=2))

    p2 = ScriptedProposer()
    a2 = _agent_factory(tmp_path, stock_dir, memory, context, ev, proposer=p2)("a02", None)
    a2.run(Idea(title="chunk", hypothesis="tune chunk sizing", targets=(P,)),
           AgentBudget(max_attempts=1, patience=1))
    assert any("tune chunk sizing" in b for b in p2.seen_briefs), \
        "the second agent never saw the first agent's work"


@pytest.mark.parametrize("mode", ["improving", "flat"])
def test_traces_are_written_for_every_attempt(tmp_path, stock_dir, memory, context, mode):
    ev = FakeEvaluator(mode=mode)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, ev)("a01", None)
    agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
              AgentBudget(max_attempts=2, patience=2))
    stats = context.stats(agent_id="a01")
    assert stats["traces"] == 1 and stats["turns"] > 0
