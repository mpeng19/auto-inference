"""The whole loop, end to end, with no GPU and no model.

If this passes, the wiring is right: agents seed diverse ideas, propose diffs,
get gated on evaluation, write experiments and edges into memory, read each
other's failures back, and stop for the right reasons.
"""
import pathlib
import threading
from dataclasses import dataclass, field

import pytest

from harness import EvalBroker, Fleet, IterativeAgent, Workspace
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
class FakeRunner:
    """Deterministic 'measurements'. `mode` picks the shape of the run."""
    mode: str = "improving"
    delay: float = 0.0
    calls: list = field(default_factory=list)
    tiers: list = field(default_factory=list)
    concurrent: int = 0
    peak: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, req):
        import time
        with self._lock:
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
        try:
            if self.delay:
                time.sleep(self.delay)
            with self._lock:
                self.calls.append(req.stack.digest)
                self.tiers.append(req.tier)
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


def _agent_factory(tmp_path, stock_dir, memory, context, evals, proposer=None):
    def make(agent_id, fleet):
        ws = Workspace(pathlib.Path(tmp_path) / agent_id, agent_id=agent_id,
                       source=FakeStock(stock_dir))
        return IterativeAgent(
            agent_id=agent_id, workspace=ws, memory=memory, context=context,
            proposer=proposer or ScriptedProposer(),
            evals=(fleet.evals if fleet is not None else evals),
            baseline={"bill_per_1k": 12.23})
    return make


def test_fleet_satisfies_the_contract():
    b = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    try:
        assert isinstance(Fleet(lambda a, f: None, b), OrchestrationService)
    finally:
        b.shutdown()


def test_end_to_end_two_agents(tmp_path, stock_dir, memory, context):
    run = FakeRunner()
    broker = EvalBroker(run, capacity=2)
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, broker), broker)
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
    broker.shutdown()
    assert run.calls, "no evaluation ever ran"
    assert st.cost_usd > 0
    # every experiment reached memory
    br = memory.recall(Recall(intent="tune", k=20))
    assert br.hits


def test_eval_queue_caps_gpu_concurrency(tmp_path, stock_dir, memory, context):
    """Attempts rent GPUs, so the queue depth is the bill."""
    run = FakeRunner(delay=0.02)
    broker = EvalBroker(run, capacity=2)
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, broker), broker)
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
    broker.shutdown()
    assert run.peak <= 2, f"ran {run.peak} concurrent evaluations against a cap of 2"


def test_duplicate_ideas_are_rejected():
    """No agent can tell it is duplicating; only the fleet holding all of them."""
    f = Fleet(lambda a, fl: None, EvalBroker(lambda r: (True, {}, ""), capacity=1))
    f._spec = FleetSpec()
    a = Idea(title="chunk", hypothesis="raise chunked prefill size for throughput")
    b = Idea(title="chunk2", hypothesis="raise chunked prefill size for throughput")
    assert f.claim_idea("a01", a) is not None
    assert f.claim_idea("a02", b) is None
    c = Idea(title="evict", hypothesis="change radix eviction to lfu ordering")
    assert f.claim_idea("a03", c) is not None


def test_agent_stops_when_nothing_improves(tmp_path, stock_dir, memory, context):
    broker = EvalBroker(FakeRunner(mode="flat"), capacity=2)
    make = _agent_factory(tmp_path, stock_dir, memory, context, broker)
    agent = make("a01", None)
    out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                    AgentBudget(max_attempts=6, patience=2))
    assert out.stop == "no_progress"
    assert len(out.attempts) <= 3, "should not spend the whole budget on a dead idea"


def test_infra_failures_retry_and_hypothesis_failures_do_not(
        tmp_path, stock_dir, memory, context):
    broker = EvalBroker(FakeRunner(mode="infra_once"), capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker)("a01", None)
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

    broker = EvalBroker(FakeRunner(), capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker,
                           proposer=Wanderer())("a01", None)
    out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                    AgentBudget(max_attempts=5, divergence_threshold=0.5))
    assert out.stop == "diverged"


def test_agents_read_each_others_failures(tmp_path, stock_dir, memory, context):
    """The reason memory is shared at all."""
    broker = EvalBroker(FakeRunner(mode="flat"), capacity=2)
    p1 = ScriptedProposer()
    a1 = _agent_factory(tmp_path, stock_dir, memory, context, broker, proposer=p1)("a01", None)
    a1.run(Idea(title="chunk", hypothesis="tune chunk sizing", targets=(P,)),
           AgentBudget(max_attempts=2, patience=2))

    p2 = ScriptedProposer()
    a2 = _agent_factory(tmp_path, stock_dir, memory, context, broker, proposer=p2)("a02", None)
    a2.run(Idea(title="chunk", hypothesis="tune chunk sizing", targets=(P,)),
           AgentBudget(max_attempts=1, patience=1))
    assert any("tune chunk sizing" in b for b in p2.seen_briefs), \
        "the second agent never saw the first agent's work"


@pytest.mark.parametrize("mode", ["improving", "flat"])
def test_traces_are_written_for_every_attempt(tmp_path, stock_dir, memory, context, mode):
    broker = EvalBroker(FakeRunner(mode=mode), capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker)("a01", None)
    agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
              AgentBudget(max_attempts=2, patience=2))
    stats = context.stats(agent_id="a01")
    assert stats["traces"] == 1 and stats["turns"] > 0


# ── the throughput property: agents must not stall on GPUs ────────────────

def test_agents_keep_working_while_gpus_are_saturated(tmp_path, stock_dir,
                                                      memory, context):
    """Ten agents, two GPU slots: nobody should be blocked from *proposing*.

    This is the property the queue exists for. With a blocking semaphore the
    eight agents without a slot did nothing at all; here they propose, submit,
    study, and only then wait on a result that is already in flight.
    """
    run = FakeRunner(delay=0.05)
    broker = EvalBroker(run, capacity=2)
    submitted = []

    class Watcher(ScriptedProposer):
        def edit(self, ws, idea, brief, attempt, history):
            # Distinct per agent: this test is about parallel throughput, and
            # identical edits would (correctly) collapse into one GPU run --
            # which `test_identical_proposals_collapse_fleet_wide` covers.
            # Varies by idea as well as attempt: an agent that wins and picks
            # up a fresh idea would otherwise re-propose attempt 0 byte for
            # byte and be (correctly) deduped, which is not what this measures.
            ws.edit(P, f"CHUNK = {8192 * (attempt + 2)}  # {ws.agent_id} {idea.id}\n"
                       f"\n\nclass SchedulePolicy:\n    pass\n")
            return f"raise CHUNK for {ws.agent_id}"

        def study(self, ws, idea, brief, history):
            # Sampled while this agent's own evaluation is in flight: the
            # agent reached here *without* waiting for a GPU, which is the
            # property under test.
            st = broker.stats()
            submitted.append(st.running + st.queued)
            return "read the queue while waiting"

    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, broker,
                                 proposer=Watcher()), broker)
    fleet.start(FleetSpec(
        agent_budget=AgentBudget(max_attempts=2, max_usd=3, patience=2,
                                 screen_first=False),
        fleet_budget=FleetBudget(max_agents=10, max_concurrent_evals=2,
                                 max_usd_total=20, max_wall_s=20)))
    import time
    deadline = time.time() + 12
    while time.time() < deadline and fleet.state().cost_usd < 20:
        time.sleep(0.05)
    fleet.stop()
    broker.shutdown()

    st = broker.stats()
    assert run.peak <= 2, "GPU cap violated"
    assert st.completed >= 6, f"only {st.completed} evaluations in 12s"
    # Every `study` call happened with work in flight, i.e. the agent got past
    # submit without waiting for a GPU. With the old blocking semaphore this
    # code path did not exist: an agent without a slot did nothing at all.
    assert submitted, "study never ran; agents are not working while waiting"
    assert min(submitted) > 0, \
        "an agent studied with no evaluation in flight -- it was not overlapped"


def test_screen_first_avoids_full_sweeps_on_dead_candidates(
        tmp_path, stock_dir, memory, context):
    """The largest throughput lever: most candidates die in the cheap tier."""
    run = FakeRunner(mode="flat")           # never beats the baseline
    broker = EvalBroker(run, capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker)("a01", None)
    agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
              AgentBudget(max_attempts=3, patience=3, screen_first=True))
    broker.shutdown()
    assert run.tiers, "nothing ran"
    assert "full" not in run.tiers, \
        f"a flat candidate was promoted to a full sweep: {run.tiers}"


def test_a_promising_screen_is_promoted_to_a_full_sweep(
        tmp_path, stock_dir, memory, context):
    run = FakeRunner(mode="improving")      # every run beats the last
    broker = EvalBroker(run, capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker)("a01", None)
    agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
              AgentBudget(max_attempts=2, patience=2, screen_first=True))
    broker.shutdown()
    assert "screen" in run.tiers and "full" in run.tiers
    assert run.tiers.index("screen") < run.tiers.index("full")


def test_idle_time_is_reported_so_stalling_is_visible(
        tmp_path, stock_dir, memory, context):
    run = FakeRunner(delay=0.05)
    broker = EvalBroker(run, capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker)("a01", None)
    out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                    AgentBudget(max_attempts=1, patience=1, screen_first=False))
    broker.shutdown()
    assert out.idle_s >= 0.0
    assert out.attempts[0].queued_s >= 0.0


def test_identical_proposals_collapse_fleet_wide(tmp_path, stock_dir, memory, context):
    """Ten agents seeded from one baseline often converge on the same edit.

    Observed while writing the test above: twenty proposals, one GPU run. That
    is a whole fleet-hour of rented capacity not spent, and it comes free from
    the stack digest being a content hash.
    """
    run = FakeRunner(delay=0.02)
    broker = EvalBroker(run, capacity=3)
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, broker), broker)
    fleet.start(FleetSpec(
        agent_budget=AgentBudget(max_attempts=2, max_usd=3, patience=2,
                                 screen_first=False),
        fleet_budget=FleetBudget(max_agents=8, max_concurrent_evals=3,
                                 max_usd_total=6, max_wall_s=15)))
    import time
    deadline = time.time() + 8
    while time.time() < deadline and broker.stats().deduped < 5:
        time.sleep(0.05)
    fleet.stop()
    broker.shutdown()
    st = broker.stats()
    assert st.deduped > st.completed, \
        f"expected identical proposals to collapse: {st}"
