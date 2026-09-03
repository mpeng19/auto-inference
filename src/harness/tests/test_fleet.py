"""The whole loop, end to end, with no GPU and no model.

If this passes, the wiring is right: agents seed diverse ideas, propose diffs,
get gated on evaluation, write experiments and edges into memory, read each
other's failures back, and stop for the right reasons.
"""
import pathlib
import threading
from dataclasses import dataclass, field
from typing import ClassVar

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
            control=fleet, baseline={"bill_per_1k": 12.23})
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


def test_a_win_is_measured_twice_and_the_worse_run_counts(
        tmp_path, stock_dir, memory, context):
    """A no-op diff scored -18% on 2026-09-02 because a level on the SLO line
    passed in its sweep and not in stock's. One sweep is not a result."""
    from harness.contracts import Attempt, EvalRequest

    run = FakeRunner(mode="improving")      # every run beats the last
    broker = EvalBroker(run, capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker)("a01", None)
    out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                    AgentBudget(max_attempts=1, patience=1, screen_first=True))
    broker.shutdown()
    assert run.tiers == ["screen", "full", "full"], run.tiers
    assert len(out.attempts) == 3                 # screen, full, replicate
    assert out.best is not None
    # the replicate scored lower (better); the first, worse, run is kept
    assert out.best.metrics["bill_per_1k"] == 12.23 - 2

    stack = None
    a = EvalRequest(stack=stack, tier="full")
    assert a.dedup_key != EvalRequest(stack=stack, tier="full", replicate=1).dedup_key

    good = Attempt(ok=True, metrics={"bill_per_1k": 10.0})
    bad = Attempt(ok=True, metrics={"bill_per_1k": 12.0})
    failed = Attempt(ok=False, failure="quality")
    assert IterativeAgent._worse(good, bad) is bad
    assert IterativeAgent._worse(good, failed) is failed


def test_verdicts_have_a_noise_floor():
    """A 0.9% screen improvement was recorded as a win on night-2; the next
    agent's brief then said the idea worked. Inside the measurement's own
    noise the honest verdict is neutral."""
    from harness.contracts import Attempt

    v = IterativeAgent._verdict
    ok = lambda pct: Attempt(idea_id="i", agent_id="a", n=0, ok=True,
                             delta={"bill_per_1k_pct": pct})
    assert v(ok(-0.9)) == "neutral"
    assert v(ok(-3.0)) == "win"
    assert v(ok(+2.0)) == "neutral"
    assert v(ok(+8.0)) == "loss"
    assert v(Attempt(idea_id="i", agent_id="a", n=0, ok=False,
                           failure="quality")) == "loss"
    assert v(Attempt(idea_id="i", agent_id="a", n=0, ok=False,
                           failure="invalid_diff")) == "invalid"


def test_spend_is_visible_per_attempt_not_per_idea():
    """The first real fleet showed $0.00 after two sweeps because cost only
    reached the total when an idea ended. Spend reported as it happens must
    land immediately, and the outcome's total must not count it again."""
    import time

    from harness.contracts import AgentOutcome

    gate = threading.Event()

    class Blocking:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def propose(self, seed=None, live_ideas=()):
            return Idea(title="t", hypothesis="h")

        def run(self, idea, budget):
            gate.wait(5)
            return AgentOutcome(agent_id=self.agent_id, idea=idea,
                                stop="no_progress", cost_usd=2.0)

    broker = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    fleet = Fleet(lambda a, f: Blocking(a), broker)
    fleet.start(FleetSpec(fleet_budget=FleetBudget(max_agents=1, max_usd_total=2.0)))
    try:
        for _ in range(100):
            if fleet.state().agents and fleet.state().agents[0].status != "starting":
                break
            time.sleep(0.02)
        fleet.report("a00", cost_delta=1.5)
        st = fleet.state()
        assert st.cost_usd == 1.5 and st.agents[0].cost_usd == 1.5
        gate.set()
        for _ in range(100):
            if fleet.state().agents[0].status == "idle":
                break
            time.sleep(0.02)
        st = fleet.state()
        assert st.cost_usd == 2.0, st.cost_usd        # not 3.5
        assert st.agents[0].cost_usd == 2.0
    finally:
        fleet.stop()
        broker.shutdown()


def test_a_screen_is_judged_against_stock_at_screen_tier(
        tmp_path, stock_dir, memory, context):
    """Stock prices higher at screen tier than at full tier, so a screen
    compared with the full baseline can never be promoted."""
    run = FakeRunner(mode="flat")                      # always $12.23
    broker = EvalBroker(run, capacity=2)
    make = _agent_factory(tmp_path, stock_dir, memory, context, broker)
    agent = make("a01", None)
    agent.baseline = {"bill_per_1k": 10.0,             # full: screen loses
                      "screen": {"bill_per_1k": 14.0}}  # screen tier: it wins
    agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
              AgentBudget(max_attempts=1, patience=1, screen_first=True))
    broker.shutdown()
    assert run.tiers == ["screen", "full"], run.tiers


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


# ── live control: pause, resume, kill, scale ──────────────────────────────

def _control_fleet(tmp_path, stock_dir, memory, context, store=None, agents=3):
    run = FakeRunner(delay=0.05)
    broker = EvalBroker(run, capacity=2)
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, broker),
                  broker, store=store, session_id="t1", tick_s=0.05)
    fleet.start(FleetSpec(
        agent_budget=AgentBudget(max_attempts=3, max_usd=5, patience=3,
                                 screen_first=False),
        fleet_budget=FleetBudget(max_agents=agents, max_concurrent_evals=2,
                                 max_usd_total=10_000, max_wall_s=30)))
    return fleet, broker, run


def test_an_agent_can_be_paused_and_resumed(tmp_path, stock_dir, memory, context):
    fleet, broker, _ = _control_fleet(tmp_path, stock_dir, memory, context)
    try:
        import time
        assert fleet.pause_agent("a00") is True
        time.sleep(0.2)
        assert fleet._slots["a00"].paused
        assert fleet.resume_agent("a00") is True
        assert not fleet._slots["a00"].paused
        assert fleet.pause_agent("nope") is False
    finally:
        fleet.stop()
        broker.shutdown()


def test_killing_one_agent_leaves_the_others_running(tmp_path, stock_dir,
                                                     memory, context):
    fleet, broker, _ = _control_fleet(tmp_path, stock_dir, memory, context)
    try:
        import time
        assert fleet.kill_agent("a00") is True
        time.sleep(0.4)
        assert fleet.should_stop("a00")
        assert not fleet.should_stop("a01")
        assert fleet.state().running
    finally:
        fleet.stop()
        broker.shutdown()


def test_scaling_adds_and_removes_agents_in_flight(tmp_path, stock_dir,
                                                   memory, context):
    fleet, broker, _ = _control_fleet(tmp_path, stock_dir, memory, context, agents=2)
    try:
        import time
        assert fleet.scale(4) == 4
        time.sleep(0.2)
        assert len(fleet._slots) == 4
        fleet.scale(1)
        time.sleep(0.2)
        live = [s for s in fleet._slots.values() if not s.stop.is_set()]
        assert len(live) == 1
        # Newest go first: an agent several attempts in has more sunk cost.
        assert live[0].agent_id == "a00"
    finally:
        fleet.stop()
        broker.shutdown()


def test_commands_arrive_through_the_store(tmp_path, stock_dir, memory, context):
    """The TUI is a different process; it can only write rows."""
    from harness.contracts.session import Command
    from harness.session import SqliteSessionStore

    store = SqliteSessionStore(tmp_path / "s.db")
    fleet, broker, _ = _control_fleet(tmp_path, stock_dir, memory, context, store)
    try:
        import time
        cid = store.send_to("t1", Command(kind="pause", agent_id="a01"))
        for _ in range(60):
            time.sleep(0.05)
            c = store.command_status(cid)
            if c and c.applied_at:
                break
        assert c.result == "paused"
        assert fleet._slots["a01"].paused
        snap = store.read("t1")
        assert snap is not None and snap.session_id == "t1"
        assert snap.pid > 0, "the pid must travel with the snapshot"
    finally:
        fleet.stop()
        broker.shutdown()


def test_a_failed_agent_is_not_relabelled_done(tmp_path, stock_dir, memory, context):
    """The trailing status report used to overwrite `failed`, which hid a real
    SQLite error behind a green row on the dashboard."""
    class Broken(ScriptedProposer):
        def seed(self, live_ideas, brief):
            raise RuntimeError("boom")

    broker = EvalBroker(FakeRunner(), capacity=1)
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, broker,
                                 proposer=Broken()), broker, tick_s=0.05)
    try:
        import time
        fleet.start(FleetSpec(
            agent_budget=AgentBudget(max_attempts=1),
            fleet_budget=FleetBudget(max_agents=1, max_concurrent_evals=1,
                                     max_usd_total=5, max_wall_s=10)))
        time.sleep(0.5)
        assert fleet._slots["a00"].view.status == "failed"
        assert "boom" in fleet._slots["a00"].view.note
    finally:
        fleet.stop()
        broker.shutdown()


def test_reseeding_backs_off_rather_than_spinning(tmp_path, stock_dir, memory, context):
    """Every reseed costs a model call; a hot loop on duplication is expensive.

    Diversity is a heuristic, so after a few collisions the agent takes the
    near-duplicate and gets on with it.
    """
    class SameIdea(ScriptedProposer):
        def seed(self, live_ideas, brief):
            return Idea(title="identical", hypothesis="the very same hypothesis",
                        targets=(P,))

    submitted_by = []
    broker = EvalBroker(
        lambda req: (True, {"bill_per_1k": 12.0, "cost_usd": 0.5}, ""), capacity=2)
    # Count *submissions*, not runs: these agents all propose the identical
    # edit, so the broker correctly collapses them into one GPU run and the
    # runner would only ever see one agent id.
    _submit = broker.submit
    broker.submit = lambda req: (submitted_by.append(req.agent_id), _submit(req))[1]
    fleet = Fleet(_agent_factory(tmp_path, stock_dir, memory, context, broker,
                                 proposer=SameIdea()), broker, tick_s=0.05,
                  max_reseeds=2)
    try:
        import time
        fleet.start(FleetSpec(
            agent_budget=AgentBudget(max_attempts=1, patience=1, screen_first=False),
            # Generous on purpose: a tight budget would be exhausted by the
            # first agent through, and the starvation under test is caused by
            # reseeding, not by money.
            fleet_budget=FleetBudget(max_agents=3, max_concurrent_evals=2,
                                     max_usd_total=500, max_wall_s=10)))
        time.sleep(1.5)
        # The property is that agents do not starve: with three agents all
        # proposing the identical idea, more than one still gets to work.
        submitters = set(submitted_by)
        assert len(submitters) >= 2, (
            f"only {submitters} ever ran; the rest starved on reseeding")
    finally:
        fleet.stop()
        broker.shutdown()


def test_a_paused_agent_keeps_showing_paused(tmp_path, stock_dir, memory, context):
    """Pause is cooperative, so the agent keeps working until its next
    checkpoint. Without this the row flipped back to "evaluating" and `pause`
    looked like it had done nothing."""
    fleet, broker, _ = _control_fleet(tmp_path, stock_dir, memory, context)
    try:
        import time
        fleet.pause_agent("a00")
        # the agent reports its own progress while finishing paid work
        fleet.report("a00", status="evaluating", activity="attempt 1: full running")
        assert fleet._slots["a00"].view.status == "paused"
        assert "full running" in fleet._slots["a00"].view.activity
        fleet.resume_agent("a00")
        fleet.report("a00", status="evaluating")
        assert fleet._slots["a00"].view.status == "evaluating"
        time.sleep(0.05)
    finally:
        fleet.stop()
        broker.shutdown()


def test_a_finished_agent_can_still_report_done(tmp_path, stock_dir, memory, context):
    """The operator-state guard must not trap an agent in `stopping` forever."""
    fleet, broker, _ = _control_fleet(tmp_path, stock_dir, memory, context)
    try:
        fleet.kill_agent("a00")
        assert fleet._slots["a00"].view.status == "stopping"
        fleet.report("a00", status="done")
        assert fleet._slots["a00"].view.status == "done"
    finally:
        fleet.stop()
        broker.shutdown()


def test_a_quality_regression_is_not_a_win(tmp_path, stock_dir, memory, context):
    """An agent maximising goodput can serve worse answers faster. The price
    model cannot see it, so the evaluator has to."""
    from harness.agent.evaluator import SimulatorEvaluator

    class FakeRes:
        ok = True
        quality_regressed = True
        quality_note = "accuracy fell 12.0 points on gsm8k"
        quality = ({"suite": "gsm8k", "regressed": True},)
        bill_per_1k = 6.0          # a large apparent win
        reason = ""
        record: ClassVar = {"serving": {"n_gpu": 1, "gpu": "H100"},
                            "model_load_s": 300.0, "levels": [{"wall_s": 200.0}]}

    ev = SimulatorEvaluator()
    import simulator
    real = simulator.Simulator
    try:
        class Stub:
            def __init__(self, **kw):
                pass

            async def eval(self):
                return FakeRes()

        simulator.Simulator = Stub
        ok, metrics, failure = ev.evaluate(object(), str(tmp_path))
    finally:
        simulator.Simulator = real
    assert not ok
    # A rejected hypothesis, not infra: re-running would reproduce it.
    assert failure == "quality"
    assert "accuracy fell" in metrics["reason"]


def test_every_evaluation_reports_what_it_cost():
    """Budgets are checked against `cost_usd`. Omitting it does not make them
    approximate -- it makes them inert, and a fleet runs with no spend control.

    Caught by inspection before the first real GPU run; every fake returned a
    cost, so the whole suite passed while the real path reported nothing.
    """
    from harness.agent.evaluator import SimulatorEvaluator

    ev = SimulatorEvaluator(n_gpu=1, gpu="H100")
    rec = {"serving": {"n_gpu": 1, "gpu": "H100"}, "model_load_s": 360.0,
           "levels": [{"wall_s": 380.0} for _ in range(5)]}
    spend = ev._spend(rec)
    assert 3.5 < spend < 4.5, spend            # 0.63 h at $3.95 GPU + 16 vCPU
    # A second GPU adds a GPU's rate for the same wall time; the CPU and
    # memory the container reserves are not doubled.
    from simulator import costs
    hours = 2260 / 3600
    assert ev._spend({**rec, "serving": {"n_gpu": 2, "gpu": "H100"}}) == \
        pytest.approx(spend + hours * costs.rate("H100", "modal", allow_retail=True), rel=0.01)


def test_cost_uses_retail_not_the_serving_basis():
    """$3.00/GPU-hr is what a provider would pay to serve; $3.95 is what we are
    billed to experiment. Charging our own budget the serving basis would
    understate spend by 24%."""
    from harness.agent.evaluator import SimulatorEvaluator
    from simulator import costs

    ev = SimulatorEvaluator()
    rec = {"serving": {"n_gpu": 1, "gpu": "H100"}, "model_load_s": 3600.0,
           "levels": []}
    assert ev._spend(rec) == pytest.approx(
        costs.container_rate("H100", 1, vcpu=ev.vcpu), rel=0.01)
    assert ev._spend(rec) > costs.rate("H100", "modal", allow_retail=True) > costs.rate("H100")


def test_the_container_bills_more_than_its_gpu():
    """A night of sweeps at the GPU rate came to $43; Modal billed more,
    and the 16 reserved vCPUs are most of the gap."""
    from simulator import costs

    gpu_only = costs.rate("H100", "modal", allow_retail=True)
    full = costs.container_rate("H100", 1, vcpu=16.0)
    assert full == pytest.approx(gpu_only + 16 * costs.MODAL_USD_PER_VCPU_HOUR)
    assert full / gpu_only > 1.5


def test_a_sweep_that_died_is_billed_for_the_time_it_ran(tmp_path):
    import time

    import simulator
    from harness.agent.evaluator import SimulatorEvaluator

    real = simulator.Simulator
    try:
        class Stub:
            def __init__(self, **kw):
                pass

            async def eval(self):
                time.sleep(0.05)
                raise RuntimeError("container died")

        simulator.Simulator = Stub
        ok, metrics, failure = SimulatorEvaluator().evaluate(object(), str(tmp_path))
    finally:
        simulator.Simulator = real
    assert not ok and failure == "infra"
    assert metrics["cost_usd"] > 0 and metrics["cost_estimated"]


def test_a_failed_sweep_still_costs(tmp_path):
    """The GPU was rented either way. A failure that reports zero lets an agent
    burn its budget on infra problems for free."""
    import simulator
    from harness.agent.evaluator import SimulatorEvaluator

    class Res:
        ok = False
        reason = "no level met the SLO"
        quality_regressed = False
        quality: tuple = ()
        record: ClassVar = {"serving": {"n_gpu": 1, "gpu": "H100"},
                            "model_load_s": 300.0, "levels": [{"wall_s": 200.0}]}

    real = simulator.Simulator
    try:
        class Stub:
            def __init__(self, **kw):
                pass

            async def eval(self):
                return Res()

        simulator.Simulator = Stub
        ok, metrics, failure = SimulatorEvaluator().evaluate(object(), str(tmp_path))
    finally:
        simulator.Simulator = real
    assert not ok and failure == "slo"
    assert metrics["cost_usd"] > 0, "a failed sweep still rented the GPU"


def test_host_sleep_is_reported_not_hidden():
    """A closed lid froze three agents for five hours on 2026-09-02 while the
    GPUs they had rented kept billing; nothing said so. The control loop
    notices a wall-clock gap and the snapshot carries it."""
    broker = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    fleet = Fleet(lambda a, f: None, broker)
    try:
        fleet.note_host_sleep(4 * 3600)
        snap = fleet._snapshot()
        assert "host slept ~240 min" in snap.note
        assert fleet._slept_s == 4 * 3600
    finally:
        broker.shutdown()


def test_agents_claim_distinct_ideas_from_the_bank(tmp_path):
    """Seeds run out; the bank hands each agent a different mechanism and
    hears back what happened to it."""
    import time

    from harness.contracts import AgentOutcome, Attempt, IdeaRecord
    from harness.ideas import SqliteIdeaBank

    bank = SqliteIdeaBank(tmp_path / "ideas.db")
    for t, m, sc in (("fused decode attention", "fuse qk softmax pv in one kernel", "kernel"),
                     ("int8 KV cache", "store kv in int8 with per-head scales", "memory")):
        bank.add(IdeaRecord(title=t, mechanism=m, hypothesis=f"{t} lowers cost", scale=sc))
    got = {}

    class Agent:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def propose(self, seed=None, live_ideas=()):
            raise AssertionError("must not self-seed while the bank has records")

        def run(self, idea, budget):
            got[self.agent_id] = idea
            return AgentOutcome(agent_id=self.agent_id, idea=idea, stop="no_progress",
                                attempts=(Attempt(idea_id=idea.id, ok=True,
                                                  experiment_id="exp_9"),), cost_usd=0.0)

    broker = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    fleet = Fleet(lambda a, f: Agent(a), broker)
    fleet.bank = bank
    fleet.start(FleetSpec(fleet_budget=FleetBudget(max_agents=2, max_usd_total=5)))
    try:
        for _ in range(200):
            if bank.count("available") == 0 and bank.count("claimed") == 0:
                break
            time.sleep(0.02)
        assert len(got) == 2
        assert {i.seeded_by for i in got.values()} == {r.id for r in bank.list()}
        assert all(r.status == "tried" and r.experiment_ids == ("exp_9",) for r in bank.list())
    finally:
        fleet.stop()
        broker.shutdown()


def test_an_unmeasured_error_returns_the_idea_to_the_bank():
    """build-1 drained six bank records in a minute on a workspace crash."""
    import pathlib
    import tempfile

    from harness.contracts import AgentOutcome, Attempt, IdeaRecord
    from harness.ideas import SqliteIdeaBank

    bank = SqliteIdeaBank(pathlib.Path(tempfile.mkdtemp()) / "b.db")
    rec = IdeaRecord(title="t", mechanism="m", hypothesis="h")
    bank.add(rec)
    broker = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    fleet = Fleet(lambda a, f: None, broker)
    fleet.bank = bank
    try:
        idea = bank.claim("a00").as_idea()
        fleet._bank_outcome(AgentOutcome(agent_id="a00", idea=idea, stop="error"))
        assert bank.get(rec.id).status == "available"
        bank.claim("a00")
        fleet._bank_outcome(AgentOutcome(agent_id="a00", idea=idea, stop="no_progress",
                                         attempts=(Attempt(idea_id=idea.id, ok=True,
                                                           experiment_id="e1"),)))
        assert bank.get(rec.id).status == "tried"
    finally:
        broker.shutdown()


def test_the_fleet_keeps_publishing_while_it_winds_down():
    """`stop` is cooperative and an agent finishes its idea first; the
    dashboard must see "stopping" with live statuses, then "stopped"."""
    import time

    from harness.contracts import AgentOutcome
    from harness.session import SqliteSessionStore

    gate = threading.Event()

    class Slow:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def propose(self, seed=None, live_ideas=()):
            return Idea(title="t", hypothesis="h")

        def run(self, idea, budget):
            gate.wait(5)
            return AgentOutcome(agent_id=self.agent_id, idea=idea, stop="no_progress")

    import pathlib
    import tempfile
    store = SqliteSessionStore(pathlib.Path(tempfile.mkdtemp()) / "s.db")
    broker = EvalBroker(lambda r: (True, {}, ""), capacity=1)
    fleet = Fleet(lambda a, f: Slow(a), broker, store=store, session_id="wind", tick_s=0.05)
    fleet.start(FleetSpec(fleet_budget=FleetBudget(max_agents=1, max_usd_total=100)))
    try:
        for _ in range(100):
            v = store.read("wind")
            if v and v.agents:
                break
            time.sleep(0.02)
        stopper = threading.Thread(target=fleet.stop, daemon=True)
        stopper.start()
        seen = set()
        for _ in range(40):
            time.sleep(0.05)
            v = store.read("wind")
            if v:
                seen.add(v.phase)
        assert "stopping" in seen, seen                # published during the drain
        gate.set()
        stopper.join(10)
        assert store.read("wind").phase == "stopped"
    finally:
        broker.shutdown()
