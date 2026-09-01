"""Reference OrchestrationService: N agent loops, one gate on GPU spend.

The gate is the whole point. Agents are I/O-bound -- they think, write text
files, and then wait 25-60 minutes for a sweep -- so running ten of them costs
almost nothing locally. What costs money is that each attempt rents an H100,
and ten agents attempting freely is ten simultaneous rentals.

So this class does exactly three things a single agent cannot do for itself:

**Hold the budget.** Whether the fleet can afford to keep going at all. The
*scheduling* of GPU time lives in `EvalService`, deliberately: this class used
to hand out slots with a semaphore and seven of ten agents sat blocked in it,
which is capacity being rented and not used.

**Keep seeds diverse.** No agent can tell whether its idea duplicates another's;
only something holding all ten can. `_too_similar` is a deliberately crude
token-overlap check: it is meant to catch "raise the batch size" proposed six
times, not to adjudicate research taste.

**Reclaim slots from dead ideas.** An agent that has learned nothing in three
attempts should hand its slot to a fresh idea rather than spend its remaining
budget proving the same negative more precisely.

Threads, not processes: the work is waiting on a network call, and sharing one
SQLite connection and one process makes state inspection trivial. If an agent
implementation ever executes untrusted code locally, that changes -- but the
modified SGLang only ever runs inside a fresh Modal container, so today it does
not.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from ..contracts.agent import AgentOutcome, AgentService, Idea
from ..contracts.orchestration import AgentState, FleetSpec, FleetState


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()
            if len(w) > 3}


class Fleet:
    """Reference implementation of `contracts.orchestration.OrchestrationService`."""

    def __init__(self, make_agent, evals, *, similarity_threshold: float = 0.6):
        # `make_agent(agent_id, fleet) -> AgentService`. Injected so the fleet
        # never needs to know how an agent thinks.
        self.make_agent = make_agent
        self._evals = evals              # contracts.evaluation.EvalService
        self.similarity_threshold = similarity_threshold
        self._lock = threading.RLock()
        self._state = FleetState()
        self._spec: FleetSpec | None = None
        self._live_ideas: list[Idea] = []
        self._pool: ThreadPoolExecutor | None = None
        self._stop = threading.Event()

    @property
    def evals(self):
        """The evaluation queue. Agents submit here and keep working."""
        return self._evals

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self, spec: FleetSpec) -> str:
        with self._lock:
            if self._state.running:
                raise RuntimeError("fleet already running")
            self._spec = spec
            self._stop.clear()
            n = spec.fleet_budget.max_agents
            self._state = FleetState(
                running=True, started_at=time.time(),
                agents=tuple(AgentState(agent_id=f"a{i:02d}") for i in range(n)))
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="agent")
        for i in range(n):
            self._pool.submit(self._run_agent, f"a{i:02d}")
        return f"fleet-{int(self._state.started_at)}"

    def state(self) -> FleetState:
        with self._lock:
            return self._state

    def stop(self, reason: str = "") -> FleetState:
        self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        with self._lock:
            self._state = replace(self._state, running=False)
            return self._state

    # ── budget ───────────────────────────────────────────────────────────
    def _within_budget(self) -> bool:
        s, b = self.state(), self._spec.fleet_budget
        if s.cost_usd >= b.max_usd_total:
            return False
        return not (s.started_at and time.time() - s.started_at > b.max_wall_s)

    # ── seeding ──────────────────────────────────────────────────────────
    def claim_idea(self, agent_id: str, proposed: Idea) -> Idea | None:
        """Register an idea if it is not what someone else is already doing."""
        with self._lock:
            for other in self._live_ideas:
                if self._too_similar(proposed, other):
                    return None
            self._live_ideas.append(proposed)
            return proposed

    def live_ideas(self) -> tuple[Idea, ...]:
        with self._lock:
            return tuple(self._live_ideas)

    def _too_similar(self, a: Idea, b: Idea) -> bool:
        ta, tb = _tokens(a.hypothesis + " " + a.title), _tokens(b.hypothesis + " " + b.title)
        if not ta or not tb:
            return False
        jaccard = len(ta & tb) / len(ta | tb)
        return jaccard >= self.similarity_threshold

    # ── bookkeeping ──────────────────────────────────────────────────────
    def _touch(self, agent_id: str, *, status: str = "", cost: float = 0.0,
               attempts: int = 0, evals_delta: int = 0, note: str = "") -> None:
        with self._lock:
            agents = []
            for a in self._state.agents:
                if a.agent_id == agent_id:
                    a = replace(a, status=status or a.status,
                                cost_usd=a.cost_usd + cost,
                                attempts=a.attempts + attempts,
                                last_seen=time.time(), note=note or a.note)
                agents.append(a)
            self._state = replace(
                self._state, agents=tuple(agents),
                cost_usd=self._state.cost_usd + cost,
                evals_in_flight=max(0, self._state.evals_in_flight + evals_delta))

    def _record_outcome(self, out: AgentOutcome) -> None:
        with self._lock:
            self._state = replace(self._state,
                                  completed=(*self._state.completed, out))
            self._live_ideas = [i for i in self._live_ideas if i.id != out.idea.id]

    # ── the agent thread ─────────────────────────────────────────────────
    def _run_agent(self, agent_id: str) -> None:
        spec = self._spec
        seeds = list(spec.seeds)
        while not self._stop.is_set() and self._within_budget():
            agent: AgentService = self.make_agent(agent_id, self)
            seed = seeds.pop(0) if seeds else None
            idea = agent.propose(seed=seed, live_ideas=self.live_ideas())
            if self.claim_idea(agent_id, idea) is None:
                self._touch(agent_id, note="idea duplicated another agent's; reseeding")
                continue
            self._touch(agent_id, status="working", note=idea.title)
            try:
                out = agent.run(idea, spec.agent_budget)
            except Exception as e:                      # one agent must not kill the fleet
                self._touch(agent_id, status="failed", note=f"{type(e).__name__}: {e}")
                with self._lock:
                    self._live_ideas = [i for i in self._live_ideas if i.id != idea.id]
                continue
            self._touch(agent_id, status="idle", cost=out.cost_usd,
                        attempts=len(out.attempts),
                        note=f"{out.stop} (idle {out.idle_s:.0f}s)")
            self._record_outcome(out)
        self._touch(agent_id, status="done")
