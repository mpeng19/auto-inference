"""OrchestrationService: the fleet.

Wide and deep at once -- N agents in parallel, each iterating on one idea. The
orchestrator owns exactly the things a single agent cannot see:

**The real contended resource is not CPU.** Agents write files and call an API;
nothing about that needs isolating. What is scarce is **GPU concurrency and
money**: every attempt is a 25-60 minute sweep on a rented H100. Ten agents
attempting freely is ten simultaneous sweeps, and the failure mode is a bill,
not a race condition.

That scheduling lives in `EvalService` (see `contracts/evaluation.py`), not
here, and it is a queue rather than a lock so an agent waiting on a GPU is
still doing work. This service owns what is left: which ideas exist, whether
they are diverse, and when an agent should give its slot up.

**Diversity is a fleet property.** No agent can tell whether its idea is a
duplicate; only something holding all ten can. Seeding goes through here.

**Stopping is a fleet decision too.** An agent that has found nothing in six
attempts should release its slot to a fresh idea rather than spend its budget
proving the same negative more precisely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .agent import AgentBudget, AgentOutcome, Idea
from .common import now
from .evaluation import EvalService


@dataclass(frozen=True)
class FleetBudget:
    """Ceilings the fleet may not cross, whatever the agents want."""
    max_agents: int = 10
    # The one that matters. Each concurrent evaluation is a GPU being rented.
    max_concurrent_evals: int = 3
    max_usd_total: float = 500.0
    max_wall_s: float = 24 * 3600


@dataclass(frozen=True)
class FleetSpec:
    baseline_metrics: dict = field(default_factory=dict)
    seeds: tuple[Idea, ...] = ()
    agent_budget: AgentBudget = field(default_factory=AgentBudget)
    fleet_budget: FleetBudget = field(default_factory=FleetBudget)
    root: str = "agents"            # per-agent working directories live here
    note: str = ""
    # A compounding fleet: every agent starts from this saved stack rather
    # than stock, and `baseline_metrics` is that stack's own measured price.
    # The digest is what the snapshot shows; `base_seed` (the base's label
    # plus the idea that produced it, when known) steers bank claims toward
    # the direction that worked.
    base_digest: str = ""
    base_seed: str = ""
    # Texts every bank claim is kept away from, on top of what is live and
    # what this fleet has tried: a campaign passes every idea its earlier
    # rounds tried, so round three does not re-run round one.
    avoid: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentState:
    agent_id: str
    idea: Idea | None = None
    status: str = "idle"            # idle | working | evaluating | done | failed
    attempts: int = 0
    cost_usd: float = 0.0
    last_seen: float = field(default_factory=now)
    note: str = ""


@dataclass(frozen=True)
class FleetState:
    running: bool = False
    started_at: float = 0.0
    agents: tuple[AgentState, ...] = ()
    cost_usd: float = 0.0
    evals_in_flight: int = 0
    completed: tuple[AgentOutcome, ...] = ()


@runtime_checkable
class OrchestrationService(Protocol):
    def start(self, spec: FleetSpec) -> str: ...

    def state(self) -> FleetState: ...

    def stop(self, reason: str = "") -> FleetState: ...

    @property
    def evals(self) -> EvalService:
        """The evaluation queue. Agents submit here; nobody holds a lock.

        Deliberately not `acquire_slot()`. The first version of this interface
        handed out slots with a semaphore and seven of ten agents sat blocked
        in it doing nothing. Waiting for a GPU and having nothing to do are
        different problems, and only the first is unavoidable.
        """
        ...
