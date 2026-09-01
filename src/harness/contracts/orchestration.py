"""OrchestrationService: the fleet.

Wide and deep at once -- N agents in parallel, each iterating on one idea. The
orchestrator owns exactly the things a single agent cannot see:

**The real contended resource is not CPU.** Agents write files and call an API;
nothing about that needs isolating. What is scarce is **GPU concurrency and
money**: every attempt is a 25-60 minute sweep on a rented H100. Ten agents
attempting freely is ten simultaneous sweeps, and the failure mode is a bill,
not a race condition. So the orchestrator gates evaluations, not processes.

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
    baseline_stack_digest: str = ""
    baseline_metrics: dict = field(default_factory=dict)
    seeds: tuple[Idea, ...] = ()
    agent_budget: AgentBudget = field(default_factory=AgentBudget)
    fleet_budget: FleetBudget = field(default_factory=FleetBudget)
    root: str = "agents"            # per-agent working directories live here
    note: str = ""


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

    def acquire_eval_slot(self, agent_id: str, timeout_s: float = 3600) -> bool:
        """Block until this agent may spend GPU time. The whole cost control."""
        ...

    def release_eval_slot(self, agent_id: str) -> None: ...
