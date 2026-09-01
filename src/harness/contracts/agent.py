"""AgentService: one idea, iterated until it pays off or is abandoned.

An agent here is not a chat loop. It is a bounded search over one hypothesis,
and the contract's job is to make three failure modes impossible to ignore:

**Seed diversity.** Ten agents that all propose raising the batch size are one
agent with a nine-times-larger bill. `propose` takes the fleet's live ideas so
a seed can be rejected as a near-duplicate before it costs a sweep.

**Divergence.** An agent iterating on "improve chunked prefill" will, given
enough attempts, end up rewriting the scheduler. That is sometimes the right
answer and usually a lost run, so every attempt is checked against the idea it
started from and the loop stops when it has drifted past a threshold rather
than when the budget runs out.

**Retries that mean something.** A failed *evaluation* (the GPU died, the
server never started) is worth retrying unchanged. A failed *hypothesis* is
not: retrying it produces the same number and a second bill. `Attempt` records
which kind of failure it was, and only one of them is retried.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .common import new_id, now

Stop = Literal["won", "exhausted", "diverged", "no_progress", "budget", "error"]


@dataclass(frozen=True)
class Idea:
    """A seed. One sentence an agent can be wrong about."""
    id: str = field(default_factory=lambda: new_id("idea"))
    title: str = ""
    hypothesis: str = ""            # "X will lower cost per output token because Y"
    targets: tuple[str, ...] = ()   # sglang paths it expects to touch
    seeded_by: str = ""             # agent id, or "human", or an experiment id
    timeline: str = "main"
    created_at: float = field(default_factory=now)


@dataclass(frozen=True)
class Attempt:
    """One diff, evaluated. The unit an agent is billed for."""
    id: str = field(default_factory=lambda: new_id("att"))
    idea_id: str = ""
    agent_id: str = ""
    n: int = 0                      # attempt number within the idea
    stack_digest: str = ""
    experiment_id: str = ""         # what memory recorded
    trace_ref: str = ""

    ok: bool = False
    # Split on purpose: an infrastructure failure is retried unchanged, a
    # rejected hypothesis is not.
    failure: Literal["", "infra", "hypothesis", "invalid_diff", "slo"] = ""
    metrics: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)     # vs the baseline
    cost_usd: float = 0.0
    ts: float = field(default_factory=now)


@dataclass(frozen=True)
class AgentOutcome:
    agent_id: str
    idea: Idea
    stop: Stop
    attempts: tuple[Attempt, ...] = ()
    best: Attempt | None = None
    cost_usd: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class AgentBudget:
    max_attempts: int = 6
    max_usd: float = 40.0
    max_wall_s: float = 6 * 3600
    # Stop early when the last `patience` attempts brought nothing. Attempts
    # are ~25 GPU-minutes each; spending six of them to confirm a dead idea is
    # the most expensive thing this system can do by accident.
    patience: int = 3
    divergence_threshold: float = 0.6


@runtime_checkable
class AgentService(Protocol):
    def propose(self, *, seed: Idea | None, live_ideas: tuple[Idea, ...]) -> Idea:
        """Pick this agent's idea, avoiding what the fleet is already doing."""
        ...

    def run(self, idea: Idea, budget: AgentBudget) -> AgentOutcome:
        """Iterate on one idea until it wins, stalls, diverges, or runs out."""
        ...

    def divergence(self, idea: Idea, attempt: Attempt) -> float:
        """0 = still the same idea, 1 = unrecognisable. Checked every attempt."""
        ...
