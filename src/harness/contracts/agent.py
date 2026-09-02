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
    # Free text from an idea bank: the mechanism worked out, the expected gain
    # and the risks. Empty for an idea an agent seeded itself. A kernel-scale
    # proposal cannot be carried in one sentence, and an agent handed only the
    # sentence re-derives the design badly and spends a sweep on it.
    design: str = ""
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
    failure: Literal["", "infra", "hypothesis", "invalid_diff", "slo",
                     "quality"] = ""
    tier: str = "full"
    metrics: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)     # vs the baseline
    cost_usd: float = 0.0
    queued_s: float = 0.0
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
    # Seconds this agent spent with nothing to do. The number the evaluation
    # queue exists to keep near zero -- an agent blocked on a GPU is capacity
    # being paid for and not used.
    idle_s: float = 0.0


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
    # Screen first, confirm only what survives. The largest throughput lever
    # in the system: a screening run is a fraction of a full sweep and most
    # candidates die in it, so running everything at full size wastes more
    # capacity than any scheduling improvement can recover.
    screen_first: bool = True
    # A screen has to beat the baseline by at least this much to earn a full
    # sweep. Above zero on purpose: a screen is noisy, and confirming
    # break-even candidates is how the queue fills with nothing.
    screen_promise_pct: float = -1.0
    # A full-tier result that clears the noise floor is measured again and
    # the worse of the two is kept. The frontier is quantised to the level
    # grid, and a level sitting on the SLO line passes or fails on noise:
    # on 2026-09-02 a no-op diff scored 18% below baseline because N=12
    # held 20 ms mean TPOT in its sweep and not in stock's. One extra sweep
    # per claimed win is the cheapest defence against recording that.
    replicate_wins: bool = True
    # Infrastructure failures are retried unchanged, but not forever: a
    # persistently broken runner would otherwise consume the whole budget
    # rediscovering that it is broken.
    max_infra_retries: int = 2
    # How long the study that runs alongside an evaluation may take. It is
    # cancelled the moment the result lands -- studying past that answers a
    # question that has been answered -- so this only bounds the other case: a
    # study that outlives a sweep and would otherwise hold the attempt open.
    study_timeout_s: float = 900.0


@runtime_checkable
class AgentControl(Protocol):
    """How a running agent asks the fleet whether it may keep going.

    An operator watching a fleet needs to pause or kill one agent without
    touching the other nine, and an agent cannot be interrupted at an arbitrary
    instant -- it may be halfway through a paid evaluation. So control is
    *cooperative*: the agent checks at points where stopping is cheap and
    resuming is coherent.

    `report` is the other direction: it is how a dashboard knows an agent is
    thinking rather than hung.
    """

    def should_stop(self, agent_id: str) -> bool:
        """True when this agent should wind up after the current attempt."""
        ...

    def wait_if_paused(self, agent_id: str, timeout_s: float = 3600) -> bool:
        """Block while paused. False means "stop instead of resuming"."""
        ...

    def report(self, agent_id: str, **fields) -> None:
        """Publish what this agent is doing right now. Never blocks."""
        ...


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
