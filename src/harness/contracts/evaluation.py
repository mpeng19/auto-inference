"""EvalService: the queue in front of the GPUs.

Split out of `OrchestrationService` because a blocking gate is the wrong shape.
The first version handed out slots with a semaphore, so seven of ten agents sat
in `acquire()` doing nothing while three ran sweeps. That is not a throughput
problem, it is a *design* problem: waiting for a GPU and having nothing to do
are different things, and only the first is unavoidable.

So evaluation is a **submit/collect** service, not a lock:

    ticket = evals.submit(EvalRequest(...))    # returns at once, always
    ...                                        # agent keeps working
    rec = evals.collect(ticket)                # blocks only when it must

Four properties the contract requires of any implementation, each aimed at a
specific way throughput is lost:

**Non-blocking submit.** An agent must never be unable to *propose*. Queue
depth is a number it can see and reason about, not a wall it hits.

**Dedup by content.** Two agents proposing the same diff is common -- the fleet
seeds around a shared baseline -- and a stack digest is a content hash. The
second submit joins the first ticket instead of renting a second GPU.

**Fairness, then priority.** A plain semaphore lets a fast agent barge
repeatedly and starve a slow one. Ordering is FIFO within a priority band, so
the only way to jump the queue is to be explicitly marked worth it.

**Tiers.** The largest lever is not scheduling at all: a screening run costs a
fraction of a full sweep, and most candidates die in it. Cheap tickets and
expensive tickets are different sizes of the same resource, and the service
knows which is which.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .common import new_id, now

Tier = Literal["screen", "full"]
EvalStatus = Literal["queued", "running", "done", "failed", "cancelled", "deduped"]


@dataclass(frozen=True)
class EvalRequest:
    """One thing to measure. `stack` is a `simulator.InferenceStack`."""
    stack: Any
    agent_id: str = ""
    idea_id: str = ""
    attempt: int = 0
    tier: Tier = "full"
    # A deliberate re-measurement of the same code. 0 is the first run; a
    # replicate is a different key on purpose, because the point of it is
    # to pay for the same measurement twice.
    replicate: int = 0
    priority: int = 0               # higher runs sooner; ties broken FIFO
    run_dir: str = ""
    label: str = ""

    @property
    def dedup_key(self) -> str:
        """Same code, same tier -> same measurement. Never pay twice --
        unless asked to (`replicate`)."""
        key = f"{getattr(self.stack, 'digest', '')}:{self.tier}"
        return f"{key}:r{self.replicate}" if self.replicate else key


@dataclass(frozen=True)
class EvalTicket:
    id: str = field(default_factory=lambda: new_id("ev"))
    request: EvalRequest | None = None
    submitted_at: float = field(default_factory=now)
    deduped_from: str = ""          # set when this joined an in-flight identical run


@dataclass(frozen=True)
class EvalRecord:
    """The result. `ok=False` with `failure` says which kind of no it is."""
    ticket_id: str
    status: EvalStatus
    ok: bool = False
    metrics: dict = field(default_factory=dict)
    # "quality" is one of these, not a footnote: an accuracy gate rejects a
    # stack that priced well and answered worse, and a reader of this field has
    # to be able to tell that from the infrastructure failure it would retry.
    failure: Literal["", "infra", "hypothesis", "slo", "invalid_diff", "quality",
                     "timeout", "cancelled"] = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    queued_s: float = 0.0
    cost_usd: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class QueueStats:
    """What an agent needs to decide whether to submit now or keep thinking."""
    queued: int = 0
    running: int = 0
    capacity: int = 0
    est_wait_s: float = 0.0
    completed: int = 0
    deduped: int = 0
    # Fraction of wall-clock with every slot busy. The number this service
    # exists to keep high; a fleet with idle GPUs is one paying for nothing.
    utilisation: float = 0.0


@runtime_checkable
class EvalService(Protocol):
    def submit(self, req: EvalRequest) -> EvalTicket:
        """Enqueue. Returns immediately, always. Never blocks, never refuses."""
        ...

    def poll(self, ticket_id: str) -> EvalRecord:
        """Current state without waiting. Safe to call in a loop."""
        ...

    def collect(self, ticket_id: str, timeout_s: float | None = None) -> EvalRecord:
        """Wait for a result. The only blocking call in the interface."""
        ...

    def cancel(self, ticket_id: str) -> bool:
        """Drop a queued ticket. A running one is left alone -- it is already paid for."""
        ...

    def stats(self) -> QueueStats: ...
