"""Strict input/output contracts. No logic lives here.

Every service in the harness is defined by a Protocol in this package plus the
dataclasses it exchanges. Implementations are swappable by construction: a
better memory store, a sandboxed agent runner, a different evaluation backend
drops in without any other service changing, which is the only way a system
that gets rewritten repeatedly stays coherent.

    MemoryService          what the fleet has tried, and what it learned
    ContextService         the transcript behind any one of those claims
    EvalService            the queue in front of the GPUs
    AgentService           one idea, iterated
    OrchestrationService   N of those, kept diverse and inside a budget
"""
from .agent import AgentBudget, AgentOutcome, AgentService, Attempt, Idea
from .common import Provenance, digest, new_id, now
from .context import ContextService, Slice, TraceMeta, Turn
from .evaluation import (
    EvalRecord,
    EvalRequest,
    EvalService,
    EvalStatus,
    EvalTicket,
    QueueStats,
    Tier,
)
from .memory import Brief, Experiment, Finding, Hit, MemoryService, Recall, Relation
from .orchestration import (
    AgentState,
    FleetBudget,
    FleetSpec,
    FleetState,
    OrchestrationService,
)

__all__ = [
    # shared
    "Provenance", "digest", "new_id", "now",
    # memory
    "MemoryService", "Experiment", "Relation", "Finding", "Recall", "Hit", "Brief",
    # context
    "ContextService", "TraceMeta", "Turn", "Slice",
    # evaluation
    "EvalService", "EvalRequest", "EvalTicket", "EvalRecord", "EvalStatus",
    "QueueStats", "Tier",
    # agent
    "AgentService", "Idea", "Attempt", "AgentOutcome", "AgentBudget",
    # orchestration
    "OrchestrationService", "FleetSpec", "FleetState", "FleetBudget", "AgentState",
]
