"""Strict input/output contracts. No logic lives here.

Every service in the harness is defined by a Protocol in this package plus the
dataclasses it exchanges. Implementations are swappable by construction: a
better memory store or a sandboxed agent runner drops in without any other
service changing, which is the only way an experimental system survives being
rewritten repeatedly.
"""
from .agent import AgentBudget, AgentOutcome, AgentService, Attempt, Idea
from .common import Provenance, digest, new_id, now
from .context import ContextService, Slice, TraceMeta, Turn
from .memory import Brief, Experiment, Finding, Hit, MemoryService, Recall, Relation
from .orchestration import (
                     AgentState,
                     FleetBudget,
                     FleetSpec,
                     FleetState,
                     OrchestrationService,
)

__all__ = [
                     "AgentBudget",
                     "AgentOutcome",
                     "AgentService",
                     "AgentState",
                     "Attempt",
                     "Brief",
                     "ContextService",
                     "Experiment",
                     "Finding",
                     "FleetBudget",
                     "FleetSpec",
                     "FleetState",
                     "Hit",
                     "Idea",
                     "MemoryService",
                     "OrchestrationService",
                     "Provenance",
                     "Recall",
                     "Relation",
                     "Slice",
                     "TraceMeta",
                     "Turn",
                     "digest",
                     "new_id",
                     "now",
]
