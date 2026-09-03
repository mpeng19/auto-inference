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
    IdeaBankService        where ideas of the right size come from, claimed one per agent
    SkillBankService       what earlier runs established, written by the manager, read by all
    OrchestrationService   N of those, kept diverse and inside a budget
    SessionStore           the seam between a detached fleet and anything watching it

`AgentControl` (in `agent.py`) is the other half of that last one: how a
running agent asks the fleet whether it may keep going.
"""
from .agent import (
    AgentBudget,
    AgentControl,
    AgentOutcome,
    AgentService,
    Attempt,
    Idea,
)
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
from .ideas import BankStatus, IdeaBankService, IdeaRecord, Scale
from .memory import Brief, Experiment, Finding, Hit, MemoryService, Recall, Relation
from .orchestration import (
    AgentState,
    FleetBudget,
    FleetSpec,
    FleetState,
    OrchestrationService,
)
from .session import (
    AgentView,
    Command,
    SessionStore,
    SessionView,
    TokenUse,
)
from .skills import Fact, FactStatus, Judge, SkillBankService

__all__ = [
    "BankStatus", "IdeaBankService", "IdeaRecord", "Scale",
    "Fact", "FactStatus", "Judge", "SkillBankService",
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
    "AgentService", "AgentControl", "Idea", "Attempt", "AgentOutcome",
    "AgentBudget",
    # session / control
    "SessionStore", "SessionView", "AgentView", "Command", "TokenUse",
    # orchestration
    "OrchestrationService", "FleetSpec", "FleetState", "FleetBudget", "AgentState",
]
