"""Auto-research harness: a fleet of agents improving an inference stack.

Ten agents, each iterating on one idea, each proposing diffs to SGLang's `srt/`
and pricing them with `simulator`. Four services with strict contracts, so any
one can be replaced without the others noticing:

    MemoryService          what the fleet has already tried, and what it learned
    ContextService         the full transcript behind any one of those claims
    EvalService            the queue in front of the GPUs
    AgentService           one idea, iterated, with divergence and retry policy
    OrchestrationService   N of those in parallel, kept diverse and in budget

The scarce thing is **GPU concurrency and money**, not CPU: every attempt rents
an H100 for 25-60 minutes. Agents themselves only write text and wait on an
API, and the modified SGLang runs exclusively inside a fresh Modal container --
so the isolation that matters is already paid for.

That scarcity is managed as a **queue, not a lock**: agents submit and keep
working. An earlier version handed out slots with a semaphore and seven of ten
agents sat blocked in it, which is capacity being paid for and not used.

    from harness import Fleet, IterativeAgent, Workspace
    from harness.memory import SqliteMemory
    from harness.context import JsonlContext

Contracts live in `harness.contracts` and hold no logic. Everything else is a
reference implementation of one of them.
"""
from .agent import Evaluator, IterativeAgent, Proposer, Workspace
from .agent.evaluator import SimulatorEvaluator
from .context import JsonlContext
from .contracts import (
    AgentBudget,
    AgentOutcome,
    AgentService,
    Attempt,
    Brief,
    ContextService,
    EvalRecord,
    EvalRequest,
    EvalService,
    EvalTicket,
    Experiment,
    Finding,
    FleetBudget,
    FleetSpec,
    FleetState,
    Idea,
    MemoryService,
    OrchestrationService,
    QueueStats,
    Recall,
    Relation,
    TraceMeta,
    Turn,
)
from .memory import SqliteMemory
from .orchestration import EvalBroker, Fleet

__all__ = [
    # services (implementations)
    "Fleet", "EvalBroker", "IterativeAgent", "Workspace",
    "SqliteMemory", "JsonlContext", "SimulatorEvaluator",
    # extension points
    "Proposer", "Evaluator",
    # contracts
    "MemoryService", "ContextService", "EvalService", "AgentService",
    "OrchestrationService",
    # exchanged types
    "Experiment", "Relation", "Finding", "Recall", "Brief",
    "TraceMeta", "Turn",
    "EvalRequest", "EvalTicket", "EvalRecord", "QueueStats",
    "Idea", "Attempt", "AgentOutcome", "AgentBudget",
    "FleetSpec", "FleetState", "FleetBudget",
]
__version__ = "0.1.0"
