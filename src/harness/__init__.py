"""Auto-research harness: a fleet of agents improving an inference stack.

Ten agents, each iterating on one idea, each proposing diffs to SGLang's `srt/`
and pricing them with `simulator`. Four services with strict contracts, so any
one can be replaced without the others noticing:

    MemoryService          what the fleet has already tried, and what it learned
    ContextService         the full transcript behind any one of those claims
    AgentService           one idea, iterated, with divergence and retry policy
    OrchestrationService   N of those in parallel, gating the only scarce thing

The scarce thing is **GPU concurrency and money**, not CPU: every attempt rents
an H100 for 25-60 minutes. Agents themselves only write text and wait on an
API, and the modified SGLang runs exclusively inside a fresh Modal container --
so the isolation that matters is already paid for, and the orchestrator gates
evaluations rather than processes.

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
                        Experiment,
                        Finding,
                        FleetBudget,
                        FleetSpec,
                        FleetState,
                        Idea,
                        MemoryService,
                        OrchestrationService,
                        Recall,
                        Relation,
                        TraceMeta,
                        Turn,
)
from .memory import SqliteMemory
from .orchestration import Fleet

__all__ = [
                        "AgentBudget",
                        "AgentOutcome",
                        "AgentService",
                        "Attempt",
                        "Brief",
                        "ContextService",
                        "Evaluator",
                        "Experiment",
                        "Finding",
                        "Fleet",
                        "FleetBudget",
                        "FleetSpec",
                        "FleetState",
                        "Idea",
                        "IterativeAgent",
                        "JsonlContext",
                        "MemoryService",
                        "OrchestrationService",
                        "Proposer",
                        "Recall",
                        "Relation",
                        "SimulatorEvaluator",
                        "SqliteMemory",
                        "TraceMeta",
                        "Turn",
                        "Workspace",
]
__version__ = "0.1.0"
