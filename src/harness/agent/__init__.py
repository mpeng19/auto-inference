"""One agent: a workspace, a proposer that edits it, an evaluator that prices it.

Implements `harness.contracts.AgentService` (`IterativeAgent`) plus the two
plug points it is built from, `Proposer` (the model) and `Evaluator` (the
GPU), so the loop runs unchanged against fakes.

    ws = Workspace(root / agent_id, agent_id=agent_id)          # stock files + candidate edits
    prop = ClaudeCodeProposer(model="opus")                    # `claude -p` in ws.candidates
    agent = IterativeAgent(agent_id, ws, memory, context, prop, evals, control)
    outcome = agent.run(idea, AgentBudget(...))                # AgentOutcome

On disk, under the agent's directory: `candidate/sglang/` (the files it
edits, read back as the diff), `runs/attempt-NNN/` (one evaluation each),
`calls/` (per-model-call token logs), `spend.jsonl` (GPU tool spend, drained
by the loop), `paper/<idea>/` (the write-up), `mcp.json` and
`candidate/.claude/skills/` (what the agent is handed). Stock SGLang is read
from the installed package or the pinned wheel in `~/.cache/auto-inference`.
"""
from .claude_code import ClaudeCodeProposer, ClaudeCodeUnavailable
from .evaluator import SimulatorEvaluator
from .loop import Evaluator, IterativeAgent, Proposer
from .stock import InstalledSglang, WheelSource, stock
from .workspace import Workspace

__all__ = [
           "ClaudeCodeProposer",
           "ClaudeCodeUnavailable",
           "Evaluator",
           "InstalledSglang",
           "IterativeAgent",
           "Proposer",
           "SimulatorEvaluator",
           "WheelSource",
           "Workspace",
           "stock",
]
