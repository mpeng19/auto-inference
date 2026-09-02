"""Agent implementations, the per-agent workspace, and the stock-source layer."""
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
