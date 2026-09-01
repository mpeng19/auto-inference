"""ContextService: the full transcript of what an agent actually did.

The split from `MemoryService` is the important part, and it is a split by
*question*, not by size:

    memory   "has anyone tried this, and what happened?"      claims
    context  "how exactly did they do it?"                    transcripts

A claim is small, curated, and worth putting in another agent's context. A
transcript is large, raw, and worth keeping only so a claim can be checked or
a mechanism recovered. Merging them produces a store where every query returns
a wall of tool calls -- which is the failure mode this boundary exists to
prevent.

So memory rows carry a `trace_ref` and nothing more. An agent that wants the
detail behind a claim resolves that ref here, and pays for what it asked for.

Traces are append-only JSONL: one turn per line, written as the agent runs.
That format is not incidental -- it means a crashed agent still has everything
up to the crash, and it means the file can be tailed by an orchestrator that
wants to notice divergence before the run ends.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .common import new_id, now

TurnKind = Literal["prompt", "thought", "tool_call", "tool_result", "message",
                   "eval_submit", "eval_result", "error"]


@dataclass(frozen=True)
class Turn:
    """One line of a trace."""
    kind: TurnKind
    ts: float = field(default_factory=now)
    name: str = ""                  # tool name, or the eval's run id
    content: str = ""
    data: dict = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0


@dataclass(frozen=True)
class TraceMeta:
    """Everything needed to interpret a trace without opening it."""
    id: str = field(default_factory=lambda: new_id("trc"))
    agent_id: str = ""
    idea_id: str = ""
    attempt: int = 0
    started_at: float = field(default_factory=now)
    ended_at: float = 0.0
    model: str = ""
    harness_commit: str = ""
    n_turns: int = 0
    cost_usd: float = 0.0
    outcome: str = ""


@dataclass(frozen=True)
class Slice:
    """A piece of a trace, returned by a query rather than the whole file."""
    trace_id: str
    turns: tuple[Turn, ...]
    reason: str = ""


@runtime_checkable
class ContextService(Protocol):
    def open(self, meta: TraceMeta) -> str:
        """Start a trace. Returns the ref that goes in `Experiment.trace_ref`."""
        ...

    def append(self, trace_ref: str, turn: Turn) -> None: ...

    def close(self, trace_ref: str, outcome: str = "", cost_usd: float = 0.0) -> None: ...

    def meta(self, trace_ref: str) -> TraceMeta | None: ...

    def read(self, trace_ref: str) -> Iterator[Turn]:
        """Stream a whole trace. Rarely what you want -- prefer `slice`."""
        ...

    def slice(self, trace_ref: str, *, kinds: tuple[TurnKind, ...] = (),
              query: str = "", limit: int = 50) -> Slice:
        """The part of a trace that answers a question.

        The default read path. Handing an agent a full transcript costs more
        than the answer is worth, and the parts worth having are almost always
        the tool calls that touched the thing being asked about.
        """
        ...

    def tail(self, trace_ref: str, n: int = 20) -> Slice:
        """The last n turns. What an orchestrator watches for divergence."""
        ...

    def stats(self, *, agent_id: str = "", idea_id: str = "") -> dict[str, Any]:
        """Cost and turn accounting, for the orchestrator's budget decisions."""
        ...
