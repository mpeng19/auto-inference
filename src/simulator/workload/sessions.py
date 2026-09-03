"""What the load generator replays: a conversation, turn by turn.

Two plain records and nothing else. They are deliberately dumb containers --
every number in them comes from `workload.tracelab`, which rescales real
coding-agent sessions to the marketplace's token shape, and the whole point of
keeping this module empty of generation logic is that a `Session` says nothing
about *where* its shape came from.

The one thing worth knowing is that a session is **closed-loop**: the runner
sends turn `k+1` only after the reply to turn `k` has arrived, then waits
`think_s[k]`. That is what makes the concurrency axis mean "users", and it is
why a session carries think times rather than arrival times.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One user message inside a conversation."""
    text: str
    max_tokens: int


@dataclass(frozen=True)
class Session:
    """A conversation, run closed-loop turn by turn."""
    idx: int
    system: str                     # stable preamble, shared across sessions
    turns: tuple[Turn, ...]
    think_s: tuple[float, ...]      # gap after each turn before the next

    @property
    def n_turns(self) -> int:
        return len(self.turns)
