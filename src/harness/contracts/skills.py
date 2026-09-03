"""The skill bank: general facts about serving, learned across runs.

Memory records *experiments*: this stack, this diff, this verdict. It is
per run and it is evidence. The skill bank records what the evidence
taught -- "FP8 greedy decoding is not deterministic across batch
compositions; a 2-point GSM8K tolerance rejects stock" -- as claims with
their evidence attached, shared across every run on this machine, written
by the manager only, and rendered into a skill every agent reads before it
edits.

A fact can be wrong, and a later run can show it. The bank does not let two
facts on the same topic contradict each other silently: adding one asks a
judge whether it contradicts what is already held on that topic, and the
losing fact is marked superseded with a pointer to its successor. Nothing
is deleted; the history of what was believed is part of the record.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .common import new_id, now

FactStatus = Literal["active", "superseded", "retracted"]


@dataclass(frozen=True)
class Fact:
    id: str = field(default_factory=lambda: new_id("fact"))
    claim: str = ""                     # one sentence, falsifiable
    topic: str = ""                     # short handle: "kv-cache", "gsm8k-gate", "chunked-prefill"
    evidence: str = ""                  # where it came from, with numbers
    source: str = ""                    # session id / experiment id / "human"
    confidence: float = 0.7             # 0-1, the writer's own
    tags: tuple[str, ...] = ()
    status: FactStatus = "active"
    superseded_by: str = ""
    created_at: float = field(default_factory=now)


# The judge: given a new fact and the active facts on its topic, which of
# the existing ones does it contradict? Injected, because deciding that two
# sentences disagree is a model's job and must be testable with a lambda.
Judge = Callable[[Fact, tuple[Fact, ...]], tuple[str, ...]]


@runtime_checkable
class SkillBankService(Protocol):
    def add(self, fact: Fact, judge: Judge | None = None) -> tuple[str, tuple[str, ...]]:
        """Store the fact; supersede what it contradicts. Returns
        (fact id, ids superseded)."""
        ...

    def get(self, fact_id: str) -> Fact | None: ...

    def list(self, topic: str = "", status: FactStatus | None = "active") -> tuple[Fact, ...]: ...

    def search(self, text: str, k: int = 8) -> tuple[Fact, ...]: ...

    def supersede(self, old_id: str, by: str) -> None: ...

    def retract(self, fact_id: str) -> None: ...

    def render(self, k: int = 40, query: str = "") -> str:
        """The active facts as a skill document an agent can read."""
        ...
