"""The idea bank: where large ideas come from, and who holds which one.

The first fleets seeded agents with one-sentence hints ("raise the decode
batch the SLO permits") and got one-line diffs back: fifteen knob turns on
stock SGLang's admission logic, every one inside measurement noise. Ideas of
the size that move cost per token -- a different decode attention kernel, KV
compression that survives the accuracy gate, speculative decoding -- have to
arrive with their mechanism, their sources and their target files already
named. That is what an `IdeaRecord` is. Producers are the inference
engineering book and the arXiv feed; the consumer is the fleet, which claims
records so that no two agents build the same mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .agent import Idea
from .common import new_id, now


def content_id(title: str, mechanism: str = "") -> str:
    """A record's id is a hash of what it says, so the same idea imported
    twice, from a re-run harvest or a second machine, is one record. The
    first 12 hex digits of SHA-1 over the normalised title and mechanism."""
    import hashlib

    norm = " ".join((title or "").split()).lower() + "\n" + " ".join((mechanism or "").split()).lower()
    return "idea_" + hashlib.sha1(norm.encode()).hexdigest()[:12]

Scale = Literal["kernel", "architecture", "memory", "scheduler",
                "parallelism", "numerics", "other"]
BankStatus = Literal["available", "claimed", "tried", "retired"]


@dataclass(frozen=True)
class IdeaRecord:
    """One mechanism worth an agent-day. Larger than an `Idea`: it carries
    where it came from and what it would take, so the agent that claims it
    starts from a design rather than a hint."""
    id: str = field(default_factory=lambda: new_id("bank"))
    title: str = ""
    # How it lowers cost per output token, in one paragraph. The thing an
    # agent implements.
    mechanism: str = ""
    # The one sentence the experiment can be wrong about.
    hypothesis: str = ""
    # "book:ch7 p.212-219" | "arxiv:2309.06180" | "human"
    source: str = ""
    source_title: str = ""
    url: str = ""
    scale: Scale = "kernel"
    targets: tuple[str, ...] = ()       # sglang paths it expects to touch
    expected_gain: str = ""             # "10-30% decode step at B>=8", with basis
    risks: str = ""                     # numerics, accuracy, what breaks
    prerequisites: str = ""             # what must be true of the stack first
    tags: tuple[str, ...] = ()
    status: BankStatus = "available"
    claimed_by: str = ""                # agent id while claimed
    experiment_ids: tuple[str, ...] = ()
    created_at: float = field(default_factory=now)

    def as_idea(self, timeline: str = "main") -> Idea:
        """The seed an agent runs. `seeded_by` points back here so the
        outcome can be recorded against the bank."""
        return Idea(title=self.title[:80], hypothesis=self.hypothesis or self.mechanism,
                    targets=self.targets, seeded_by=self.id, timeline=timeline)

    @property
    def text(self) -> str:
        """What similarity is judged on."""
        return " ".join((self.title, self.mechanism, self.hypothesis, " ".join(self.tags)))


@runtime_checkable
class IdeaBankService(Protocol):
    """Records in, one distinct claim out per agent."""

    def add(self, rec: IdeaRecord) -> str: ...

    def get(self, rec_id: str) -> IdeaRecord | None: ...

    def list(self, status: BankStatus | None = None,
             scale: Scale | None = None) -> tuple[IdeaRecord, ...]: ...

    def search(self, text: str, k: int = 8) -> tuple[IdeaRecord, ...]: ...

    def claim(self, agent_id: str, avoid: tuple[str, ...] = (),
              live_scales: tuple[str, ...] = (), seed: str = "") -> IdeaRecord | None:
        """Hand `agent_id` one available record and mark it claimed.

        Without `seed`: the record least like `avoid` (the texts of ideas
        already live or tried), ties broken toward a `scale` not in
        `live_scales` -- diversity across a fleet. With `seed`: the record
        most like the seed text among those not close to `avoid` -- an
        operator or an agent steering toward a direction. None when nothing
        is available."""
        ...

    def related(self, rec_id: str, k: int = 5) -> tuple[IdeaRecord, ...]:
        """The k records nearest to `rec_id` by text, any status: what has
        been tried near this idea, and what could follow it."""
        ...

    def seed(self, source: str = "book") -> int:
        """Load a built-in seed set into the bank; returns how many records
        it added or refreshed. Content-addressed ids make this idempotent."""
        ...

    def release(self, rec_id: str, status: BankStatus = "available") -> None: ...

    def record_outcome(self, rec_id: str, experiment_id: str,
                       status: BankStatus = "tried") -> None: ...

    def count(self, status: BankStatus | None = None) -> int: ...
