"""MemoryService: the fleet's shared history of experiments.

**What makes this hard is the read, not the write.** Ten agents write roughly
twice an hour each; any store handles that. The question is whether a read
returns something that changes what an agent does next.

Three requirements the contract encodes, each earned:

**1. The history is a graph, not a list.** An experiment is *derived from*
another, *contradicts* another, *supersedes* another. Those edges are the only
way to answer "has anyone tried this and why did it fail" without re-reading
every experiment ever run. `Recall` therefore returns lineage, not just hits.

**2. Timelines branch.** Ten agents exploring in parallel produce ten
independent chains that occasionally merge or contradict. A flat timestamp
order flattens that into nonsense, so every experiment names its `timeline` and
its parents, and ordering questions are asked within or across timelines
explicitly.

**3. A list of hits is not enough, and this is measured.** `agent-db` ran the
obvious version -- retrieve relevant facts, put them in context -- and got a
**clean null out-of-sample**: facts performed the same as a placebo on tasks
that did not produce them. The one condition that separated was a *brief*: a
synthesised statement of what is currently known and what is still open. So
`recall()` returns a `Brief`, and returning raw hits is a degenerate case of
it. An implementation that only does semantic search satisfies the type and
will fail the eval, which is the correct outcome.

Anything implementing `MemoryService` must also be measurable against a
placebo, so `Brief` carries the token cost it will impose on the reader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .common import Provenance, new_id, now

# What one experiment says about another.
RelationKind = Literal[
    "derived_from",   # this idea came out of that result
    "replicates",     # ran the same thing again, deliberately
    "contradicts",    # measured the opposite
    "supersedes",     # same question, better method; prefer this one
    "refuted_by",     # a later result invalidated this one
]

Verdict = Literal["win", "loss", "neutral", "invalid", "pending"]


@dataclass(frozen=True)
class Experiment:
    """One evaluated idea. The node type of the history graph."""
    id: str = field(default_factory=lambda: new_id("exp"))
    agent_id: str = ""
    idea_id: str = ""
    timeline: str = "main"          # which parallel chain this belongs to
    ts: float = field(default_factory=now)

    hypothesis: str = ""            # what the agent expected, before running
    rationale: str = ""             # why it expected it
    stack_digest: str = ""          # the diff evaluated (simulator.InferenceStack)
    eval_digest: str = ""           # the measurement configuration

    verdict: Verdict = "pending"
    metrics: dict = field(default_factory=dict)   # bill_per_1k, n_star, ...
    baseline_metrics: dict = field(default_factory=dict)
    summary: str = ""               # what actually happened, in one paragraph
    tags: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)

    # Where the full detail lives. Memory holds the claim; ContextService holds
    # the transcript that produced it. Keeping them apart is what stops the
    # memory store from becoming a log nobody can query.
    trace_ref: str = ""


@dataclass(frozen=True)
class Relation:
    """A typed edge. `note` says why, because an untyped graph is a rumour."""
    src: str
    dst: str
    kind: RelationKind
    note: str = ""
    ts: float = field(default_factory=now)


@dataclass(frozen=True)
class Finding:
    """A claim that outlived the experiment that produced it.

    Separate from `Experiment` on purpose: an experiment is a thing that
    happened, a finding is a thing we believe. Findings are what a brief is
    built from, and they are what goes stale.
    """
    id: str = field(default_factory=lambda: new_id("fnd"))
    claim: str = ""
    kind: Literal["result", "method", "negative", "gotcha"] = "result"
    evidence: tuple[str, ...] = ()          # experiment ids
    confidence: float = 0.5
    provenance: Provenance = field(default_factory=Provenance)
    superseded_by: str = ""

    @property
    def live(self) -> bool:
        return not self.superseded_by


@dataclass(frozen=True)
class Recall:
    """A read request. Says what the agent is about to do, not what to match.

    `intent` is deliberately the agent's *next action* rather than a search
    string: "I am about to try raising chunked_prefill_size" retrieves the
    three agents who already did and the reason it failed, which a keyword
    query for "chunked prefill" would bury under every mention of prefill.
    """
    intent: str
    agent_id: str = ""
    idea_id: str = ""
    timeline: str = ""
    k: int = 8
    max_tokens: int = 4000          # the brief must fit the reader's budget
    include_negative: bool = True   # failures are the expensive knowledge
    since: float | None = None


@dataclass(frozen=True)
class Hit:
    experiment: Experiment
    score: float
    why: str = ""                   # which edge or term made it relevant
    lineage: tuple[str, ...] = ()    # path from the agent's own work, if any


@dataclass(frozen=True)
class Brief:
    """What `recall` returns: a synthesis, with the hits that back it.

    `text` is what goes in the agent's context. `hits` and `findings` are the
    audit trail -- an agent that wants detail follows `trace_ref` into the
    ContextService rather than being handed a transcript it did not ask for.
    """
    text: str
    hits: tuple[Hit, ...] = ()
    findings: tuple[Finding, ...] = ()
    open_questions: tuple[str, ...] = ()
    est_tokens: int = 0
    placebo: bool = False           # set by an A/B harness; never by the store


@runtime_checkable
class MemoryService(Protocol):
    """The whole surface. Four writes, two reads, one maintenance call."""

    # ── write ──
    def record(self, exp: Experiment) -> str: ...
    def relate(self, rel: Relation) -> None: ...
    def assert_finding(self, f: Finding) -> str: ...
    def supersede(self, finding_id: str, by: str) -> None: ...

    # ── read ──
    def recall(self, q: Recall) -> Brief: ...
    def lineage(self, experiment_id: str, depth: int = 4) -> tuple[Experiment, ...]:
        """Ancestors and descendants: how this idea came to be tried."""
        ...

    # ── maintenance ──
    def prune_stale(self, current_stack: str = "", current_eval: str = "") -> int:
        """Mark findings whose provenance no longer holds. Returns how many."""
        ...
