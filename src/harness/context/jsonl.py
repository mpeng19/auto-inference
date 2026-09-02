"""Reference ContextService: append-only JSONL, one file per trace.

Append-only is the whole design. An agent that crashes at turn 400 still has
399 turns on disk, and an orchestrator can `tail` a live file to notice an
agent drifting long before the run ends -- neither of which is true of a
transcript assembled at the end.

One file per trace, not one database, because the access pattern is *write one,
read one*: an agent appends to its own trace, and a reader resolves exactly one
`trace_ref` at a time. A shared table would serialise ten writers for no
benefit. The index is a small sidecar so `stats` does not have to open them.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Iterator
from dataclasses import asdict

from ..contracts.context import Slice, TraceMeta, Turn, TurnKind


class JsonlContext:
    """Reference implementation of `contracts.context.ContextService`."""

    #: Bump when a field changes meaning. Readers should refuse an unknown
    #: major version rather than guess.
    SCHEMA_VERSION = 1

    def __init__(self, root: str | pathlib.Path = "traces",
                 session_id: str = ""):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._seq: dict[str, int] = {}

    # ── layout ───────────────────────────────────────────────────────────
    def _path(self, ref: str) -> pathlib.Path:
        return self.root / f"{ref}.jsonl"

    def _meta_path(self, ref: str) -> pathlib.Path:
        return self.root / f"{ref}.meta.json"

    # ── write ────────────────────────────────────────────────────────────
    def open(self, meta: TraceMeta) -> str:
        self._meta_path(meta.id).write_text(json.dumps(asdict(meta), indent=1))
        self._path(meta.id).touch()
        return meta.id

    def append(self, trace_ref: str, turn: Turn) -> None:
        m = self.meta(trace_ref)
        self._seq[trace_ref] = seq = self._seq.get(trace_ref, -1) + 1
        record = {
            "v": self.SCHEMA_VERSION,
            "trace_id": trace_ref,
            "seq": seq,
            "session_id": self.session_id,
            "agent_id": getattr(m, "agent_id", ""),
            "idea_id": getattr(m, "idea_id", ""),
            "attempt": getattr(m, "attempt", 0),
            **asdict(turn),
        }
        with self._path(trace_ref).open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def close(self, trace_ref: str, outcome: str = "", cost_usd: float = 0.0) -> None:
        m = self.meta(trace_ref)
        if m is None:
            return
        import time
        d = asdict(m)
        d.update(ended_at=time.time(), outcome=outcome, cost_usd=cost_usd,
                 n_turns=sum(1 for _ in self.read(trace_ref)))
        self._meta_path(trace_ref).write_text(json.dumps(d, indent=1))

    # ── read ─────────────────────────────────────────────────────────────
    def meta(self, trace_ref: str) -> TraceMeta | None:
        p = self._meta_path(trace_ref)
        return TraceMeta(**json.loads(p.read_text())) if p.is_file() else None

    def read(self, trace_ref: str) -> Iterator[Turn]:
        p = self._path(trace_ref)
        if not p.is_file():
            return
        fields = {f.name for f in dataclasses.fields(Turn)}
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # The envelope is provenance for a downstream loader; in-process
                # readers want the Turn back, so drop what is not part of it.
                yield Turn(**{k: v for k, v in d.items() if k in fields})

    def slice(self, trace_ref: str, *, kinds: tuple[TurnKind, ...] = (),
              query: str = "", limit: int = 50) -> Slice:
        """The part that answers a question. The default read path.

        Handing an agent a whole transcript costs more than the answer is
        worth; what is nearly always wanted is the handful of turns that
        touched the thing being asked about.
        """
        q = query.lower()
        out = []
        for t in self.read(trace_ref):
            if kinds and t.kind not in kinds:
                continue
            if q and q not in (t.content + " " + t.name + " "
                               + json.dumps(t.data, default=str)).lower():
                continue
            out.append(t)
        reason = f"kinds={kinds or 'any'} query={query!r}"
        return Slice(trace_id=trace_ref, turns=tuple(out[-limit:]), reason=reason)

    def tail(self, trace_ref: str, n: int = 20) -> Slice:
        turns = list(self.read(trace_ref))[-n:]
        return Slice(trace_id=trace_ref, turns=tuple(turns), reason=f"last {n}")

    def stats(self, *, agent_id: str = "", idea_id: str = "") -> dict:
        n_traces = cost = turns = 0
        for mp in self.root.glob("*.meta.json"):
            m = TraceMeta(**json.loads(mp.read_text()))
            if agent_id and m.agent_id != agent_id:
                continue
            if idea_id and m.idea_id != idea_id:
                continue
            n_traces += 1
            cost += m.cost_usd
            turns += m.n_turns
        return {"traces": n_traces, "cost_usd": round(cost, 4), "turns": turns}
