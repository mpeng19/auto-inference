"""Finding, reading and exporting agent traces.

Traces are the debugging record: what an agent called, in what order, how long
it waited, where its tokens went. They are consumed two ways -- a person
reading one to understand a run, and a profile database ingesting all of them
-- so this module does exactly those two things and nothing else.

`docs/trace-schema.md` is the contract. Nothing here interprets a trace beyond
grouping and filtering; anything cleverer belongs in whatever loads them.
"""
from __future__ import annotations

import json
import pathlib
import shutil
from dataclasses import dataclass, field

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TraceFile:
    path: pathlib.Path
    trace_id: str
    session_id: str = ""
    agent_id: str = ""
    idea_id: str = ""
    attempt: int = 0
    n_turns: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    outcome: str = ""
    cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.ended_at - self.started_at) if self.ended_at else 0.0


def _roots(root: str | pathlib.Path | None) -> list[pathlib.Path]:
    """Where traces live: an explicit root, else every fleet under ./agents."""
    if root:
        p = pathlib.Path(root)
        return [p / "traces"] if (p / "traces").is_dir() else [p]
    out = []
    for base in (pathlib.Path.cwd() / "agents", pathlib.Path.home() / ".auto-inference"):
        if base.is_dir():
            out.extend(sorted(d for d in base.rglob("traces") if d.is_dir()))
    return out


def find(root: str | pathlib.Path | None = None, *, session_id: str = "",
         agent_id: str = "", outcome: str = "", min_turns: int = 0) -> list[TraceFile]:
    """Every trace, newest first, with its sidecar folded in.

    `outcome` and `min_turns` exist because an API outage on build-4 left
    4,700 three-line error traces beside the 60 that record work; without
    a filter the listing's first page is all outage."""
    out: list[TraceFile] = []
    for d in _roots(root):
        for f in sorted(d.glob("*.jsonl")):
            meta = {}
            side = f.with_suffix("").with_suffix(".meta.json")
            if not side.is_file():
                side = f.parent / (f.stem + ".meta.json")
            if side.is_file():
                try:
                    meta = json.loads(side.read_text())
                except json.JSONDecodeError:
                    meta = {}
            first = _first_line(f)
            sid = first.get("session_id", "")
            aid = meta.get("agent_id") or first.get("agent_id", "")
            if session_id and sid != session_id:
                continue
            if agent_id and aid != agent_id:
                continue
            if outcome and meta.get("outcome", "") != outcome:
                continue
            n_turns = int(meta.get("n_turns", 0) or 0) or _count(f)
            if min_turns and n_turns < min_turns:
                continue
            out.append(TraceFile(
                path=f, trace_id=meta.get("id") or f.stem, session_id=sid,
                agent_id=aid, idea_id=meta.get("idea_id", ""),
                attempt=int(meta.get("attempt", 0) or 0),
                n_turns=n_turns,
                started_at=float(meta.get("started_at", 0) or 0),
                ended_at=float(meta.get("ended_at", 0) or 0),
                outcome=meta.get("outcome", ""),
                cost_usd=float(meta.get("cost_usd", 0) or 0), meta=meta))
    return sorted(out, key=lambda t: -t.started_at)


def read(path: str | pathlib.Path, *, kinds: tuple[str, ...] = (),
         query: str = "", limit: int = 0) -> list[dict]:
    """Records, filtered and normalised.

    Every record comes back with `v`, `seq`, `trace_id` and the agent/idea
    ids, filled from position and the sidecar when a line lacks them, so a
    loader can rely on the envelope. A newer-than-known `v` is refused:
    guessing at a field whose meaning changed is worse than stopping.
    """
    path = pathlib.Path(path)
    side = path.parent / (path.stem + ".meta.json")
    meta = {}
    if side.is_file():
        try:
            meta = json.loads(side.read_text())
        except json.JSONDecodeError:
            meta = {}
    q = query.lower()
    out = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        v = d.get("v", 0)
        if v > SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema v{v} is newer than this reader (v{SCHEMA_VERSION}). "
                "Upgrade rather than guessing at fields whose meaning changed.")
        d.setdefault("v", 0)
        d.setdefault("seq", i)
        d.setdefault("trace_id", meta.get("id") or path.stem)
        for k in ("agent_id", "idea_id", "attempt"):
            d.setdefault(k, meta.get(k, 0 if k == "attempt" else ""))
        d.setdefault("session_id", "")
        if kinds and d.get("kind") not in kinds:
            continue
        if q and q not in json.dumps(d, default=str).lower():
            continue
        out.append(d)
    return out[-limit:] if limit else out


def export(out_dir: str | pathlib.Path, root: str | pathlib.Path | None = None,
           *, session_id: str = "") -> dict:
    """Copy traces into one directory, with a manifest a loader can verify against.

    Files are copied unchanged. The manifest exists so an ingest can prove it
    read every line rather than discovering a truncated file later.
    """
    dest = pathlib.Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files, total = [], 0
    for t in find(root, session_id=session_id):
        shutil.copy2(t.path, dest / t.path.name)
        side = t.path.parent / (t.path.stem + ".meta.json")
        if side.is_file():
            shutil.copy2(side, dest / side.name)
        n = _count(t.path)
        total += n
        files.append({"file": t.path.name, "trace_id": t.trace_id,
                      "session_id": t.session_id, "agent_id": t.agent_id,
                      "lines": n, "outcome": t.outcome})
    manifest = {"schema_version": SCHEMA_VERSION, "traces": len(files),
                "lines": total, "files": files}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _first_line(p: pathlib.Path) -> dict:
    with p.open() as f:
        for line in f:
            if line.strip():
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return {}
    return {}


def _count(p: pathlib.Path) -> int:
    with p.open() as f:
        return sum(1 for line in f if line.strip())
