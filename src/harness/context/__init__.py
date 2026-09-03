"""Transcripts: what an agent actually did, one append-only file per idea.

`JsonlContext` implements `harness.contracts.ContextService`. The agent loop
opens a trace per idea, appends a `Turn` per phase and closes it with the
outcome; readers resolve the `trace_ref` a memory row carries.

    ctx = JsonlContext(root / "traces", session_id=...)
    ref = ctx.open(TraceMeta(agent_id=..., idea_id=...))
    ctx.append(ref, Turn(kind="thought", name="propose", content=..., data={...}))
    ctx.close(ref, outcome="won", cost_usd=1.5)
    ctx.slice(ref, kinds=("eval_result",)) / ctx.tail(ref) / ctx.stats(agent_id=...)

Writes `<root>/<trace id>.jsonl` (one envelope-wrapped turn per line, see
`docs/trace-schema.md`) and `<trace id>.meta.json` beside it. `harness.traces`
and `harness.timeline` read the same files.
"""
from .jsonl import JsonlContext

__all__ = ["JsonlContext"]
