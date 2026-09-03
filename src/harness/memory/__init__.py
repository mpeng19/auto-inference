"""The fleet's shared history: every experiment, its verdict, and a brief.

`SqliteMemory` implements `harness.contracts.MemoryService`. The agent loop
records an `Experiment` per priced attempt and a `Relation` to the attempt
before it; before every edit it asks for a `Brief` on what it is about to do,
and `harness tool recall` asks the same question from an agent's shell.

    mem = SqliteMemory(root / "memory.db")
    mem.record(Experiment(...)); mem.relate(Relation(src, dst, "derived_from"))
    brief = mem.recall(Recall(intent="...", agent_id=..., idea_id=...))   # Brief.text
    mem.assert_finding(Finding(...)); mem.prune_stale(current_stack=...)

One SQLite file per fleet (`<root>/memory.db`): experiments, typed edges,
findings, FTS5 indexes over both, and an `embeddings` table. Reads combine
text match (BM25 blended with sentence-embedding cosine when the optional
model is installed: `uv sync --group embeddings`; see `harness.embeddings`),
graph distance from the asking agent's own work, and recency.
"""
from .sqlite import SqliteMemory

__all__ = ["SqliteMemory"]
