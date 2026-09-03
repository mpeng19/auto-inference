"""A queryable database for GPU profiling traces, built for agents.

A kineto trace is hundreds of thousands of events and tens of megabytes; no
agent is going to read one. But trace debugging is pattern matching -- where
did the gap go, what ran between these two ops, is comm overlapping compute --
and those are queries. So a trace is ingested once into SQLite and answered
compactly, with timeline renders for the cases where a picture is faster.

This is the tool that turns "decode is at 28% of memory bandwidth"
(`docs/methodology.md` §8.3) into "and here is the kernel where it goes".

**What it takes.** A chrome-trace file as `torch.profiler` / SGLang's profiler
write it -- `.json`, `.json.gz` (what the server actually produces) or a
`.jsonl` stream of events. Only `ph: "X"` spans and `thread_name` metadata are
kept; everything else in the file is skipped.

**Entry points.**

    ingest      tracedb.ingest.ingest(trace_path, db_path) -> {"events", "steps", "span_us", "db"}
                Idempotent per file, not per database: ingesting twice into one
                db doubles the spans. `harness.profile.ingest` wraps it and picks
                the path below.
    query API   tracedb.query.<fn>(TraceStore(db), ...) -> dict | list[dict]
                summary, ops, ops_grouped, slowest, gaps, between, overlap,
                steps, step_diff, launches, gpu_idle; tracedb.render.timeline
                draws a window to PNG. Patterns: `*` is a wildcard and anchors
                (`attn*` is a prefix); no wildcard means substring (`attn_out`
                also finds `kernel_attn_out`).
    MCP server  python -m tracedb.mcp_server --db t.sqlite [--out out/]
                (also the `tracedb-mcp` console script). One server per
                database; tools are the query API under `trace_*` names.
                `harness.profile` writes the `--mcp-config` file that gives an
                agent `tracedb` (its own profile) and `tracedb_stock`.
    CLI         tracedb --db t.sqlite ingest trace.json.gz
                tracedb --db t.sqlite summary | ops | gaps | ... | render
                Same functions, JSON on stdout. Handy at a terminal; agents
                get the MCP surface instead.

**On disk.** Reads the trace file. Writes the SQLite file at `db_path`
(parent directories created) plus its `-wal`/`-shm` sidecars: the store opens
every connection in WAL mode, query-only ones included, so opening a database
is itself a write. A fleet keeps them under `<fleet root>/profiles/<stack
digest>.sqlite`; `simulate profile` under `<--out>/trace.sqlite`. Renders go
to `--out` (`out/` by default). `tracedb.synth` writes a synthetic fixture for
the tests; `tracedb.modal_trace` rents a GPU and writes `fixtures/real_trace.json`.
"""
__version__ = "0.1.0"
