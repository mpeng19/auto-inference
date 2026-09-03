"""A queryable database for GPU profiling traces, built for agents.

A kineto trace is hundreds of thousands of events and tens of megabytes; no
agent is going to read one. But trace debugging is pattern matching -- where
did the gap go, what ran between these two ops, is comm overlapping compute --
and those are queries. So a trace is ingested once into SQLite and answered
compactly, with timeline renders for the cases where a picture is faster.

This is the tool that turns "decode is at 28% of memory bandwidth"
(`docs/methodology.md` §8.3) into "and here is the kernel where it goes".

    tracedb ingest trace.json --db t.sqlite
    tracedb summary --db t.sqlite
    tracedb gaps attn_out mlp_in --min-gap 100 --db t.sqlite

Agents reach it as MCP tools rather than through the shell; see
`harness.profile` for how a fleet wires it in.
"""
__version__ = "0.1.0"
