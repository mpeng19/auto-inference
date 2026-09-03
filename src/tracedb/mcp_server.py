"""MCP server exposing tracedb to agents (stdio).

The harness writes this into an agent's MCP config for it (`harness.profile`),
naming the running interpreter so it works without an install:

  {"command": "<python>", "args": ["-m", "tracedb.mcp_server", "--db", "<t.sqlite>"]}

By hand, from a checkout:

  claude mcp add tracedb -- uv run tracedb-mcp --db /path/to/t.sqlite

One server per database, so an agent can hold its own profile and stock's side
by side under different tool prefixes. Every result is compact JSON;
`trace_render` writes a PNG and returns its path so the agent can open the
image of exactly the window a query surfaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import query as Q
from .render import timeline
from .store import TraceStore

mcp = MCPServer("tracedb")
_state: dict = {"db": "trace.sqlite", "out": "out"}


def _store() -> TraceStore:
    return TraceStore(_state["db"])


def _j(x) -> str:
    return json.dumps(x, default=str)


@mcp.tool()
def trace_summary() -> str:
    """Trace overview: time span, tracks with busy fractions, top ops by total time."""
    return _j(Q.summary(_store()))


@mcp.tool()
def trace_ops(pattern: str = "*") -> str:
    """Duration stats per op name matching a substring/glob pattern (* = wildcard)."""
    return _j(Q.ops(_store(), pattern))


@mcp.tool()
def trace_gaps(after: str, before: str, min_gap_us: float = 0, limit: int = 25) -> str:
    """All instances where an op matching `after` is immediately followed on the same track by an
    op matching `before` with idle time between them; sorted by gap, with summary stats."""
    return _j(Q.gaps(_store(), after, before, min_gap_us, limit))


@mcp.tool()
def trace_between(a: str, b: str, limit: int = 20) -> str:
    """For each op matching `a`, the latency until the NEXT op matching `b` (any track), with
    how many spans intervene. Use for launch latency or cross-stream causality."""
    return _j(Q.between(_store(), a, b, limit))


@mcp.tool()
def trace_overlap(a: str, b: str) -> str:
    """How much of pattern `a`'s time overlaps pattern `b` on other tracks (merged intervals).
    E.g. a='nccl*', b='kernel_*' answers how well comm is hidden under compute."""
    return _j(Q.overlap(_store(), a, b))


@mcp.tool()
def trace_steps() -> str:
    """ProfilerStep boundaries, duration stats, and outlier (slow/fast) steps."""
    return _j(Q.steps(_store()))


@mcp.tool()
def trace_ops_grouped(pattern: str = "*") -> str:
    """Duration stats with templated kernel names canonicalized/grouped (use for real GPU traces)."""
    return _j(Q.ops_grouped(_store(), pattern))


@mcp.tool()
def trace_step_diff(idx: int) -> str:
    """Per-op time in step `idx` vs the median step — answers 'why is step N slow'."""
    return _j(Q.step_diff(_store(), idx))


@mcp.tool()
def trace_launches() -> str:
    """CPU->GPU kernel launch latency via correlation ids (p50/p99/max + slowest pairs).
    High latency means the GPU is waiting on the CPU."""
    return _j(Q.launches(_store()))


@mcp.tool()
def trace_gpu_idle(min_gap_us: float = 50) -> str:
    """Gaps on GPU streams, each blamed on the CPU activity covering the gap — answers
    'why is the GPU idle here', aggregated per blamed op."""
    return _j(Q.gpu_idle(_store(), min_gap_us))


@mcp.tool()
def trace_slowest(pattern: str = "*", k: int = 20) -> str:
    """Top-k individual span instances by duration for ops matching pattern."""
    return _j(Q.slowest(_store(), pattern, k))


@mcp.tool()
def trace_render(t0: float, t1: float, tracks: str = "*", marks: list[float] | None = None) -> str:
    """Render the [t0, t1] window (microseconds, from span timestamps returned by other tools)
    as a timeline PNG; returns the file path — open it with an image viewer/Read tool."""
    out = Path(_state["out"]) / f"timeline_{int(t0)}_{int(t1)}.png"
    return _j(timeline(_store(), t0, t1, out, tracks.replace("*", "%"), marks=marks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="trace.sqlite")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    _state["db"], _state["out"] = a.db, a.out
    mcp.run()


if __name__ == "__main__":
    main()
