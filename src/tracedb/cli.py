"""tracedb CLI: ingest a trace once, then query it cheaply.

  tracedb ingest trace.json --db t.sqlite
  tracedb summary --db t.sqlite
  tracedb ops 'attn*' | gaps 'attn_out' 'mlp_in' --min-gap 100 | between a b
  tracedb overlap 'nccl*' 'gemm*' | steps | slowest '*' -k 10
  tracedb render --t0 5000 --t1 20000 --out win.png
"""
from __future__ import annotations

import argparse
import json

from . import query as Q
from .ingest import ingest
from .render import timeline
from .store import TraceStore


def _p(x):
    print(json.dumps(x, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracedb")
    ap.add_argument("--db", default="trace.sqlite")
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("ingest"); g.add_argument("trace")
    sp.add_parser("summary")
    g = sp.add_parser("ops"); g.add_argument("pattern", nargs="?", default="%"); g.add_argument("--grouped", action="store_true")
    g = sp.add_parser("gaps"); g.add_argument("after"); g.add_argument("before")
    g.add_argument("--min-gap", type=float, default=0); g.add_argument("--limit", type=int, default=25)
    g = sp.add_parser("between"); g.add_argument("a"); g.add_argument("b"); g.add_argument("--limit", type=int, default=20)
    g = sp.add_parser("overlap"); g.add_argument("a"); g.add_argument("b")
    sp.add_parser("steps")
    g = sp.add_parser("stepdiff"); g.add_argument("idx", type=int)
    sp.add_parser("launches")
    g = sp.add_parser("idle"); g.add_argument("--min-gap", type=float, default=50)
    g = sp.add_parser("slowest"); g.add_argument("pattern", nargs="?", default="%"); g.add_argument("-k", type=int, default=20)
    g = sp.add_parser("render"); g.add_argument("--t0", type=float, required=True); g.add_argument("--t1", type=float, required=True)
    g.add_argument("--out", default="out/timeline.png"); g.add_argument("--tracks", default="%")
    g.add_argument("--mark", type=float, action="append", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "ingest":
        _p(ingest(a.trace, a.db)); return
    st = TraceStore(a.db)
    if a.cmd == "summary": _p(Q.summary(st))
    elif a.cmd == "ops": _p(Q.ops_grouped(st, a.pattern) if a.grouped else Q.ops(st, a.pattern))
    elif a.cmd == "gaps": _p(Q.gaps(st, a.after, a.before, a.min_gap, a.limit))
    elif a.cmd == "between": _p(Q.between(st, a.a, a.b, a.limit))
    elif a.cmd == "overlap": _p(Q.overlap(st, a.a, a.b))
    elif a.cmd == "steps": _p(Q.steps(st))
    elif a.cmd == "stepdiff": _p(Q.step_diff(st, a.idx))
    elif a.cmd == "launches": _p(Q.launches(st))
    elif a.cmd == "idle": _p(Q.gpu_idle(st, a.min_gap))
    elif a.cmd == "slowest": _p(Q.slowest(st, a.pattern, a.k))
    elif a.cmd == "render": _p(timeline(st, a.t0, a.t1, a.out, a.tracks, marks=a.mark))


if __name__ == "__main__":
    main()
