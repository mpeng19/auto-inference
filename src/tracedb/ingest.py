"""Ingest chrome-trace JSON (kineto/torch.profiler/tensorboard) or JSONL-of-events into TraceStore."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .store import TraceStore

# No bare "cuda": it matched the CPU-side cuda_runtime/cuda_driver spans and
# could classify a launch thread as GPU on its first span.
_GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation", "mps")
_STEP_RE = re.compile(r"ProfilerStep#?\s*(\d+)")


def _iter_events(path: Path):
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip().rstrip(",")
                if line and line not in ("[", "]"):
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    else:
        data = json.loads(path.read_text())
        events = data.get("traceEvents", data) if isinstance(data, dict) else data
        yield from events


def ingest(trace_path: str | Path, db_path: str | Path) -> dict:
    trace_path, db_path = Path(trace_path), Path(db_path)
    st = TraceStore(db_path)
    rows, steps = [], []
    n = 0
    t_min, t_max = float("inf"), float("-inf")
    for ev in _iter_events(trace_path):
        ph = ev.get("ph")
        if ph == "M":  # metadata: thread/process names
            if ev.get("name") == "thread_name":
                st.set_track_label(ev.get("pid"), ev.get("tid", 0), name=(ev.get("args") or {}).get("name", "").strip())
            continue
        if ph != "X":
            continue
        ts, dur = ev.get("ts"), ev.get("dur", 0)
        if ts is None:
            continue
        name, cat = ev.get("name", "?"), ev.get("cat", "")
        m = _STEP_RE.search(name)
        if m:
            steps.append((int(m.group(1)), ts, dur))
        kind = "gpu" if any(c in cat for c in _GPU_CATS) else ("cpu" if cat else "")
        tid = ev.get("tid", 0)
        track = st.track_id(ev.get("pid", 0), tid, kind=kind)
        args = ev.get("args") or {}
        corr = args.get("correlation")
        if corr is None:
            corr = args.get("External id")        # id 0 is a real id
        rows.append((st.name_id(name), track, float(ts), float(dur), cat, corr))
        t_min, t_max = min(t_min, ts), max(t_max, ts + dur)
        n += 1
        if len(rows) >= 20000:
            st.add_spans(rows); rows = []
    if rows:
        st.add_spans(rows)
    st.conn.executemany("INSERT INTO steps(idx, ts, dur) VALUES (?,?,?)", steps)
    # mark gpu tracks by content when cat heuristics were empty
    st.conn.execute("""UPDATE tracks SET kind='gpu' WHERE kind='' AND id IN
        (SELECT DISTINCT track_id FROM spans WHERE cat LIKE '%kernel%' OR cat LIKE '%memcpy%')""")
    st.finalize({"source": str(trace_path), "events": n, "t0": t_min, "t1": t_max,
                 "steps": len(steps)})
    return {"events": n, "steps": len(steps), "span_us": (t_max - t_min) if n else 0, "db": str(db_path)}
