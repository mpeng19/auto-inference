"""Pattern queries over an ingested trace: op stats, gaps, op-pair distances, overlap, steps."""
from __future__ import annotations

import re as _re
import statistics as st
from collections import defaultdict

from .store import TraceStore


def canon_name(name: str) -> str:
    """Collapse templated/mangled kernel names into readable groups."""
    n = name
    n = _re.sub(r"<.*>", "<>", n)                      # template args
    n = _re.sub(r"\(.*\)", "", n)                      # signatures
    n = _re.sub(r"^void\s+", "", n)
    n = _re.sub(r"0x[0-9a-f]+", "0xN", n)
    n = _re.sub(r"#\d+", "#N", n)
    return n.strip()[:120]


def _like(pattern: str) -> str:
    p = pattern.replace("*", "%")
    return p if "%" in p else f"%{p}%"


def name_ids(store: TraceStore, pattern: str) -> dict[int, str]:
    return {r["id"]: r["name"] for r in
            store.conn.execute("SELECT id, name FROM names WHERE name LIKE ?", (_like(pattern),))}


def ops(store: TraceStore, pattern: str = "%", limit: int = 30) -> list[dict]:
    rows = store.conn.execute("""
        SELECT n.name, COUNT(*) cnt, SUM(s.dur) total_us, AVG(s.dur) mean_us, MAX(s.dur) max_us,
               COUNT(DISTINCT s.track_id) tracks
        FROM spans s JOIN names n ON n.id = s.name_id
        WHERE n.name LIKE ? GROUP BY n.id ORDER BY total_us DESC LIMIT ?""",
        (_like(pattern), limit)).fetchall()
    return [dict(r) for r in rows]


def ops_grouped(store: TraceStore, pattern: str = "%", limit: int = 25) -> list[dict]:
    """Like ops(), but aggregated over canonicalized names (templated kernels grouped)."""
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for r in ops(store, pattern, limit=100000):
        c = canon_name(r["name"])
        a = agg[c]; a[0] += r["cnt"]; a[1] += r["total_us"]; a[2] = max(a[2], r["max_us"])
    rows = [{"name": k, "cnt": v[0], "total_us": round(v[1], 1), "max_us": v[2]} for k, v in agg.items()]
    rows.sort(key=lambda r: -r["total_us"])
    return rows[:limit]


def step_diff(store: TraceStore, idx: int, top: int = 15) -> dict:
    """Per-op time in step `idx` vs the median across all steps — answers 'why is step N slow'."""
    srow = store.conn.execute("SELECT ts, dur FROM steps WHERE idx=?", (idx,)).fetchone()
    if srow is None:
        return {"error": f"no step {idx}"}
    all_steps = store.conn.execute("SELECT idx, ts, dur FROM steps ORDER BY idx").fetchall()

    def per_op(t0, t1):
        d = defaultdict(float)
        for r in store.conn.execute(
                "SELECT n.name, s.dur FROM spans s JOIN names n ON n.id=s.name_id "
                "WHERE s.ts >= ? AND s.ts < ? AND n.name NOT LIKE 'ProfilerStep%'", (t0, t1)):
            d[canon_name(r["name"])] += r["dur"]
        return d

    target = per_op(srow["ts"], srow["ts"] + srow["dur"])
    others = [per_op(r["ts"], r["ts"] + r["dur"]) for r in all_steps if r["idx"] != idx]
    med = {}
    for k in target:
        vals = sorted(o.get(k, 0.0) for o in others)
        med[k] = vals[len(vals) // 2] if vals else 0.0
    diffs = [{"op": k, "step_us": round(v, 1), "median_us": round(med[k], 1),
              "delta_us": round(v - med[k], 1)} for k, v in target.items()]
    diffs.sort(key=lambda r: -r["delta_us"])
    return {"step": idx, "step_dur_us": srow["dur"],
            "median_step_dur_us": sorted(r["dur"] for r in all_steps)[len(all_steps) // 2],
            "top_regressions": diffs[:top]}


def gaps(store: TraceStore, after: str, before: str, min_gap_us: float = 0,
         limit: int = 50) -> list[dict]:
    """Adjacent-pair gaps on the same track: span matching `after` immediately followed by one
    matching `before`, with idle time between them."""
    a, b = name_ids(store, after), name_ids(store, before)
    if not a or not b:
        return []
    rows = store.conn.execute(f"""
        WITH seq AS (
          SELECT s.id, s.name_id, s.track_id, s.ts, s.dur,
                 LEAD(s.id) OVER w nid, LEAD(s.name_id) OVER w nname,
                 LEAD(s.ts) OVER w nts
          FROM spans s WINDOW w AS (PARTITION BY s.track_id ORDER BY s.ts))
        SELECT seq.id a_id, seq.ts a_ts, seq.dur a_dur, seq.track_id,
               nid b_id, nts b_ts, (nts - seq.ts - seq.dur) gap_us
        FROM seq
        WHERE seq.name_id IN ({",".join(map(str, a))}) AND nname IN ({",".join(map(str, b))})
          AND (nts - seq.ts - seq.dur) >= ?
        ORDER BY gap_us DESC LIMIT ?""", (min_gap_us, limit)).fetchall()
    out = [dict(r) for r in rows]
    if out:
        gs = [r["gap_us"] for r in out]
        head = {"summary": {"instances": len(out), "gap_mean_us": round(st.mean(gs), 1),
                            "gap_max_us": max(gs), "after": after, "before": before}}
        return [head, *out]
    return out


def between(store: TraceStore, a_pat: str, b_pat: str, limit: int = 30, max_a: int = 3000) -> list[dict]:
    """For each span matching a, the NEXT span matching b anywhere after it (any track):
    latency b.start - a.end and how many spans intervene on a's track."""
    a, b = name_ids(store, a_pat), name_ids(store, b_pat)
    if not a or not b:
        return []
    a_rows = store.conn.execute(
        f"SELECT id, track_id, ts, dur FROM spans WHERE name_id IN ({','.join(map(str, a))}) "
        f"ORDER BY ts LIMIT ?", (max_a,)).fetchall()
    out = []
    for ar in a_rows:
        a_end = ar["ts"] + ar["dur"]
        br = store.conn.execute(
            f"SELECT id, track_id, ts FROM spans WHERE name_id IN ({','.join(map(str, b))}) "
            f"AND ts >= ? ORDER BY ts LIMIT 1", (a_end,)).fetchone()
        if br is None:
            continue
        n_between = store.conn.execute(
            "SELECT COUNT(*) c FROM spans WHERE track_id=? AND ts>=? AND ts<?",
            (ar["track_id"], a_end, br["ts"])).fetchone()["c"]
        out.append({"a_id": ar["id"], "a_end": a_end, "b_id": br["id"], "b_ts": br["ts"],
                    "latency_us": round(br["ts"] - a_end, 1), "spans_between_on_a_track": n_between})
    out.sort(key=lambda r: -r["latency_us"])
    if out:
        ls = [r["latency_us"] for r in out]
        head = {"summary": {"instances": len(out), "latency_mean_us": round(st.mean(ls), 1),
                            "latency_p50_us": round(st.median(ls), 1), "latency_max_us": max(ls)}}
        return [head, *out[:limit]]
    return out


def overlap(store: TraceStore, a_pat: str, b_pat: str, limit: int = 20) -> dict:
    """Total time spans matching a overlap spans matching b on DIFFERENT tracks (e.g. compute vs
    comm), plus the largest overlapping pairs."""
    def spans_of(pat):
        ids = name_ids(store, pat)
        if not ids:
            return []
        return store.conn.execute(
            f"SELECT id, track_id, ts, dur, name_id FROM spans WHERE name_id IN ({','.join(map(str, ids))}) "
            "ORDER BY ts LIMIT 200000").fetchall()
    A, B = spans_of(a_pat), spans_of(b_pat)
    total_a = sum(r["dur"] for r in A)
    # merge B intervals (per differing track) so one A span overlapping many B spans counts once
    def merged(intervals):
        out = []
        for s, e in sorted(intervals):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    a_tracks = {r["track_id"] for r in A}
    b_merged = merged([(r["ts"], r["ts"] + r["dur"]) for r in B if r["track_id"] not in a_tracks or len(a_tracks) > 1])
    ov_total, pairs = 0.0, []
    j0 = 0
    for ar in A:
        a0, a1 = ar["ts"], ar["ts"] + ar["dur"]
        while j0 < len(b_merged) and b_merged[j0][1] < a0:
            j0 += 1
        j = j0
        while j < len(b_merged) and b_merged[j][0] < a1:
            ov = min(a1, b_merged[j][1]) - max(a0, b_merged[j][0])
            if ov > 0:
                ov_total += ov
                pairs.append((ov, ar["id"], None, max(a0, b_merged[j][0])))
            j += 1
    pairs.sort(key=lambda x: -x[0])
    return {"a": a_pat, "b": b_pat, "a_spans": len(A), "b_spans": len(B),
            "a_total_us": round(total_a, 1), "overlap_us": round(ov_total, 1),
            "overlap_frac_of_a": round(ov_total / total_a, 4) if total_a else 0,
            "exposed_us": round(total_a - ov_total, 1),   # a-time NOT hidden under b (e.g. exposed comm)
            "top_pairs": [{"overlap_us": round(o, 1), "a_id": ai, "b_id": bi, "at_ts": t}
                          for o, ai, bi, t in pairs[:limit]]}


def steps(store: TraceStore) -> dict:
    rows = [dict(r) for r in store.conn.execute("SELECT idx, ts, dur FROM steps ORDER BY idx")]
    if not rows:
        return {"steps": 0, "note": "no ProfilerStep markers found"}
    durs = [r["dur"] for r in rows]
    mean, sd = st.mean(durs), (st.pstdev(durs) or 1e-9)
    outliers = [r | {"z": round((r["dur"] - mean) / sd, 1)} for r in rows if abs(r["dur"] - mean) > 2 * sd]
    return {"steps": len(rows), "dur_mean_us": round(mean, 1), "dur_p50_us": round(st.median(durs), 1),
            "dur_min_us": min(durs), "dur_max_us": max(durs), "outliers": outliers, "first": rows[:3]}


def slowest(store: TraceStore, pattern: str = "%", k: int = 20) -> list[dict]:
    rows = store.conn.execute("""
        SELECT s.id, n.name, s.ts, s.dur, t.name track, t.kind
        FROM spans s JOIN names n ON n.id=s.name_id JOIN tracks t ON t.id=s.track_id
        WHERE n.name LIKE ? ORDER BY s.dur DESC LIMIT ?""", (_like(pattern), k)).fetchall()
    return [dict(r) for r in rows]


def launches(store: TraceStore, limit: int = 20) -> dict:
    """CPU->GPU delay via kineto correlation ids: runtime launch span (cpu) paired with its kernel
    span (gpu). NOTE: this measures QUEUE DELAY, not launch overhead — a large p50 means the CPU
    runs far ahead of the GPU (deep async queue, usually healthy); look at the *distribution* and
    at which kernels sit at the tail. Near-zero delays with a starved GPU indicate CPU-bound."""
    rows = store.conn.execute("""
        SELECT c.id c_id, cn.name c_name, c.ts c_ts, c.dur c_dur,
               g.id g_id, gn.name g_name, g.ts g_ts, (g.ts - c.ts - c.dur) lat_us
        FROM spans c JOIN tracks ct ON ct.id=c.track_id AND ct.kind != 'gpu'
             JOIN spans g ON g.corr = c.corr AND g.id != c.id
             JOIN tracks gt ON gt.id=g.track_id AND gt.kind = 'gpu'
             JOIN names cn ON cn.id=c.name_id JOIN names gn ON gn.id=g.name_id
        WHERE c.corr IS NOT NULL LIMIT 100000""").fetchall()
    if not rows:
        return {"pairs": 0, "note": "no correlation-linked cpu->gpu pairs found"}
    lats = sorted(r["lat_us"] for r in rows)
    slow = sorted(rows, key=lambda r: -r["lat_us"])[:limit]
    return {"pairs": len(rows), "lat_p50_us": round(lats[len(lats) // 2], 1),
            "lat_p99_us": round(lats[int(len(lats) * 0.99)], 1), "lat_max_us": round(lats[-1], 1),
            "slowest": [{"kernel": canon_name(r["g_name"]), "launch": r["c_name"],
                         "lat_us": round(r["lat_us"], 1), "kernel_ts": r["g_ts"]} for r in slow]}


def gpu_idle(store: TraceStore, min_gap_us: float = 50, limit: int = 25) -> dict:
    """Gaps on GPU tracks, each blamed on what the CPU was doing during the gap."""
    gpu_tracks = [r["id"] for r in store.conn.execute("SELECT id FROM tracks WHERE kind='gpu'")]
    out, blame_total = [], defaultdict(float)
    for tid in gpu_tracks:
        rows = store.conn.execute("""
            WITH seq AS (SELECT ts, dur, LEAD(ts) OVER (ORDER BY ts) nts
                         FROM spans WHERE track_id=?)
            SELECT ts + dur gap_start, nts gap_end, (nts - ts - dur) gap_us
            FROM seq WHERE (nts - ts - dur) >= ?""", (tid, min_gap_us)).fetchall()
        for g in rows:
            cpu = store.conn.execute("""
                SELECT n.name, SUM(MIN(s.ts + s.dur, ?) - MAX(s.ts, ?)) cover
                FROM spans s JOIN tracks t ON t.id = s.track_id AND t.kind != 'gpu'
                     JOIN names n ON n.id = s.name_id
                WHERE s.ts < ? AND s.ts + s.dur > ? AND n.name NOT LIKE 'ProfilerStep%'
                      AND n.name NOT LIKE 'PyTorch Profiler%' AND n.name NOT LIKE '%profiler%'
                GROUP BY n.id ORDER BY cover DESC LIMIT 3""",
                (g["gap_end"], g["gap_start"], g["gap_end"], g["gap_start"])).fetchall()
            blamed = canon_name(cpu[0]["name"]) if cpu else "(nothing on CPU)"
            blame_total[blamed] += g["gap_us"]
            out.append({"track_id": tid, "gap_start": g["gap_start"], "gap_us": round(g["gap_us"], 1),
                        "cpu_during_gap": [{"op": canon_name(c["name"]), "cover_us": round(c["cover"], 1)}
                                           for c in cpu]})
    out.sort(key=lambda r: -r["gap_us"])
    return {"gaps": len(out), "idle_total_us": round(sum(r["gap_us"] for r in out), 1),
            "blame_by_op": sorted(({"op": k, "idle_us": round(v, 1)} for k, v in blame_total.items()),
                                  key=lambda r: -r["idle_us"])[:10],
            "largest": out[:limit]}


def summary(store: TraceStore) -> dict:
    meta = store.meta()
    span = (meta.get("t1", 0) - meta.get("t0", 0)) or 1
    tracks = []
    for t in store.conn.execute("SELECT * FROM tracks"):
        ivs = store.conn.execute("SELECT ts, ts+dur e FROM spans WHERE track_id=? ORDER BY ts", (t["id"],)).fetchall()
        if not ivs:
            continue
        busy, cur_s, cur_e = 0.0, ivs[0]["ts"], ivs[0]["e"]
        for r in ivs[1:]:  # merged-interval busy time (nested spans count once)
            if r["ts"] <= cur_e:
                cur_e = max(cur_e, r["e"])
            else:
                busy += cur_e - cur_s
                cur_s, cur_e = r["ts"], r["e"]
        busy += cur_e - cur_s
        tracks.append({"track_id": t["id"], "name": t["name"] or f"pid{t['pid']}/tid{t['tid']}",
                       "kind": t["kind"], "spans": len(ivs), "busy_frac": round(busy / span, 3)})
    return {"meta": meta, "tracks": sorted(tracks, key=lambda x: -x["busy_frac"]),
            "top_ops": ops(store, "%", 12)}
