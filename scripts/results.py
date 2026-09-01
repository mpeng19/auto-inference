"""Read benchmark runs back out of the results Volume and compare them.

    uv run modal run scripts/results.py::ls
    uv run modal run scripts/results.py::show --name <file>
    uv run modal run scripts/results.py::compare --a <file> --b <file>
    uv run modal run scripts/results.py::pull          # copy all runs locally

Every run is a self-describing JSON blob: config, workload, trace digest,
overlay digest, provenance, per-request records and metrics. Comparisons are
only meaningful when the trace digests match -- otherwise two configs were
measured against different traffic, which is the easiest way to manufacture a
fake improvement.
"""
from __future__ import annotations

import json
import pathlib

import modal

from autoinf.modal_app import image, results_vol

app = modal.App("auto-inference-results", image=image)
LOCAL = pathlib.Path(__file__).resolve().parents[1] / "runs"


@app.function(volumes={"/results": results_vol}, timeout=600)
def _ls() -> list[dict]:
    import os
    d = pathlib.Path("/results/runs")
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
            out.append({
                "file": f.name,
                "note": r.get("note"),
                "status": r.get("status"),
                "model": (r.get("serving") or {}).get("model", "")[:38],
                "overlay": (r.get("overlay") or {}).get("digest"),
                "n_workloads": len(r.get("runs", [])),
                "size_kb": round(f.stat().st_size / 1024),
            })
        except Exception as e:
            out.append({"file": f.name, "error": str(e)[:80]})
    return out


@app.function(volumes={"/results": results_vol}, timeout=600)
def _get(name: str) -> dict:
    return json.loads(pathlib.Path(f"/results/runs/{name}").read_text())


@app.function(volumes={"/results": results_vol}, timeout=900)
def _all() -> list[dict]:
    d = pathlib.Path("/results/runs")
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.is_dir() else []


def _table(runs: list[dict]) -> None:
    hdr = (f"{'workload':<15}{'goodput':>9}{'thruput':>10}{'p99 TTFT':>11}"
           f"{'p99 TPOT':>10}{'ok':>6}{'fail':>6}{'lag p99':>10}")
    print(hdr); print("-" * len(hdr))
    for r in runs:
        m = r["metrics"]
        t = m.get("ttft_ms") or {}; o = m.get("tpot_ms") or {}
        lag = m.get("client_dispatch_lag_ms") or {}
        print(f"{r['workload']['name']:<15}{m['goodput_rps']:>9.2f}"
              f"{m['throughput_rps']:>10.2f}{(t.get('p99') or 0):>11.0f}"
              f"{(o.get('p99') or 0):>10.1f}{m['n_ok']:>6}{m['n_failed']:>6}"
              f"{(lag.get('p99') or 0):>10.0f}")


@app.local_entrypoint()
def ls():
    rows = _ls.remote()
    if not rows:
        print("no runs yet"); return
    print(f"{'file':<44}{'note':<12}{'status':<9}{'overlay':<10}{'wl':>3}{'KB':>7}")
    print("-" * 85)
    for r in rows:
        print(f"{r['file']:<44}{str(r.get('note'))[:11]:<12}"
              f"{str(r.get('status')):<9}{str(r.get('overlay')):<10}"
              f"{r.get('n_workloads', 0):>3}{r.get('size_kb', 0):>7}")


@app.local_entrypoint()
def show(name: str):
    r = _get.remote(name)
    print(json.dumps({k: r.get(k) for k in
                      ("note", "status", "model_load_s", "warmup_s", "total_wall_s",
                       "serving_digest", "failure")}, indent=2, default=str))
    print("\nprovenance:", json.dumps(r.get("provenance"), indent=2, default=str))
    if r.get("overlay", {}).get("n_overlays"):
        print("overlays:", r["overlay"]["applied"])
    if r.get("canaries"):
        print("\ncanary digests:", json.dumps(r["canaries"]["digests"], indent=2))
    if r.get("runs"):
        print()
        _table(r["runs"])


@app.local_entrypoint()
def compare(a: str, b: str):
    """Compare two runs workload-by-workload, refusing mismatched traces."""
    ra, rb = _get.remote(a), _get.remote(b)
    ia = {x["workload"]["name"]: x for x in ra.get("runs", [])}
    ib = {x["workload"]["name"]: x for x in rb.get("runs", [])}

    print(f"A = {a}  (overlay {ra.get('overlay', {}).get('digest')})")
    print(f"B = {b}  (overlay {rb.get('overlay', {}).get('digest')})\n")
    hdr = f"{'workload':<15}{'goodput A':>11}{'goodput B':>11}{'delta %':>10}  trace"
    print(hdr); print("-" * len(hdr))
    for name in sorted(set(ia) | set(ib)):
        x, y = ia.get(name), ib.get(name)
        if not x or not y:
            print(f"{name:<15}{'--':>11}{'--':>11}{'--':>10}  missing in one run")
            continue
        ga, gb = x["metrics"]["goodput_rps"], y["metrics"]["goodput_rps"]
        same = x["trace_digest"] == y["trace_digest"]
        d = ((gb - ga) / ga * 100) if ga else float("nan")
        print(f"{name:<15}{ga:>11.2f}{gb:>11.2f}{d:>+10.1f}  "
              f"{'same' if same else 'DIFFERENT -- not comparable'}")

    if ra.get("canaries") and rb.get("canaries"):
        from autoinf.canary import compare as ccmp
        c = ccmp(ra["canaries"]["outputs"], rb["canaries"]["outputs"])
        print(f"\ncanaries: {c['n_identical']}/{c['n']} identical "
              f"(rate {c['exact_match_rate']})")
        for k, v in c["per_canary"].items():
            if v["status"] == "diverged":
                print(f"  {k}: diverged at char {v['first_divergence']} "
                      f"({v['frac_identical']:.0%} identical prefix)")


@app.local_entrypoint()
def pull():
    LOCAL.mkdir(exist_ok=True)
    n = 0
    for r in _all.remote():
        stamp = r.get("result_path", "").split("/")[-1] or f"run{n}.json"
        (LOCAL / stamp).write_text(json.dumps(r, indent=2, default=str))
        n += 1
    print(f"pulled {n} run(s) -> {LOCAL}")


@app.local_entrypoint()
def batches(name: str):
    """What decode batch did the scheduler actually run?

    The output coefficient came out far worse than the memory-bandwidth
    roofline at the nominal concurrency. Roofline and measurement only agree
    at a much smaller batch, so the question is whether the scheduler held
    decode batches together or the load never reached the requested level.
    """
    r = _get.remote(name)
    print(f"{'phase':<12}{'users':>7}{'batch mean':>12}{'p50':>7}{'max':>7}"
          f"{'idle':>7}{'queued':>8}")
    print("-" * 60)
    for lv in r.get("levels", []):
        b = (lv.get("batch") or {}).get("running") or {}
        q = (lv.get("batch") or {}).get("queued") or {}
        print(f"{'level':<12}{lv['n_users']:>7}{b.get('mean', 0):>12.1f}"
              f"{b.get('p50', 0):>7.0f}{b.get('max', 0):>7.0f}"
              f"{b.get('frac_idle', 0):>7.2f}{q.get('mean', 0):>8.1f}")
    for mx in r.get("mixes", []):
        b = (mx.get("batch") or {}).get("running") or {}
        q = (mx.get("batch") or {}).get("queued") or {}
        g = max(mx.get("gpu_seconds", 1), 1e-9)
        print(f"{mx['mix']:<12}{mx['n_users']:>7}{b.get('mean', 0):>12.1f}"
              f"{b.get('p50', 0):>7.0f}{b.get('max', 0):>7.0f}"
              f"{b.get('frac_idle', 0):>7.2f}{q.get('mean', 0):>8.1f}"
              f"   out/s {mx['output_tokens']/g:>7.1f}")


@app.local_entrypoint()
def mixes(name: str):
    """Raw phase-B rows, so attribution can be recomputed locally.

    `report()` skips the attribution block when no level met the SLOs, but the
    mixes still ran and their counters are in the record. The SLO frontier and
    the cost attribution are separate questions measured at separate operating
    points; a phase-A miss does not invalidate phase B.
    """
    r = _get.remote(name)
    for m in r.get("mixes", []):
        b = (m.get("batch") or {}).get("running") or {}
        print(json.dumps({k: m.get(k) for k in
                          ("mix", "n_users", "gpu_seconds", "uncached_tokens",
                           "cached_tokens", "output_tokens", "cache_hit_rate")}
                         | {"batch_mean": b.get("mean")}))


@app.local_entrypoint()
def forward_time(name: str):
    """Compare NNLS attribution against SGLang's own forward-pass GPU time.

    Two independent measurements of the same quantity, sharing no machinery:

      * NNLS regresses total GPU-seconds against token counts over four
        workload mixes with deliberately different ratios. Denominator is wall
        clock x GPU count.
      * `sglang:forward_execution_seconds_total` is CUDA-event time around each
        forward pass, labelled by phase. It needs no mixes and no regression.

    Agreement validates both far more strongly than either alone, because a
    shared error is implausible. Disagreement localises which assumption in the
    regression is wrong -- most likely that wall clock equals work time even at
    saturation.

    Requires SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 (see modal_app._server_env).
    """
    r = _get.remote(name)
    mixes = r.get("mixes", [])
    if not mixes:
        print("no phase-B mixes in this run")
        return

    print(f"{'mix':<10}{'GPU-sec (wall)':>16}{'forward GPU-sec':>18}{'busy frac':>11}")
    print("-" * 56)
    any_fwd = False
    for m in mixes:
        ctr = m.get("server_counters") or {}
        fwd = sum(v for k, v in ctr.items()
                  if "forward_execution_seconds" in k and "dp_cooperation" not in k)
        any_fwd = any_fwd or fwd > 0
        g = m.get("gpu_seconds", 0.0)
        print(f"{m['mix']:<10}{g:>16.1f}{fwd:>18.1f}"
              f"{(fwd / g if g else 0):>11.3f}")

    if not any_fwd:
        print("\n  forward_execution_seconds_total is still zero.")
        print("  Either the env var did not reach the server process, or the")
        print("  counter is per-scheduler and needs --enable-metrics-for-all-schedulers.")
        return
    print("\n  busy frac = forward GPU-time / wall GPU-time. At saturation the")
    print("  regression assumes this is ~1.0; anything well below it means wall")
    print("  clock overstates work and every per-token cost is inflated by 1/frac.")
