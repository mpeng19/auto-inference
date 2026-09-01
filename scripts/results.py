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


@app.local_entrypoint()
def phases(name: str):
    """Per-phase forward GPU time, and prices computed straight from it.

    With `busy_frac ~ 1.0` established, the interesting content is the phase
    label: if forward time splits prefill from decode, effective input price
    is `prefill_gpu_s / input_tokens` and no decomposition is needed.
    """
    r = _get.remote(name)
    for m in r.get("mixes", []):
        ctr = m.get("server_counters") or {}
        fwd = {k: v for k, v in ctr.items() if "forward_execution_seconds" in k}
        print(f"\n{m['mix']}  (wall GPU-s {m.get('gpu_seconds')})")
        if not fwd:
            print("   no forward_execution counters")
            continue
        for k, v in sorted(fwd.items()):
            print(f"   {k} = {v:.2f}")


@app.local_entrypoint()
def direct_price(name: str):
    """Price a run from sweep-A levels alone, with no regression.

    The end-to-end test of the direct path: real traffic, phase-split GPU time,
    priced at the hit rate the system actually achieved.
    """
    from autoinf.pricing import price_direct

    r = _get.remote(name)
    n_gpu = (r.get("serving") or {}).get("n_gpu", 1)
    print(f"{'users':>6}{'extend s':>10}{'decode s':>10}{'in tok':>12}"
          f"{'out tok':>10}{'hit':>7}{'eff-in $/M':>12}{'out $/M':>10}")
    print("-" * 78)
    for lv in r.get("levels", []):
        ctr = lv.get("server_counters") or {}
        ext = ctr.get("sglang:forward_execution_seconds_total[extend]")
        dec = ctr.get("sglang:forward_execution_seconds_total[decode]")
        if ext is None or dec is None:
            print(f"{lv['n_users']:>6}   no phase-split counters on this level")
            continue
        # NOT x n_gpu: `forward_execution_seconds_total` is already summed
        # across TP ranks, so it IS GPU-seconds. Verified by ::sanity --
        # fwd/(wall x n_gpu) = 1.00, where per-rank timing would give 0.50.
        # Multiplying here doubled every output price.
        p = price_direct(gpu_seconds_input=ext,
                         gpu_seconds_output=dec,
                         input_tokens=lv["prompt_tokens"],
                         output_tokens=lv["output_tokens"],
                         cached_tokens=lv["cached_tokens"])
        print(f"{lv['n_users']:>6}{ext:>10.1f}{dec:>10.1f}"
              f"{lv['prompt_tokens']:>12,.0f}{lv['output_tokens']:>10,.0f}"
              f"{p.hit_rate:>7.3f}{p.effective_in_per_m:>12.4f}{p.out_per_m:>10.3f}")
    print(f"\n  basis: $3.00/GPU-hr, 50% utilisation, break-even (no margin)")
    print("  eff-in = extend GPU-s / ALL input tokens -- cached tokens cost no")
    print("  prefill, so a higher hit rate lowers this automatically. That is the")
    print("  whole point: caching well shows up as a cheaper price, not as a")
    print("  number we normalise away.")


@app.local_entrypoint()
def sanity(name: str):
    """Is forward GPU-time physically possible against wall clock?

    Forward time can never exceed wall_seconds x n_gpu. If it does, the counter
    is being aggregated differently than assumed -- e.g. already summed across
    TP ranks, in which case multiplying by n_gpu double-counts.
    """
    r = _get.remote(name)
    n_gpu = (r.get("serving") or {}).get("n_gpu", 1)
    print(f"n_gpu = {n_gpu}\n")
    print(f"{'level':>7}{'wall s':>9}{'wall x n_gpu':>14}{'extend':>9}{'decode':>9}"
          f"{'fwd total':>11}{'fwd/wall':>10}{'fwd/(wall*n)':>14}")
    print("-" * 84)
    for lv in r.get("levels", []):
        c = lv.get("server_counters") or {}
        e = c.get("sglang:forward_execution_seconds_total[extend]", 0.0)
        d = c.get("sglang:forward_execution_seconds_total[decode]", 0.0)
        w = lv.get("wall_s", 0.0)
        print(f"{lv['n_users']:>7}{w:>9.1f}{w*n_gpu:>14.1f}{e:>9.1f}{d:>9.1f}"
              f"{e+d:>11.1f}{(e+d)/max(w,1e-9):>10.2f}{(e+d)/max(w*n_gpu,1e-9):>14.2f}")
    print("\n  fwd/(wall x n_gpu) must be <= 1.0. If it sits near 1/n_gpu instead,")
    print("  the counter is ONE rank's time and must be multiplied by n_gpu.")
    print("  If it sits near 1.0, it is already aggregated -- do not multiply.")


@app.local_entrypoint()
def tiers(name: str):
    """Which SLO tier failed at each level, and by how much."""
    r = _get.remote(name)
    for lv in r.get("levels", []):
        print(f"\nN = {lv['n_users']} users   good_frac {lv.get('good_frac', 0):.3f}"
              f"   meets_slo {lv.get('meets_slo')}")
        for t in lv.get("slo_tiers", []):
            q = t["percentile"]
            for kind, lim in (("ttft", t["ttft_limit"]), ("tpot", t["tpot_limit"])):
                v = t[kind]
                if v is None:
                    continue
                ok = v <= lim
                print(f"   p{q} {kind.upper():<5} {v:>8.1f} ms  vs {lim:>7.1f} ms"
                      f"   {'ok' if ok else 'FAIL by %.0f%%' % ((v/lim-1)*100)}")


def _spec(s: str) -> tuple[str, float]:
    """'p90:2818' -> ('p90', 2818.0). Blank or 'none' disables the bound."""
    if not s or s.lower() in ("none", "off", "-"):
        return ("p50", float("inf"))
    q, _, lim = s.partition(":")
    q = q if q.startswith("p") else "p" + q
    return (q, float(lim))


@app.local_entrypoint()
def rescore(name: str, ttft: str = "p90:2818", tpot: str = "p50:20"):
    """Re-judge a finished sweep at a different SLO, and price it there.

    Every level stores its full percentile set (p50/p90/p95/p99) precisely so
    that changing which order statistic the frontier is judged at costs
    nothing. A sweep is 25 GPU-minutes; re-reading it is free.

    It also lifts a restriction `SLO.tiers()` bakes in: there, TTFT and TPOT
    must share a percentile. The market gives us a p90 for TTFT (published
    latency percentiles) and only a p50 for TPOT (throughput percentiles run
    the wrong way -- p99 throughput is the FASTEST 1%), so the two bounds
    genuinely live at different quantiles.

        uv run modal run scripts/results.py::rescore --name X \
            --ttft p90:2818 --tpot p50:20
    """
    from autoinf.pricing import price_direct

    tq, tlim = _spec(ttft)
    pq, plim = _spec(tpot)
    r = _get.remote(name)
    n_gpu = (r.get("serving") or {}).get("n_gpu", 1)
    print(f"judging: {tq} TTFT <= {tlim:.0f} ms   and   {pq} TPOT <= {plim:.1f} ms"
          f"   ({n_gpu}x{(r.get('serving') or {}).get('gpu', '?')})\n")

    print(f"{'users':>6}{'n req':>7}{'goodput':>9}{'batch':>7}"
          f"{tq + ' TTFT':>11}{pq + ' TPOT':>11}{'hit':>7}{'':>8}")
    print("-" * 66)
    passing: list[dict] = []
    for lv in r.get("levels", []):
        tt = (lv.get("ttft_ms") or {}).get(tq)
        tp = (lv.get("tpot_ms") or {}).get(pq)
        nreq = (lv.get("ttft_ms") or {}).get("n", 0)
        b = (lv.get("batch") or {}).get("running") or {}
        ok = (tt is not None and tp is not None and tt <= tlim and tp <= plim
              and lv.get("n_failed", 0) == 0)
        if ok:
            passing.append(lv)
        print(f"{lv['n_users']:>6}{nreq:>7}{lv['goodput_rps']:>9.2f}"
              f"{(b.get('mean') or 0):>7.1f}{(tt or 0):>11.0f}{(tp or 0):>11.1f}"
              f"{(lv.get('cache_hit_rate') or 0):>7.2f}"
              f"{'  OK' if ok else '  MISS':>8}")

    if not passing:
        print("\nno level met this SLO")
        return
    star = max(passing, key=lambda l: l["n_users"])
    sb = (star.get("batch") or {}).get("running") or {}
    print(f"\nN* = {star['n_users']} users   goodput {star['goodput_rps']:.2f} rps"
          f"   cache hit {star['cache_hit_rate']:.3f}"
          f"   batch {sb.get('mean', 0):.1f}")

    ctr = star.get("server_counters") or {}
    ext = ctr.get("sglang:forward_execution_seconds_total[extend]")
    dec = ctr.get("sglang:forward_execution_seconds_total[decode]")
    if ext is None or dec is None:
        print("  no phase-split counters on this level; cannot price it")
        return
    p = price_direct(gpu_seconds_input=ext, gpu_seconds_output=dec,
                     input_tokens=star["prompt_tokens"],
                     output_tokens=star["output_tokens"],
                     cached_tokens=star["cached_tokens"])
    print(f"\n  extend {ext:.1f} GPU-s over {star['prompt_tokens']:,.0f} input tokens")
    print(f"  decode {dec:.1f} GPU-s over {star['output_tokens']:,.0f} output tokens")
    print(f"\n  effective input  ${p.effective_in_per_m:.4f}/M")
    print(f"  output           ${p.out_per_m:.4f}/M")
    print(f"  basis: ${p.usd_per_gpu_hour:.2f}/GPU-hr, "
          f"{p.utilization:.0%} utilisation, break-even")

    # Rank against each provider's REALISED effective price -- the one they
    # actually achieve at their own hit rate. Re-blending everyone to a common
    # hit rate would erase the thing being optimised (SS5e): caching well is
    # part of serving well, and Venice and Cloudflare realising 0.0% on the
    # same model and the same traffic is a serving-stack difference, not a
    # workload difference.
    from autoinf.modal_app import MARKET_AS_OF, MARKET_REALISED
    board = sorted([(n, e, h) for n, e, h, _ in MARKET_REALISED] +
                   [("** us **", p.effective_in_per_m, p.hit_rate)],
                   key=lambda r: r[1])
    ours = 1 + sum(1 for _, e, _ in board if e < p.effective_in_per_m)
    print(f"\n  realised effective input price, OpenRouter {MARKET_AS_OF}")
    print(f"  {'#':>3}  {'provider':<12}{'eff-in $/M':>12}{'hit':>8}")
    for i, (n, e, h) in enumerate(board, 1):
        print(f"  {i:>3}  {n:<12}{e:>12.4f}{h:>8.3f}"
              + ("   <--" if n == "** us **" else ""))
    print(f"\n  rank {ours} of {len(board)}")
