"""Modal app: launch SGLang on a GPU and benchmark it in the same container.

Why server and client share a container: the load generator talks to the
server over loopback, so no network sits between them. Running the client on
a laptop would put 20-80ms of WAN latency in front of every TTFT measurement,
which is larger than most of the effects we are trying to detect.

The cost is CPU contention between client and server, which is why `cpu=` is
set generously and why `client_dispatch_lag_ms` is reported on every run. If
that lag grows, raise the CPU allocation or move the client to a second
container in the same region.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import asdict

import modal

APP_NAME = "auto-inference"

# ── versions: pin everything, change deliberately ────────────────
SGLANG_VERSION = "0.5.18"
CUDA_TAG = "12.8.1-devel-ubuntu22.04"
PY_VERSION = "3.12"

SERVER_PORT = 30000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

# HF cache lives on a Volume so weights survive container death. At
# $0.09/GiB/month, the 30B FP8 dev model (~31GB) costs ~$2.80/mo to park;
# the 235B target (~235GB) would be ~$21/mo. Delete what you are not using.
hf_cache = modal.Volume.from_name("auto-inference-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("auto-inference-results", create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .entrypoint([])
    .apt_install("git", "libnuma-dev")
    .pip_install(
        f"sglang[all]=={SGLANG_VERSION}",
        "aiohttp>=3.9",
        "hf-transfer>=0.1.6",
        "huggingface-hub>=0.26",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        # hf_transfer is retired; Xet is the current fast-download path.
        # (probe_env flagged HF_HUB_ENABLE_HF_TRANSFER as deprecated.)
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
    # Ship our harness source into the image.
    .add_local_python_source("autoinf")
    # Source overlays: files here replace their counterparts inside the
    # installed sglang package at container start. Mounted (copy=False), so
    # editing a scheduler costs a container start, not an image rebuild.
    .add_local_dir("overlays", "/overlays", copy=False)
)

app = modal.App(APP_NAME)


def _provenance() -> dict:
    """Everything that must be recorded for a result to be interpretable later."""
    out = {"sglang_version": SGLANG_VERSION, "cuda_tag": CUDA_TAG}
    try:
        out["gpus"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True,
        ).strip().splitlines()
    except Exception as e:
        out["gpus"] = f"unavailable: {e}"
    try:
        import torch
        out["torch"] = str(torch.__version__)
        out["n_visible_gpus"] = torch.cuda.device_count()
    except Exception:
        pass
    return out


@app.function(
    image=image,
    gpu="H100",                       # override per-call: .with_options(gpu="H100:8")
    cpu=8.0,                          # headroom so the client is not the bottleneck
    volumes={"/cache": hf_cache, "/results": results_vol},
    timeout=90 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def bench(serving: dict, workloads: list[dict], slo: dict, note: str = "",
          warmup_n: int = 20, canaries: bool = True) -> dict:
    """Launch one server, run many workloads against it.

    The server is started once and reused across every workload in the list.
    Model load is ~350s cold and dominates a short trace, so launching per
    workload would spend most of the budget on loading rather than measuring.
    Anything that must vary per server launch (a ServingConfig field, an
    overlay) needs a separate call; anything that is just traffic shape does not.
    """
    from autoinf import overlay
    from autoinf.bench import complete, run_trace, wait_until_ready, warmup
    from autoinf.canary import CANARIES, digest as cdigest
    from autoinf.config import SLO, ServingConfig, WorkloadConfig
    from autoinf.metrics import summarize
    from autoinf.workload import build_trace

    # The ramp workload issues ~3000 requests and the client holds a socket
    # per in-flight request. A default 1024-fd limit would surface as
    # connection errors attributed to the *server*, which would be wrong.
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
        print(f"RLIMIT_NOFILE {soft} -> {resource.getrlimit(resource.RLIMIT_NOFILE)[0]}",
              flush=True)
    except Exception as e:
        print(f"could not raise fd limit: {e}", flush=True)

    sc = ServingConfig(**serving)
    sl = SLO(**slo)
    wcs = [WorkloadConfig(**w) for w in workloads]

    # Apply source overlays before the server imports anything. Raises if an
    # overlay has gone stale against the installed SGLang, rather than quietly
    # producing a result attributed to the wrong code.
    ov = overlay.apply("/overlays")
    if ov["n_overlays"]:
        print(f"applied {ov['n_overlays']} overlay(s): {ov['applied']}", flush=True)

    cmd = ["python", "-m", "sglang.launch_server",
           "--host", "127.0.0.1", "--port", str(SERVER_PORT), *sc.to_sglang_args()]
    print("launching:", " ".join(cmd), flush=True)

    record: dict = {
        "note": note,
        "serving": asdict(sc), "serving_digest": sc.digest(),
        "slo": asdict(sl),
        "provenance": _provenance(),
        "overlay": ov,
        "runs": [],
    }

    log_path = "/tmp/sglang.log"
    t_launch = time.perf_counter()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            record["model_load_s"] = round(asyncio.run(wait_until_ready(
                SERVER_URL, timeout_s=2400, proc=proc, log_path=log_path)), 1)
            print(f"server ready in {record['model_load_s']}s", flush=True)
            # Persist any weights just downloaded, so the next run starts warm.
            # Without this the 30B FP8 model is re-fetched (~350s) every run.
            try:
                hf_cache.commit()
            except Exception as e:
                print(f"  (hf_cache commit failed: {e})", flush=True)

            record["warmup_s"] = round(
                asyncio.run(warmup(SERVER_URL, sc.model, warmup_n)), 1)

            # Canaries first, on an otherwise idle server, so their outputs are
            # not perturbed by whatever batch the workload happens to form.
            if canaries:
                outs = {}
                for name, prompt, mt in CANARIES:
                    outs[name] = asyncio.run(complete(SERVER_URL, sc.model, prompt, mt))
                record["canaries"] = {
                    "outputs": outs,
                    "digests": {k: cdigest(v) for k, v in outs.items()},
                }

            for wc in wcs:
                trace = build_trace(wc)
                print(f"-- workload {wc.name}: {len(trace.requests)} reqs, "
                      f"{trace.duration_s:.0f}s", flush=True)
                t0 = time.perf_counter()
                results = asyncio.run(run_trace(trace, SERVER_URL, sc.model))
                m = summarize(results, sl)
                record["runs"].append({
                    "workload": asdict(wc),
                    "workload_digest": wc.digest(),
                    "trace_digest": trace.digest(),
                    "trace_describe": trace.describe(),
                    "bench_wall_s": round(time.perf_counter() - t0, 1),
                    "metrics": m,
                    "per_request": [asdict(r) for r in results],
                })
                print(f"   goodput {m['goodput_rps']:.2f} rps | "
                      f"p99 TTFT {(m['ttft_ms'] or {}).get('p99')} | "
                      f"failed {m['n_failed']} | "
                      f"client lag p99 {(m['client_dispatch_lag_ms'] or {}).get('p99')}",
                      flush=True)

            record["status"] = "ok"

        except Exception as e:
            record["status"] = "failed"
            record["failure"] = f"{type(e).__name__}: {e}"
            print("FAILED:", record["failure"], flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    record["total_wall_s"] = round(time.perf_counter() - t_launch, 1)
    try:
        with open(log_path, "r", errors="replace") as f:
            record["server_log_tail"] = f.read()[-8000:]
    except Exception:
        pass

    stamp = f"{int(time.time())}-{sc.digest()}-{ov['digest']}"
    os.makedirs("/results/runs", exist_ok=True)
    with open(f"/results/runs/{stamp}.json", "w") as f:
        json.dump(record, f, indent=2, default=str)
    results_vol.commit()
    record["result_path"] = f"/results/runs/{stamp}.json"

    return record


@app.function(image=image, gpu="H100", volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=60 * 60)
def prefetch(model: str) -> str:
    """Download weights into the Volume once, so benchmark runs start warm."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(model)
    hf_cache.commit()
    return p


def bench_for(sc, region: str | None = None):
    """Pick the right resource shape for a ServingConfig.

    `bench` is declared with a 1xH100 default; anything larger is an override
    at call time rather than a second copy of the function. CPU scales with GPU
    count because the client shares the container and must not become the
    bottleneck.

    `region` pins the datacenter. Leaving it unset lets Modal place runs
    wherever there is capacity, which adds a variance source across repeats;
    pin it once a baseline exists so later comparisons are like-for-like.
    """
    n = max(1, sc.n_gpu)
    opts = {
        "gpu": f"{sc.gpu}:{n}" if n > 1 else sc.gpu,
        "cpu": float(max(8, 2 * n)),
        "timeout": 60 * 60 * (2 if n > 1 else 1) + 1800,
    }
    if region:
        opts["region"] = region
    return bench.with_options(**opts)


def _summary_row(name: str, m: dict) -> str:
    t = m.get("ttft_ms") or {}
    o = m.get("tpot_ms") or {}
    lag = m.get("client_dispatch_lag_ms") or {}
    return (f"{name:<15}{m['goodput_rps']:>9.2f}{m['throughput_rps']:>10.2f}"
            f"{(t.get('p99') or 0):>11.0f}{(o.get('p99') or 0):>10.1f}"
            f"{m['n_failed']:>8}{(lag.get('p99') or 0):>11.0f}")


def _print_table(runs: list[dict]) -> None:
    hdr = (f"{'workload':<15}{'goodput':>9}{'thruput':>10}{'p99 TTFT':>11}"
           f"{'p99 TPOT':>10}{'failed':>8}{'lag p99':>11}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in runs:
        print(_summary_row(r["workload"]["name"], r["metrics"]))
    print("\nlag p99 is the CLIENT's dispatch lag. If it is large the load "
          "generator was the bottleneck and the server numbers are void.")


@app.local_entrypoint()
def smoke():
    """Smallest end-to-end run: one short workload on one H100."""
    from autoinf.config import SLO, ServingConfig, WorkloadConfig
    wc = WorkloadConfig(name="smoke", n_requests=60, request_rate=2.0)
    rec = bench.remote(asdict(ServingConfig()), [asdict(wc)], asdict(SLO()),
                       note="smoke")
    print(json.dumps({k: rec.get(k) for k in
                      ("status", "model_load_s", "warmup_s", "total_wall_s",
                       "result_path", "failure")}, indent=2, default=str))
    if rec.get("runs"):
        _print_table(rec["runs"])
    if rec.get("canaries"):
        print("\ncanary digests:", json.dumps(rec["canaries"]["digests"], indent=2))


@app.local_entrypoint()
def suite(scale: float = 1.0, seed: int = 0):
    """Full eval suite against one server launch -- the 1-GPU test case."""
    from autoinf.config import SLO, ServingConfig
    from autoinf.workload import suite as wsuite
    wl = [asdict(c) for c in wsuite(seed=seed, scale=scale).values()]
    rec = bench.remote(asdict(ServingConfig()), wl, asdict(SLO()), note="suite")
    print(json.dumps({k: rec.get(k) for k in
                      ("status", "model_load_s", "total_wall_s", "result_path",
                       "failure")}, indent=2, default=str))
    if rec.get("runs"):
        _print_table(rec["runs"])


@app.local_entrypoint()
def noise(repeats: int = 5, workload: str = "sustained"):
    """Noise floor: the SAME config and trace, N separate server launches.

    Restarting the server each time is deliberate -- it captures the variance
    an experiment actually faces (fresh container, fresh allocator, possibly
    different physical host), not just steady-state jitter within one process.
    Whatever spread this shows is the smallest effect any result can claim.
    """
    import statistics
    from autoinf.config import SLO, ServingConfig
    from autoinf.workload import suite as wsuite

    wc = asdict(wsuite(seed=0)[workload])
    sc = asdict(ServingConfig())
    good, ttft, canaries = [], [], []

    for i in range(repeats):
        print(f"\n=== repeat {i + 1}/{repeats} ===", flush=True)
        rec = bench.remote(sc, [wc], asdict(SLO()), note=f"noise-{i}")
        if rec.get("status") != "ok" or not rec.get("runs"):
            print("  FAILED:", rec.get("failure")); continue
        m = rec["runs"][0]["metrics"]
        good.append(m["goodput_rps"])
        ttft.append((m["ttft_ms"] or {}).get("p99") or 0.0)
        if rec.get("canaries"):
            canaries.append(rec["canaries"]["outputs"])
        print(f"  goodput {good[-1]:.3f} rps | p99 TTFT {ttft[-1]:.0f} ms")

    def cv(xs):
        return statistics.pstdev(xs) / statistics.fmean(xs) if len(xs) > 1 and \
            statistics.fmean(xs) else None

    print(f"\n=== noise floor over {len(good)} runs ({workload}) ===")
    for label, xs in (("goodput_rps", good), ("p99_ttft_ms", ttft)):
        if xs:
            c = cv(xs)
            print(f"  {label:<14} median {statistics.median(xs):>9.3f}  "
                  f"min {min(xs):>9.3f}  max {max(xs):>9.3f}  "
                  f"CV {c:.4f}" if c is not None else "")
    if len(canaries) >= 2:
        from autoinf.canary import compare
        c = compare(canaries[0], canaries[1])
        print(f"\ncanary floor (same config, two runs): "
              f"exact match {c['n_identical']}/{c['n']} = {c['exact_match_rate']}")
        print("  This is the divergence baseline. Any config that diverges MORE "
              "than this is suspect; equal divergence is ordinary batching "
              "non-determinism, not a bug.")


@app.local_entrypoint()
def suite_8x(model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
             scale: float = 1.0, seed: int = 0, region: str = ""):
    """Phase 2: the full suite on 8xH100 against the 235B target model.

    ~$31.60/hr. The 235B FP8 weights are ~235GB, leaving ~400GB of the 640GB
    for KV. Run `prefetch` first -- downloading 235GB inside a benchmark run
    wastes 8 GPUs' time on network I/O.

    Note the Starter plan caps GPU concurrency at 10, so this consumes 8 of 10
    and no second run can proceed alongside it.
    """
    from autoinf.config import SLO, ServingConfig
    from autoinf.workload import suite as wsuite

    sc = ServingConfig(model=model, gpu="H100", n_gpu=8, tp_size=8, ep_size=8)
    wl = [asdict(c) for c in wsuite(seed=seed, scale=scale).values()]
    rec = bench_for(sc, region or None).remote(
        asdict(sc), wl, asdict(SLO()), note="suite-8x")
    print(json.dumps({k: rec.get(k) for k in
                      ("status", "model_load_s", "total_wall_s", "result_path",
                       "failure")}, indent=2, default=str))
    if rec.get("runs"):
        _print_table(rec["runs"])


@app.function(image=image, gpu="H100:8", cpu=16.0,
              volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3 * 60 * 60)
def prefetch_big(model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8") -> str:
    """Pull the 235B weights (~235GB) into the Volume. Storage is $0.09/GiB/mo,
    so this costs ~$21/month to keep parked -- delete it when not in use."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(model)
    hf_cache.commit()
    return p


@app.local_entrypoint()
def saturate(lo: float = 5.0, hi: float = 160.0, duration: float = 300.0):
    """Find the saturation knee: ramp hard until the SLOs actually break.

    The first suite run met both SLOs on 100% of requests in every workload,
    including a ramp to 32 rps -- i.e. it measured an idle server. Optimisation
    work is meaningless until we know where the system actually bends, because
    a scheduler change cannot show up in a server that is not stressed.

    This ramps well past the expected knee. Success looks like *failure*: a
    region where goodput falls below throughput.
    """
    from autoinf.config import SLO, ServingConfig, WorkloadConfig
    wc = WorkloadConfig(name="saturate", arrival="ramp", request_rate=lo,
                        ramp_end_rate=hi, duration_s=duration, n_requests=None)
    rec = bench.remote(asdict(ServingConfig()), [asdict(wc)], asdict(SLO()),
                       note="saturate", canaries=False)
    print(json.dumps({k: rec.get(k) for k in
                      ("status", "model_load_s", "total_wall_s", "result_path",
                       "failure")}, indent=2, default=str))
    if not rec.get("runs"):
        return
    run = rec["runs"][0]
    _print_table([run])

    # Bucket per-request results by arrival time to locate where it bends.
    # Aggregate numbers hide the knee; the whole point is the shape.
    import collections
    per = run["per_request"]
    dur = run["trace_describe"]["duration_s"]
    nb = 12
    buckets = collections.defaultdict(list)
    for r in per:
        b = min(nb - 1, int(r["scheduled_s"] / dur * nb))
        buckets[b].append(r)

    slo = rec["slo"]
    print(f"\n{'window':<14}{'offered':>9}{'ok':>7}{'met SLO':>9}"
          f"{'p99 TTFT':>10}{'p99 TPOT':>10}")
    print("-" * 59)
    for b in range(nb):
        rs = buckets.get(b, [])
        if not rs:
            continue
        w = dur / nb
        ttfts, tpots, met = [], [], 0
        for r in rs:
            if not r["ok"] or r["first_token_s"] is None:
                continue
            t = (r["first_token_s"] - r["dispatched_s"]) * 1000
            ttfts.append(t)
            p = None
            if r["end_s"] is not None and r["output_tokens"] > 1:
                p = (r["end_s"] - r["first_token_s"]) * 1000 / (r["output_tokens"] - 1)
                tpots.append(p)
            if t <= slo["ttft_ms"] and (p is None or p <= slo["tpot_ms"]):
                met += 1
        f = lambda xs: sorted(xs)[int(len(xs) * 0.99)] if xs else 0
        print(f"{b * w:>5.0f}-{(b + 1) * w:<8.0f}{len(rs) / w:>9.1f}"
              f"{len(ttfts):>7}{met / len(rs) * 100:>8.0f}%"
              f"{f(ttfts):>10.0f}{f(tpots):>10.1f}")
    print("\nThe knee is the first window where 'met SLO' drops below 100%.")
