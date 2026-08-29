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
    timeout=60 * 60,
    # secrets=[modal.Secret.from_name("huggingface")],  # only for gated repos
)
def bench(serving: dict, workload: dict, slo: dict, note: str = "") -> dict:
    """Run one (config, trace) pair. Returns a complete, self-describing record."""
    from autoinf.bench import run_trace, wait_until_ready
    from autoinf.config import SLO, ServingConfig, WorkloadConfig
    from autoinf.metrics import summarize
    from autoinf.workload import build_trace

    from autoinf import overlay

    sc = ServingConfig(**serving)
    wc = WorkloadConfig(**workload)
    sl = SLO(**slo)

    # Apply source overlays before the server imports anything. Raises if an
    # overlay has gone stale against the installed SGLang, rather than quietly
    # producing a result attributed to the wrong code.
    ov = overlay.apply("/overlays")
    if ov["n_overlays"]:
        print(f"applied {ov['n_overlays']} overlay(s): {ov['applied']}", flush=True)

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--host", "127.0.0.1", "--port", str(SERVER_PORT),
        *sc.to_sglang_args(),
    ]
    print("launching:", " ".join(cmd), flush=True)

    log_path = "/tmp/sglang.log"
    t_launch = time.perf_counter()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

        record: dict = {
            "note": note,
            "serving": asdict(sc), "serving_digest": sc.digest(),
            "workload": asdict(wc), "workload_digest": wc.digest(),
            "slo": asdict(sl),
            "provenance": _provenance(),
            # Which serving *code* produced this result, not just which config.
            "overlay": ov,
        }

        try:
            load_s = asyncio.run(wait_until_ready(SERVER_URL, timeout_s=1800))
            record["model_load_s"] = load_s
            print(f"server ready in {load_s:.1f}s", flush=True)

            trace = build_trace(wc)
            record["trace_digest"] = trace.digest()

            t_bench = time.perf_counter()
            results = asyncio.run(run_trace(trace, SERVER_URL, sc.model))
            record["bench_wall_s"] = time.perf_counter() - t_bench

            record["metrics"] = summarize(results, sl)
            record["per_request"] = [asdict(r) for r in results]
            record["status"] = "ok"

        except Exception as e:
            record["status"] = "failed"
            record["failure"] = f"{type(e).__name__}: {e}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    record["total_wall_s"] = time.perf_counter() - t_launch
    # Keep the tail of the server log; it is where OOMs and bad flags surface.
    try:
        with open(log_path, "r", errors="replace") as f:
            record["server_log_tail"] = f.read()[-8000:]
    except Exception:
        pass

    # Persist inside the container too, so a crashed caller does not lose the run.
    stamp = f"{int(time.time())}-{sc.digest()}-{wc.digest()}-{ov['digest']}"
    os.makedirs("/results/runs", exist_ok=True)
    with open(f"/results/runs/{stamp}.json", "w") as f:
        json.dump(record, f, indent=2, default=str)
    results_vol.commit()

    return record


@app.function(image=image, gpu="H100", volumes={"/cache": hf_cache}, timeout=60 * 60)
def prefetch(model: str) -> str:
    """Download weights into the Volume once, so benchmark runs start warm."""
    from huggingface_hub import snapshot_download
    p = snapshot_download(model)
    hf_cache.commit()
    return p


@app.local_entrypoint()
def main(smoke: bool = True):
    """Smallest useful run: dev model, 1xH100, short trace."""
    from autoinf.config import SLO, ServingConfig, WorkloadConfig

    sc = ServingConfig()
    wc = WorkloadConfig(n_requests=60, request_rate=2.0) if smoke else WorkloadConfig()
    sl = SLO()

    rec = bench.remote(asdict(sc), asdict(wc), asdict(sl), note="smoke")
    m = rec.get("metrics", {})
    print(json.dumps({
        "status": rec["status"],
        "model_load_s": rec.get("model_load_s"),
        "goodput_rps": m.get("goodput_rps"),
        "throughput_rps": m.get("throughput_rps"),
        "ttft_p99": (m.get("ttft_ms") or {}).get("p99"),
        "tpot_p99": (m.get("tpot_ms") or {}).get("p99"),
        "failed": m.get("n_failed"),
    }, indent=2, default=str))
