"""Run one concurrency sweep on a GPU, and nothing else.

The old app had fifteen entrypoints because it was a research harness. This
has one, because the product has one measurement: sweep offered load on real
traffic until the SLOs stop holding, and read phase-split GPU time at the last
level that held.

**Server and client share a container**, talking over loopback. A laptop-side
client would put 20-80 ms of WAN latency in front of every TTFT measurement --
larger than most effects we want to detect. The cost is CPU contention, which
is why `cpu=` is generous and why `client_dispatch_lag_ms` is reported on every
level: if that grows, the client was the bottleneck and the run is void.

**Nothing samples the GPU during a measured window.** Polling nvidia-smi four
times a second from this container starved the load generator and moved N* from
128 to 32 -- the instrument changed what it measured, worst at high load where
the answer lives. The only in-window sampling is one local HTTP GET every two
seconds for the running batch.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import subprocess
import time
from dataclasses import asdict

import modal

# Everything a fresh clone needs to override. Nothing here is account-specific:
# the volumes are created on first use in whatever workspace you are logged
# into, and the HF secret is optional.
APP_NAME = os.environ.get("SIMULATOR_APP_NAME", "auto-inference")
HF_CACHE_VOLUME = os.environ.get("SIMULATOR_HF_VOLUME", "auto-inference-hf-cache")
RESULTS_VOLUME = os.environ.get("SIMULATOR_RESULTS_VOLUME", "auto-inference-results")
# Opt-in by name. Unset means no secret is attached, which is the right
# default: both Qwen3 checkpoints are Apache-2.0 and ungated, so a fresh clone
# needs no token, and requiring one would make a new user's first `make deploy`
# fail on a secret they have no reason to have.
HF_SECRET = os.environ.get("SIMULATOR_HF_SECRET", "")

SGLANG_VERSION = "0.5.18"
CUDA_TAG = "12.8.1-devel-ubuntu22.04"
PY_VERSION = "3.12"

SERVER_PORT = 30000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


def _hf_secret() -> list:
    """Hugging Face auth, attached only when `SIMULATOR_HF_SECRET` names one.

    **Never hydrated here.** This runs at decorator time, i.e. on every
    `import simulator.runner.modal_runner`, and an existence check makes a
    network round trip -- which took 57 seconds under a blocked socket and
    would have made the module un-importable offline. Modal resolves the name
    when the app is deployed, which is the right time to find out it is wrong.

    Needed only for a gated model or if anonymous Hub rate limits bite:

        modal secret create huggingface HF_TOKEN=hf_...
        export SIMULATOR_HF_SECRET=huggingface
    """
    return [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python=PY_VERSION)
    .entrypoint([])
    .apt_install("git", "libnuma-dev")
    .pip_install(
        f"sglang[all]=={SGLANG_VERSION}",
        "aiohttp>=3.9",
        "hf-transfer>=0.1.6",
        "huggingface-hub>=0.26",
        "uvloop>=0.19",
        "pandas>=2.0",           # TraceLab parquet
        "pyarrow>=15",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("simulator")
)

app = modal.App(APP_NAME)


def _server_env() -> dict:
    """Environment for the SGLang server process.

    `SGLANG_ENABLE_METRICS_DEVICE_TIMER` wraps every forward pass in CUDA
    events and emits `forward_execution_seconds_total` labelled by phase. That
    is **actual GPU time**, and the phase split is the entire basis of the
    price. Without it the counter is declared and never incremented, which is
    why an early attempt to read it returned zeros and we spent weeks on a
    regression that existed to work around a flag being off.
    """
    import os
    env = dict(os.environ)
    env["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "1"
    return env


def _provenance() -> dict:
    out = {"sglang_version": SGLANG_VERSION, "cuda_tag": CUDA_TAG}
    try:
        out["gpus"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception as e:
        out["gpus"] = f"unavailable: {e}"
    try:
        import torch
        out["torch"] = str(torch.__version__)
    except Exception:
        pass
    return out


# The load generator is CPU-bound at high concurrency (see the note on
# `client_dispatch_lag_ms` above); 16 vCPUs at retail is ~$2.19/h on top of
# the GPU. `SWEEP_VCPU` exists so the evaluator can bill what was reserved.
SWEEP_VCPU = 16.0


@app.function(
    image=image, gpu="H100", cpu=SWEEP_VCPU,
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=_hf_secret(),
    timeout=4 * 60 * 60,
)
def sweep(serving: dict, slo: dict, stack: dict, levels: list[int],
          seconds_per_level: float = 120.0, repeats: int = 1,
          target_in: int = 0, target_out: int = 0, n_sessions: int = 300,
          canaries: bool = True, note: str = "", allow_stale: bool = False,
          profile_level: int = 0, profile_steps: int = 20,
          quality_suites: tuple = ("gsm8k",), quality_n: int = 50,
          quality_baseline: dict | None = None,
          quality_tolerance_pp: float = 10.0) -> dict:
    """Sweep concurrent conversations; return one record per level.

    Every level records its full percentile set and its raw server counters, so
    the SLO can be re-judged and the run re-priced afterwards without touching
    a GPU. A sweep is 25-60 GPU-minutes; re-reading it is free, and which order
    statistic the frontier is judged at is a choice we have changed three times.
    """
    from simulator.config import ServingConfig
    from simulator.measure import canary as canary_mod
    from simulator.measure import quality as quality_mod
    from simulator.measure import server as srv
    from simulator.measure.loadgen import (
        client_health,
        install_fast_loop,
        run_concurrent_users,
    )
    from simulator.measure.metrics import detect_collapse, summarize
    from simulator.slo import SLO
    from simulator.stack import InferenceStack
    from simulator.workload.tracelab import (
        describe,
        load_sessions,
        scale_to_market,
        to_sessions,
    )

    try:
        import resource
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    except Exception:
        pass

    sc, sl = ServingConfig(**serving), SLO.from_dict(slo)
    st = InferenceStack.from_dict(stack)
    install_fast_loop()

    rec: dict = {"status": "failed", "note": note, "serving": asdict(sc),
                 "serving_digest": sc.digest(), "slo": sl.as_dict(),
                 "stack_digest": st.digest, "levels": [],
                 "seconds_per_level": seconds_per_level, "repeats": repeats,
                 "provenance": _provenance(), "started_at": time.time()}

    problems = sc.validate()
    if problems:
        rec["failure"] = "invalid ServingConfig: " + "; ".join(problems)
        print("INVALID CONFIG:", *problems, sep="\n  ", flush=True)
        return _save(rec, sc)

    try:
        rec["stack"] = st.apply(allow_stale=allow_stale)
        print("stack:", st.describe(), flush=True)
    except Exception as e:
        rec["failure"] = f"{type(e).__name__}: {e}"
        print("STACK REFUSED:", rec["failure"], flush=True)
        return _save(rec, sc)

    cmd = ["python", "-m", "sglang.launch_server", "--host", "127.0.0.1",
           "--port", str(SERVER_PORT), *sc.to_sglang_args()]
    print("launching:", " ".join(cmd), flush=True)
    log_path = "/tmp/sglang.log"

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=_server_env())
        try:
            rec["model_load_s"] = round(asyncio.run(srv.wait_until_ready(
                SERVER_URL, 2400, proc=proc, log_path=log_path, stall_s=420)), 1)
            hf_cache.commit()
            asyncio.run(srv.warmup(SERVER_URL, sc.model, 20))

            # Real coding-agent traffic, rescaled so each request carries the
            # marketplace's average token counts. Input and output are scaled
            # separately: uniform scaling preserves TraceLab's 291:1 ratio,
            # while the real traffic runs 9.9:1 -- and that ratio decides
            # whether output tokens are 12% of the bill or 80% of it.
            raw = load_sessions(min_rounds=4, max_rounds=40,
                                max_sessions=n_sessions, seed=0)
            scaled, rec["market_scaling"] = scale_to_market(
                raw, target_in, target_out)
            pool = to_sessions(scaled, seed=0)
            rec["traffic"] = describe(scaled)
            print(f"traffic: {rec['traffic']['n_sessions']} sessions, "
                  f"input p50 {rec['traffic']['input_tokens_p50']} tok, "
                  f"intrinsic reuse {rec['traffic']['aggregate_hit_rate']}", flush=True)

            def make_session(uid, n):
                return pool[(uid * 7919 + n * 104729) % len(pool)]

            if canaries:
                rec["canary"] = asyncio.run(
                    canary_mod.run(SERVER_URL, sc.model))
                print(f"canary: {rec['canary'].get('summary', '')}", flush=True)

            # Quality before load, on an idle server: this measures the model,
            # not the scheduler. An optimisation that serves worse answers
            # faster wins on every latency number, and nothing in the price
            # model can see it.
            rec["quality"] = []
            base = quality_baseline or {}
            for suite in (quality_suites or ()):
                try:
                    q = asyncio.run(quality_mod.run(
                        SERVER_URL, sc.model, suite=suite, n=quality_n,
                        baseline_accuracy=base.get(suite)))
                    bad, why = quality_mod.regressed(q, tolerance_pp=quality_tolerance_pp)
                    row = {**q.as_dict(), "regressed": bad, "why": why}
                    rec["quality"].append(row)
                    print(f"quality {suite}: {q.correct}/{q.n} = "
                          f"{q.accuracy:.1%}"
                          + (f"  ({q.delta_pct:+.1f} pts vs baseline)"
                             if q.delta_pct is not None else "")
                          + ("   REGRESSION" if bad else ""), flush=True)
                except Exception as e:
                    rec["quality"].append({"suite": suite,
                                           "error": f"{type(e).__name__}: {e}"})
                    print(f"quality {suite} failed: {e}", flush=True)

            async def measure(n, secs):
                async with srv.BatchSampler(SERVER_URL) as bs:
                    out = await run_concurrent_users(
                        make_session, SERVER_URL, sc.model, n, secs)
                return out, bs.summary()

            for n_users in levels:
                for rep in range(repeats):
                    # Profiling perturbs what it measures, so it runs at one
                    # level only and never at the level the price comes from
                    # unless asked for explicitly.
                    profiling = profile_level and n_users == profile_level and rep == 0
                    if profiling:
                        pdir = f"/results/profiles/{int(time.time())}-N{n_users}"
                        os.makedirs(pdir, exist_ok=True)
                        rec.setdefault("profiles", []).append(
                            {"level": n_users, "dir": pdir,
                             "start": asyncio.run(srv.start_profile(
                                 SERVER_URL, pdir, profile_steps))})
                    before = asyncio.run(srv.scrape(SERVER_URL))
                    t0 = time.perf_counter()
                    res, batch = asyncio.run(measure(n_users, seconds_per_level))
                    wall = time.perf_counter() - t0
                    after = asyncio.run(srv.scrape(SERVER_URL))

                    warm = min(20.0, 0.2 * seconds_per_level)
                    m = summarize(res, sl, warmup_s=warm)
                    ctr = (srv.diff(before, after) or {}).get("counters", {}) \
                        if (before and after) else {}
                    p_tok = ctr.get("sglang:prompt_tokens_total", 0.0)
                    c_tok = ctr.get("sglang:cached_tokens_total", 0.0)
                    g_tok = ctr.get("sglang:generation_tokens_total", 0.0)

                    lvl = {
                        "n_users": n_users, "repeat": rep,
                        "wall_s": round(wall, 1),
                        "goodput_rps": m["goodput_rps"],
                        "throughput_rps": m["throughput_rps"],
                        "good_frac": m["good_frac"], "n_failed": m["n_failed"],
                        # Every percentile, not just the one we judge at today.
                        "ttft_ms": m["ttft_ms"], "tpot_ms": m["tpot_ms"],
                        "e2e_ms": m["e2e_ms"],
                        "prompt_tokens": p_tok, "cached_tokens": c_tok,
                        "uncached_tokens": max(0.0, p_tok - c_tok),
                        "output_tokens": g_tok,
                        "cache_hit_rate": round(c_tok / p_tok, 4) if p_tok else None,
                        "batch": batch,
                        "collapse": detect_collapse(res),
                        "client_health": client_health(res),
                        "server_counters": ctr,
                    }
                    if profiling:
                        asyncio.run(srv.stop_profile(SERVER_URL))
                        results_vol.commit()
                        lvl["profile_dir"] = rec["profiles"][-1]["dir"]
                    v = sl.judge(lvl)
                    lvl["meets_slo"] = v.ok
                    lvl["slo_checks"] = v.checks
                    lvl["slo_warnings"] = v.warnings
                    rec["levels"].append(lvl)
                    print(f"  N={n_users:<5} rep{rep}  goodput {lvl['goodput_rps']:>6.2f}"
                          f"  batch {(batch.get('running') or {}).get('mean', 0):>5.1f}"
                          f"  hit {(lvl['cache_hit_rate'] or 0):.2f}"
                          f"  {'OK' if v.ok else 'MISS  <- ' + (v.binding or '')}",
                          flush=True)

            rec["status"] = "ok"
        except Exception as e:
            import traceback
            rec["failure"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-4000:]
            print("FAILED:", rec["failure"], flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    with contextlib.suppress(Exception):
        rec["server_log_tail"] = pathlib.Path(log_path).read_text(
            errors="replace")[-6000:]
    return _save(rec, sc)


def _save(rec: dict, sc) -> dict:
    import os
    rec["finished_at"] = time.time()
    stamp = f"{int(time.time())}-sweep-{sc.digest()}"
    os.makedirs("/results/runs", exist_ok=True)
    path = f"/results/runs/{stamp}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    results_vol.commit()
    rec["result_path"] = path
    print(f"saved {path}", flush=True)
    return rec


@app.function(image=image, volumes={"/results": results_vol}, timeout=900)
def fetch_profile(rel_dir: str) -> dict:
    """Return the captured trace files, so they can be ingested locally.

    Bytes rather than a path: a kineto trace is a few MB and the alternative is
    teaching every caller how to mount a Modal volume.
    """
    import base64
    import os

    d = rel_dir if rel_dir.startswith("/results") else f"/results/profiles/{rel_dir}"
    out = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.path.getsize(p) < 200_000_000:
                with open(p, "rb") as f:
                    out.append({"name": name, "size": os.path.getsize(p),
                                "b64": base64.b64encode(f.read()).decode()})
    return {"dir": d, "files": out}


@app.function(image=image, volumes={"/results": results_vol}, timeout=600)
def fetch(name: str) -> dict:
    """Read a stored sweep back. Used by `Simulator.collect`."""
    return json.load(open(f"/results/runs/{name}"))


@app.function(image=image, volumes={"/results": results_vol}, timeout=600)
def ls(limit: int = 40) -> list[str]:
    import os
    d = "/results/runs"
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d))[-limit:]
