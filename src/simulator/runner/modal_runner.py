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
# Kept in step with `api.APP_NAME`, which reads the same variable to look these
# functions up again from the client side.
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


def _server_env(extra: dict | None = None) -> dict:
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
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    # Set last: a candidate may set anything else, but not turn off the
    # counter the price is read from.
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
        # Raising the descriptor limit is an optimisation for the top of the
        # level grid, not a requirement: swallowed so a container whose policy
        # forbids it still runs the sweep, and hits the ceiling as a client
        # error the run record shows rather than as a launch failure.
        pass

    sc, sl = ServingConfig(**serving), SLO.from_dict(slo)
    st = InferenceStack.from_dict(stack)
    if st.serving:
        # The candidate's launch line. Applied before validate() so a bad
        # override fails as cheaply as a bad tp_size does.
        try:
            sc = sc.with_overrides(st.serving)
        except ValueError as e:
            rec_early = {"status": "failed", "note": note, "serving": serving,
                         "stack_digest": st.digest, "levels": [],
                         "failure": f"serving override rejected: {e}",
                         "started_at": time.time()}
            print("INVALID OVERRIDE:", e, flush=True)
            return _save(rec_early, ServingConfig(**serving))
    install_fast_loop()

    rec: dict = {"status": "failed", "note": note, "serving": asdict(sc),
                 "serving_digest": sc.digest(), "slo": sl.as_dict(),
                 "stack_digest": st.digest, "levels": [],
                 "serving_overrides": st.serving, "server_env": st.env,
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
                                env=_server_env(st.env))
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
                    # Quality is a property of the stack, not of the sweep, so
                    # it is scored once per (stack, suite, n) and reused: a
                    # candidate promoted from a screen does not pay for GSM8K
                    # and LongBench twice, and stock never pays again.
                    cache = pathlib.Path(f"/results/quality/{st.digest}-{suite}-{quality_n}.json")
                    if cache.is_file():
                        cached = json.loads(cache.read_text())
                        q = quality_mod.QualityResult(
                            suite=suite, n=cached["n"], correct=cached["correct"],
                            errors=cached["errors"], baseline_accuracy=base.get(suite),
                            items=cached.get("items") or [])
                        q_cached = True
                    else:
                        q = asyncio.run(quality_mod.run(
                            SERVER_URL, sc.model, suite=suite, n=quality_n,
                            baseline_accuracy=base.get(suite)))
                        q_cached = False
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps({"n": q.n, "correct": q.correct,
                                                     "errors": q.errors, "items": q.items[:5]}))
                        results_vol.commit()
                    bad, why = quality_mod.regressed(q, tolerance_pp=quality_tolerance_pp)
                    row = {**q.as_dict(), "regressed": bad, "why": why, "cached": q_cached}
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
    """Read one stored sweep back by filename, for a run whose call id is lost.

    `Simulator.collect` does not use this -- it holds the Modal call and gets
    the record from `FunctionCall.get`. This is the escape hatch for the case
    that leaves nothing to collect: `ls` names the file, this returns it, and
    `simulate rescore` prices it without a GPU.
    """
    with open(f"/results/runs/{name}") as f:
        return json.load(f)


@app.function(image=image, volumes={"/results": results_vol}, timeout=600)
def ls(limit: int = 40) -> list[str]:
    d = "/results/runs"
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d))[-limit:]


# The workbench is not a sweep, so it does not need the sweep's 16 vCPUs:
# there is no load generator to feed, just one script driving one engine.
# **vCPUs are billed on top of the GPU** ($0.1368/core-hour, see
# `costs.container_rate`), so twelve idle cores would be $1.64/hour of nothing
# on top of the H100 -- about a fifth of the bill for doing no work.
WORKBENCH_VCPU = 4.0


@app.function(
    image=image, gpu="H100", cpu=WORKBENCH_VCPU,
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=_hf_secret(),
    # The ceiling, not the budget: `timeout_s` is what the script actually
    # gets. Generous because an engine load is 3-5 minutes before a kernel
    # script has run a single line.
    timeout=2 * 60 * 60,
)
def workbench(stack: dict, script: str, timeout_s: int = 600,
              files: dict[str, str] | None = None) -> dict:
    """Run one script on an H100 against a candidate stack.

    A sweep answers one question -- what does this cost to serve -- and takes
    17-35 minutes to do it. Kernel work asks different questions: does this
    Triton kernel compile, is it faster than the one it replaces, does it
    produce the same numbers. None of those need a load generator, a market
    workload or a price, and paying half an hour for each of them is why an
    agent doing kernel work would otherwise get one bit an hour.

    So: apply the stack the same way a sweep does, run a script, hand back what
    it printed. `files` are extra helper text files (path -> text) written
    beside the script; the script runs with them as its working directory, so
    `import my_helper` works.

    Returns a dict rather than raising, including when the stack is refused or
    the script times out. A caller in a search loop needs the reason as data.
    """
    import shutil

    from simulator.stack import InferenceStack, sglang_root

    t0 = time.perf_counter()
    st = InferenceStack.from_dict(stack)
    out: dict = {"ok": False, "exit_code": None, "stdout": "", "stderr": "",
                 "elapsed_s": 0.0, "gpu": _gpu_name(), "stack": {}}

    try:
        # Restores stock first, then writes the candidate over it -- the same
        # call the sweep makes, so a workbench result and a sweep result are
        # about the same code.
        out["stack"] = st.apply()
        print("stack:", st.describe(), flush=True)
    except Exception as e:
        out["stderr"] = f"STACK REFUSED: {type(e).__name__}: {e}"
        print(out["stderr"], flush=True)
        return _bill(out, t0)

    scratch = pathlib.Path(f"/tmp/workbench-{int(time.time() * 1000)}")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        out["files"] = _write_helpers(scratch, files or {})
    except ValueError as e:
        out["stderr"] = str(e)
        return _bill(out, t0)
    entry = scratch / "script.py"
    entry.write_text(script)

    # `import sglang` must reach the package `apply()` just wrote over, not
    # some copy. Naming site-packages explicitly says so; the guard in
    # `_write_helpers` covers the other end, since Python puts the script's own
    # directory ahead of PYTHONPATH.
    env = dict(os.environ)
    site = str(sglang_root().parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [site, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("SGLANG_ENABLE_METRICS_DEVICE_TIMER", "1")

    # Containers are reused between calls, so a warm one can be holding a
    # snapshot of /results taken before another container wrote to it. That is
    # exactly the equivalence reference's shape -- written by one run, read by
    # the next -- and the failure it produces is an intermittent "the reference
    # is not there" that looks like anything but a stale mount.
    with contextlib.suppress(Exception):
        results_vol.reload()

    s0 = time.perf_counter()
    try:
        r = subprocess.run(["python", str(entry)], cwd=str(scratch), env=env,
                           capture_output=True, text=True, timeout=timeout_s)
        out["exit_code"], out["stdout"], out["stderr"] = r.returncode, r.stdout, r.stderr
        out["ok"] = r.returncode == 0
    except subprocess.TimeoutExpired as e:
        out["exit_code"] = -1
        out["stdout"] = (e.stdout or b"").decode(errors="replace") \
            if isinstance(e.stdout, bytes) else (e.stdout or "")
        out["stderr"] = ((e.stderr or b"").decode(errors="replace")
                         if isinstance(e.stderr, bytes) else (e.stderr or "")) \
            + f"\nTIMEOUT: the script did not finish within {timeout_s}s"
    out["elapsed_s"] = round(time.perf_counter() - s0, 2)
    # Anything the script wrote under /results is only durable once committed;
    # `equivalence` depends on this to cache its reference across containers.
    with contextlib.suppress(Exception):
        results_vol.commit()

    # Tails, not the whole log: a Triton autotune run prints megabytes, and the
    # useful part of a failure is always at the end.
    out["stdout"], out["stderr"] = out["stdout"][-20000:], out["stderr"][-20000:]
    print(f"script exited {out['exit_code']} after {out['elapsed_s']}s", flush=True)
    return _bill(out, t0)


def _bill(out: dict, t0: float) -> dict:
    """Attach what this container cost us, at Modal's retail rate.

    Billed on the whole function -- applying the stack and committing the
    volume are container-seconds too -- not on `elapsed_s`, which is the
    script alone. It still undercounts by the container start and image pull,
    which Modal bills and we cannot see from in here.
    """
    from simulator import costs

    out["container_s"] = round(time.perf_counter() - t0, 2)
    out["cost_usd"] = round(
        out["container_s"] * costs.container_rate("H100", 1, vcpu=WORKBENCH_VCPU)
        / 3600.0, 4)
    return out


def _gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception as e:
        return f"unavailable: {e}"


def _write_helpers(scratch: pathlib.Path, files: dict[str, str]) -> list[str]:
    """Write the helper files, refusing the two ways they can lie.

    A path escaping the scratch directory would write over the container, and
    anything called `sglang` there would shadow the applied package -- Python
    puts the script's own directory at the front of `sys.path`, ahead of the
    PYTHONPATH that names site-packages. Both fail loudly rather than producing
    a run that measures something other than the stack.
    """
    root = scratch.resolve()
    written = []
    for rel, text in sorted(files.items()):
        if rel.split("/")[0].split(".")[0] == "sglang":
            raise ValueError(
                f"helper file {rel!r} would shadow the applied sglang package; "
                "name it something else")
        p = (scratch / rel).resolve()
        if not str(p).startswith(str(root) + os.sep):
            raise ValueError(f"helper file {rel!r} escapes the scratch directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        written.append(rel)
    return written


def _inside(root: str, path: str) -> bool:
    """Is `path` under `root` once **both** are resolved?

    Both sides, because a volume mount is a symlink: comparing the resolved
    file against the literal "/results" rejected every real file on the first
    run.
    """
    r = os.path.realpath(root)
    return os.path.commonpath([r, os.path.realpath(path)]) == r


@app.function(image=image, volumes={"/results": results_vol}, timeout=600)
def read_results(paths: list[str]) -> dict:
    """Read JSON files off the results volume, by path. One call, many files.

    `workbench` caps stdout at 20 KB, which is the right size for a script's
    log and an order of magnitude too small for a teacher-forced scoring run
    (2,000 positions x three numbers). Those land on the volume, and this is
    how they come back -- including the equivalence reference, which is
    computed once and then read by every candidate after it.
    """
    results_vol.reload()          # another container wrote these, not this one
    out: dict = {}
    for p in paths:
        full = p if p.startswith("/results/") else f"/results/{p.lstrip('/')}"
        if not _inside("/results", full):
            out[p] = {"ok": False, "error": "path escapes /results"}
            continue
        if not os.path.isfile(full):
            out[p] = {"ok": False, "error": "not found"}
            continue
        try:
            with open(full) as f:
                out[p] = {"ok": True, "json": json.load(f)}
        except Exception as e:
            out[p] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return out
