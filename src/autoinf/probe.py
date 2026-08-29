"""H100 probe: answer the open questions before spending real GPU time.

Two stages, cheapest first.

    # ~2 min, no weights downloaded, ~$0.15
    uv run modal run src/autoinf/probe.py::probe_env

    # ~10-15 min including a 31GB download, ~$1
    uv run modal run src/autoinf/probe.py::probe_serve

`probe_env` settles the two questions that would invalidate every later
experiment — does the image build, and are the SGLang flag names real. A flag
that SGLang silently ignores produces a run that looks successful and measures
the wrong configuration, which is the worst failure mode available to us.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time

import modal

from autoinf.config import ServingConfig
from autoinf.modal_app import SERVER_PORT, SERVER_URL, hf_cache, image

app = modal.App("auto-inference-probe", image=image)


@app.function(gpu="H100", cpu=4.0, timeout=15 * 60)
def probe_env() -> dict:
    """Image, hardware and SGLang flag introspection. No model download."""
    out: dict = {}

    def sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=300).stdout.strip()
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    # ── Q4: what hardware do we actually get? ────────────────────
    out["nvidia_smi"] = sh(["nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,compute_cap",
        "--format=csv,noheader"])
    out["cpu_count"] = sh(["nproc"])
    out["mem_total"] = sh(["bash", "-lc", "free -g | awk 'NR==2{print $2\" GiB\"}'"])

    try:
        import torch
        out["torch"] = str(torch.__version__)
        out["torch_cuda"] = str(torch.version.cuda)
        out["n_gpus"] = torch.cuda.device_count()
        out["gpu_name"] = str(torch.cuda.get_device_name(0))
        out["gpu_capability"] = [int(x) for x in torch.cuda.get_device_capability(0)]
        free, total = torch.cuda.mem_get_info()
        out["gpu_mem_free_gib"] = round(free / 2**30, 1)
        out["gpu_mem_total_gib"] = round(total / 2**30, 1)
    except Exception as e:
        out["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        import sglang
        out["sglang"] = getattr(sglang, "__version__", "unknown")
    except Exception as e:
        out["sglang_error"] = f"{type(e).__name__}: {e}"

    # ── Q2: are our flag names real? ─────────────────────────────
    help_text = sh(["python", "-m", "sglang.launch_server", "--help"])
    out["help_chars"] = len(help_text)

    wanted = [a for a in ServingConfig(ep_size=1, max_total_tokens=1,
                                       context_length=1, enable_prefix_caching=False)
              .to_sglang_args() if a.startswith("--")]
    out["flag_check"] = {f: ("ok" if f in help_text else "MISSING") for f in wanted}
    out["flags_missing"] = [f for f, v in out["flag_check"].items() if v == "MISSING"]

    # Values accepted by --schedule-policy vary between releases.
    for line in help_text.splitlines():
        if "--schedule-policy" in line:
            out["schedule_policy_help"] = line.strip()[:300]

    out["verdict"] = "PASS" if not out["flags_missing"] else "FLAGS NEED FIXING"
    print(json.dumps(out, indent=2, default=str), flush=True)
    return out


@app.function(gpu="H100", cpu=8.0, volumes={"/cache": hf_cache}, timeout=45 * 60)
def probe_serve(model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8") -> dict:
    """Launch the real server and check the serving-behaviour assumptions."""
    import aiohttp

    sc = ServingConfig(model=model)
    cmd = ["python", "-m", "sglang.launch_server",
           "--host", "127.0.0.1", "--port", str(SERVER_PORT), *sc.to_sglang_args()]
    print("launching:", " ".join(cmd), flush=True)

    out: dict = {"model": model, "cmd": " ".join(cmd)}
    log_path = "/tmp/sglang-probe.log"

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            from autoinf.bench import wait_until_ready
            t0 = time.perf_counter()
            out["model_load_s"] = round(asyncio.run(wait_until_ready(SERVER_URL, 2400)), 1)
            print(f"ready in {out['model_load_s']}s", flush=True)
            out.update(asyncio.run(_serving_checks(model)))
            out["status"] = "ok"
        except Exception as e:
            out["status"] = "failed"
            out["failure"] = f"{type(e).__name__}: {e}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        out["server_log_tail"] = open(log_path, errors="replace").read()[-6000:]
    except Exception:
        pass
    print(json.dumps({k: v for k, v in out.items() if k != "server_log_tail"},
                     indent=2, default=str), flush=True)
    return out


async def _stream(session, model: str, prompt: str, max_tokens: int,
                  ignore_eos: bool) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    t0 = time.perf_counter()
    ttft = None
    deltas = 0
    usage = None
    async with session.post(SERVER_URL + "/v1/chat/completions", json=payload) as r:
        if r.status != 200:
            return {"error": f"HTTP {r.status}: {(await r.text())[:200]}"}
        async for raw in r.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ch = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    deltas += 1
    return {"ttft_ms": round(ttft * 1000, 1) if ttft else None,
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "deltas": deltas, "usage": usage}


async def _repeat(session, model, prompt, max_tokens, n, ignore_eos=True) -> list[dict]:
    out = []
    for _ in range(n):
        out.append(await _stream(session, model, prompt, max_tokens, ignore_eos))
    return out


def _dist(xs: list[float]) -> dict:
    """Median-centred summary. n=1 latency numbers are not evidence."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    mid = xs[len(xs) // 2]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return {
        "n": len(xs), "median": round(mid, 1),
        "min": round(xs[0], 1), "max": round(xs[-1], 1),
        "cv": round((var ** 0.5) / mean, 3) if mean else None,
    }


async def _flush(session) -> bool:
    """Drop the radix cache so a 'cold' measurement is genuinely cold."""
    for path in ("/flush_cache", "/flush_cache/"):
        try:
            async with session.post(SERVER_URL + path) as r:
                if r.status in (200, 204):
                    await asyncio.sleep(0.5)
                    return True
        except Exception:
            pass
    return False


async def _serving_checks(model: str) -> dict:
    import aiohttp

    res: dict = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        # ── Q3: is ignore_eos honoured? ──────────────────────────
        want = 64
        on = await _stream(s, model, "Say hi.", want, ignore_eos=True)
        off = await _stream(s, model, "Say hi.", want, ignore_eos=False)
        got = (on.get("usage") or {}).get("completion_tokens")
        res["ignore_eos"] = {
            "asked": want, "got_with_flag": got,
            "got_without_flag": (off.get("usage") or {}).get("completion_tokens"),
            "honoured": got == want,
        }

        # ── Q6: usage accounting ─────────────────────────────────
        u = on.get("usage") or {}
        res["usage_in_stream"] = bool(u)
        res["deltas_vs_tokens"] = {
            "deltas": on.get("deltas"), "completion_tokens": u.get("completion_tokens"),
            "equal": on.get("deltas") == u.get("completion_tokens"),
            "note": "if unequal, counting deltas would bias TPOT; bench.py uses usage",
        }

        # ── warm up, then measure single-request noise ───────────
        # This is a preview of the noise floor: identical requests, no load,
        # nothing else running. Whatever spread shows up here is a lower bound
        # on the spread of any benchmark number.
        await _repeat(s, model, "Say hi.", 32, 5)
        idle = await _repeat(s, model, "Say hi.", 32, 20)
        res["idle_ttft_ms"] = _dist([r.get("ttft_ms") for r in idle])
        res["idle_total_ms"] = _dist([r.get("total_ms") for r in idle])

        # ── prefix cache, measured properly ──────────────────────
        # Previous run compared one cold sample to one warm sample and got a
        # nonsense answer (warm slower than cold). Flush between trials and
        # repeat, so the comparison is between distributions.
        long_prompt = ("system latency throughput scheduler " * 300) + " Summarize."
        res["flush_supported"] = await _flush(s)

        colds, warms = [], []
        for _ in range(6):
            await _flush(s)
            c = await _stream(s, model, long_prompt, 8, True)
            w = await _stream(s, model, long_prompt, 8, True)
            if c.get("ttft_ms"):
                colds.append(c["ttft_ms"])
            if w.get("ttft_ms"):
                warms.append(w["ttft_ms"])

        res["prefix_cold_ttft_ms"] = _dist(colds)
        res["prefix_warm_ttft_ms"] = _dist(warms)
        cm = res["prefix_cold_ttft_ms"].get("median")
        wm = res["prefix_warm_ttft_ms"].get("median")
        if cm and wm:
            res["prefix_speedup_median"] = round(cm / wm, 2)
            res["prefix_verdict"] = (
                "cache helps" if cm / wm > 1.3 else
                "no measurable benefit — investigate before trusting prefix_heavy"
            )

        try:
            async with s.get(SERVER_URL + "/get_server_info") as r:
                info = await r.json()
                res["server_info"] = {k: info.get(k) for k in
                    ("max_running_requests", "max_total_num_tokens", "chunked_prefill_size",
                     "schedule_policy", "mem_fraction_static", "disable_radix_cache",
                     "quantization", "kv_cache_dtype", "context_length", "tp_size")}
        except Exception as e:
            res["server_info"] = f"{type(e).__name__}: {e}"
    return res


@app.local_entrypoint()
def main(stage: str = "env"):
    r = probe_env.remote() if stage == "env" else probe_serve.remote()
    print(json.dumps({k: v for k, v in r.items() if k != "server_log_tail"},
                     indent=2, default=str))
