"""H100 probe: answer the open questions before spending real GPU time.

Two stages, cheapest first.

    # ~2 min, no weights downloaded, ~$0.15
    PYTHONPATH=src uv run modal run src/autoinf/probe.py::probe_env

    # ~10-15 min including a 31GB download, ~$1
    PYTHONPATH=src uv run modal run src/autoinf/probe.py::probe_serve

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


async def _serving_checks(model: str) -> dict:
    import aiohttp

    res: dict = {}
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as s:
        # Q3: is ignore_eos honoured? Ask for exactly 64 tokens on a prompt the
        # model would normally answer in a handful, and see what comes back.
        want = 64
        with_ignore = await _stream(s, model, "Say hi.", want, ignore_eos=True)
        without = await _stream(s, model, "Say hi.", want, ignore_eos=False)
        res["ignore_eos_on"] = with_ignore
        res["ignore_eos_off"] = without

        got = (with_ignore.get("usage") or {}).get("completion_tokens")
        res["ignore_eos_honoured"] = (got == want)
        res["ignore_eos_note"] = (
            f"asked {want}, got {got}. If these differ, output length is not "
            "controlled and decode workload varies run to run."
        )

        # Q6: does the server report usage in the stream?
        res["usage_in_stream"] = bool(with_ignore.get("usage"))
        u = with_ignore.get("usage") or {}
        res["delta_equals_token_count"] = (with_ignore.get("deltas") == u.get("completion_tokens"))

        # Prefix cache: identical long prompt twice. The second should be
        # markedly faster to first token if the radix cache is working.
        long_prompt = ("system latency throughput scheduler " * 300) + " Summarize."
        cold = await _stream(s, model, long_prompt, 8, ignore_eos=True)
        warm = await _stream(s, model, long_prompt, 8, ignore_eos=True)
        res["prefix_cold"] = cold
        res["prefix_warm"] = warm
        if cold.get("ttft_ms") and warm.get("ttft_ms"):
            res["prefix_speedup"] = round(cold["ttft_ms"] / warm["ttft_ms"], 2)

        # Server-reported scheduler state, if exposed.
        for path in ("/get_server_info", "/health_generate"):
            try:
                async with s.get(SERVER_URL + path) as r:
                    res[f"endpoint{path}"] = (await r.text())[:800] if r.status == 200 \
                        else f"HTTP {r.status}"
            except Exception as e:
                res[f"endpoint{path}"] = f"{type(e).__name__}"
    return res


@app.local_entrypoint()
def main(stage: str = "env"):
    r = probe_env.remote() if stage == "env" else probe_serve.remote()
    print(json.dumps({k: v for k, v in r.items() if k != "server_log_tail"},
                     indent=2, default=str))
