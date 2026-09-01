"""Open-loop async load generator.

Fires each request at its scheduled arrival time regardless of whether earlier
requests have finished. Every request gets its own task; the dispatcher never
blocks on a response.

Two self-checks are built in, because a load generator that is itself the
bottleneck produces server metrics that look plausible and are meaningless:

  * `dispatch_lag_ms` — how late each request was actually sent. If this grows
    over the run, the client is saturated and the run must be discarded.
  * `n_failed` / `errors` — connection errors are outcomes, not exceptions.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp

from .metrics import RequestResult
from .workload import Trace


async def _one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    req,
    t0: float,
    timeout_s: float,
) -> RequestResult:
    # Sleep until this request's scheduled arrival, then fire.
    delay = req.arrival_s - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)

    scheduled_s = req.arrival_s
    dispatched_s = time.perf_counter() - t0

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Pin the decode length so the workload is controlled rather than
        # dependent on when the model happens to emit EOS. SGLang honours this
        # as an OpenAI-API extension; verify it is not silently dropped.
        "ignore_eos": True,
    }

    first_token_s: float | None = None
    end_s: float | None = None
    n_deltas = 0
    usage_out: int | None = None
    usage_in: int | None = None

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:200]
                return RequestResult(
                    req.idx, scheduled_s, dispatched_s, None, None, None, 0,
                    False, f"HTTP {resp.status}: {body}",
                )

            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage_out = chunk["usage"].get("completion_tokens")
                    usage_in = chunk["usage"].get("prompt_tokens")

                for ch in chunk.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - t0
                        n_deltas += 1

            end_s = time.perf_counter() - t0

    except asyncio.TimeoutError:
        return RequestResult(
            req.idx, scheduled_s, dispatched_s, first_token_s, None, None,
            n_deltas, False, f"timeout after {timeout_s}s",
        )
    except Exception as e:  # connection reset, server died mid-stream, ...
        return RequestResult(
            req.idx, scheduled_s, dispatched_s, first_token_s, None, None,
            n_deltas, False, f"{type(e).__name__}: {e}"[:200],
        )

    # Trust the server's usage count when it gives one; fall back to counting
    # streamed deltas (which is an approximation — a delta is not always a
    # single token).
    out_tokens = usage_out if usage_out is not None else n_deltas
    ok = first_token_s is not None and out_tokens > 0
    return RequestResult(
        req.idx, scheduled_s, dispatched_s, first_token_s, end_s,
        usage_in, out_tokens, ok, None if ok else "no tokens returned",
    )


async def run_trace(
    trace: Trace,
    base_url: str,
    model: str,
    timeout_s: float = 600.0,
    connection_limit: int = 4096,
) -> list[RequestResult]:
    """Replay `trace` against `base_url` and return one record per request."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=connection_limit, force_close=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(_one(session, url, model, r, t0, timeout_s))
            for r in trace.requests
        ]
        results = await asyncio.gather(*tasks)

    return sorted(results, key=lambda r: r.idx)


async def wait_until_ready(base_url: str, timeout_s: float = 1800.0,
                           proc=None, log_path: str | None = None,
                           stall_s: float = 420.0) -> float:
    """Block until the server answers /health. Returns seconds waited.

    Model load for a 30B MoE is minutes, not seconds; benchmarking a
    still-loading server is a classic way to produce garbage TTFT numbers.

    Three ways this can end badly, and all three are handled:

      * **The process dies.** `proc` is polled, so a crash aborts at once.
      * **The process hangs.** A live process making no progress is not caught
        by a liveness check. `stall_s` aborts when the server log has not
        changed for that long -- this actually happened: a run sat at
        "loading shards: 0%" for the full 2400s timeout, costing ~$2.60 of
        H100 time to learn nothing. Loads normally finish in 90-505s, so a
        7-minute silence means something is wrong, not slow.
      * **It is merely slow.** `timeout_s` remains the outer bound.

    `log_path` is echoed periodically so a stuck load is visible while it is
    happening rather than in a post-mortem.
    """
    url = base_url.rstrip("/") + "/health"
    start = time.perf_counter()
    last_echo = 0.0
    last_size, last_change = -1, time.perf_counter()

    def _log_size() -> int:
        try:
            return os.path.getsize(log_path) if log_path else -1
        except OSError:
            return -1

    async with aiohttp.ClientSession() as s:
        while time.perf_counter() - start < timeout_s:
            # Progress is measured by the server log growing. A live process
            # writing nothing for `stall_s` is stuck, not loading.
            if log_path:
                sz = _log_size()
                if sz != last_size:
                    last_size, last_change = sz, time.perf_counter()
                elif time.perf_counter() - last_change > stall_s:
                    tail = ""
                    try:
                        tail = open(log_path, errors="replace").read()[-2000:]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"server load stalled: no log output for {stall_s:.0f}s "
                        f"({time.perf_counter() - start:.0f}s elapsed). Normal "
                        f"loads finish in 90-505s.\n--- log tail ---\n{tail}")
            if proc is not None and proc.poll() is not None:
                tail = ""
                if log_path:
                    try:
                        tail = open(log_path, errors="replace").read()[-3000:]
                    except Exception:
                        pass
                raise RuntimeError(
                    f"server process exited with code {proc.returncode} after "
                    f"{time.perf_counter() - start:.0f}s\n--- log tail ---\n{tail}"
                )
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return time.perf_counter() - start
            except Exception:
                pass

            elapsed = time.perf_counter() - start
            if log_path and elapsed - last_echo >= 30:
                last_echo = elapsed
                try:
                    lines = open(log_path, errors="replace").read().splitlines()
                    if lines:
                        print(f"  [{elapsed:.0f}s loading] {lines[-1][:150]}", flush=True)
                except Exception:
                    pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"server not ready after {timeout_s}s")


async def complete(base_url: str, model: str, prompt: str, max_tokens: int,
                   timeout_s: float = 300.0) -> str:
    """Non-streaming completion, for canaries where only the text matters."""
    async with aiohttp.ClientSession() as s_:
        async with s_.post(
            base_url.rstrip("/") + "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0.0, "stream": False},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as r:
            if r.status != 200:
                return f"<HTTP {r.status}>"
            body = await r.json()
            return (body.get("choices") or [{}])[0].get("message", {}).get("content", "")


async def warmup(base_url: str, model: str, n: int = 20) -> float:
    """Send throwaway requests so measurement starts against a warm server.

    The first requests after launch pay for lazy CUDA graph capture and
    allocator growth; including them makes the first workload of a suite look
    systematically worse than the rest.
    """
    import time as _t
    t0 = _t.perf_counter()
    async with aiohttp.ClientSession() as s_:
        for _ in range(n):
            try:
                async with s_.post(
                    base_url.rstrip("/") + "/v1/chat/completions",
                    json={"model": model,
                          "messages": [{"role": "user", "content": "warmup"}],
                          "max_tokens": 16, "temperature": 0.0, "stream": False,
                          "ignore_eos": True},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as r:
                    await r.read()
            except Exception:
                pass
    return _t.perf_counter() - t0
