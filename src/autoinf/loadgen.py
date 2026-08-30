"""Scalable load generation.

Measured limits of the single-event-loop client (mock server, 500-token
prompts, 4s responses, 120 tokens each):

    concurrent streams   ITL error, json parse   ITL error, byte scan
                   800                    1.7%                   1.2%
                  1600                    2.2%                   2.2%
                  2400                   65.9%                  45.0%

**One event loop is trustworthy to roughly 1600 concurrent streams** and falls
apart by 2400. The failure is quiet and dangerous: the client's own scheduling
delay is attributed to the server, so inter-token latency reads high and a
*worse* config can look better purely by loading the client differently.

Where that leaves us:

  * 1xH100 today runs near 146 concurrent — 10x headroom.
  * 8xH100 at ~90 rps with ~10s end-to-end is ~900 — still inside the limit.

So **single-process is the default and multiprocessing is not needed yet.**
That is a deliberate choice, not an oversight: sharding brings real hazards
(see `run_trace_mp`) and buys nothing below the wall.

What is worth doing now:

1. **Byte-scan SSE parsing.** Only two things matter per chunk — the first
   content delta and the final usage frame. `json.loads` on every chunk costs
   ~13k parses/second at 56 rps for information we discard. This did not move
   the wall (the limit is event-loop scheduling, not parsing) but it lowers CPU
   and cut distortion at the wall from 66% to 45%.

2. **uvloop** where available: a 2-4x faster loop for free.

3. **`client_health()` on every run.** The real protection is not a bigger
   client, it is knowing when the client is the bottleneck. A run whose
   dispatch lag or ITL is suspect must be discarded, not interpreted.

`run_trace_mp` exists for when we do cross the wall, with its constraints
documented on the function.
"""
from __future__ import annotations

import asyncio
import json
import math
import multiprocessing as mp
import os
import time

from .metrics import RequestResult
from .workload import Trace

# Distortion sets in between 1600 and 2400; keep a healthy margin.
SAFE_CONCURRENCY_PER_WORKER = 1200

# Client-health thresholds. Above these a run is not evidence.
LAG_P99_WARN_MS = 25.0
LAG_P99_FAIL_MS = 100.0

_CONTENT = b'"content":"'
_USAGE = b'"usage"'
_DONE = b"[DONE]"
_QUOTE = 0x22


def install_fast_loop() -> str:
    try:
        import uvloop
        uvloop.install()
        return "uvloop"
    except Exception:
        return "asyncio"


def plan(trace: Trace, assumed_e2e_s: float = 6.0,
         max_workers: int | None = None) -> dict:
    """How many processes this trace needs, and why."""
    rate = trace.observed_rate
    peak = rate
    # For time-varying arrivals the peak, not the mean, sets the requirement.
    if trace.requests:
        span = max(1.0, trace.duration_s / 20)
        counts: dict[int, int] = {}
        for r in trace.requests:
            b = int(r.arrival_s / span)
            counts[b] = counts.get(b, 0) + 1
        peak = max(counts.values()) / span if counts else rate

    concurrency = peak * assumed_e2e_s
    need = max(1, math.ceil(concurrency / SAFE_CONCURRENCY_PER_WORKER))
    cap = max_workers or max(1, (os.cpu_count() or 4) - 2)
    workers = min(need, cap)
    return {
        "mean_rate_rps": round(rate, 2),
        "peak_rate_rps": round(peak, 2),
        "assumed_e2e_s": assumed_e2e_s,
        "estimated_peak_concurrency": round(concurrency),
        "workers_needed": need,
        "workers_used": workers,
        "capped_by_cpu": need > cap,
        "safe_per_worker": SAFE_CONCURRENCY_PER_WORKER,
    }


async def _one(session, url: str, model: str, req, start_wall: float,
               timeout_s: float) -> RequestResult:
    import aiohttp

    # Absolute wall-clock scheduling. Every worker computes the same instant for
    # a given request, so sharding cannot shift the arrival process.
    target = start_wall + req.arrival_s
    delay = target - time.time()
    if delay > 0:
        await asyncio.sleep(delay)

    dispatched = time.time() - start_wall
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }

    first = None
    end = None
    n = 0
    u_in = u_out = None
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            if r.status != 200:
                body = (await r.text())[:200]
                return RequestResult(req.idx, req.arrival_s, dispatched, None, None,
                                     None, 0, False, f"HTTP {r.status}: {body}")
            async for raw in r.content:
                if not raw.startswith(b"data:"):
                    continue
                # Fast path: a content delta, identified without JSON parsing.
                i = raw.find(_CONTENT)
                if i != -1:
                    j = i + len(_CONTENT)
                    if j < len(raw) and raw[j] != _QUOTE:      # non-empty content
                        if first is None:
                            first = time.time() - start_wall
                        n += 1
                        continue
                if _DONE in raw:
                    break
                if _USAGE in raw:
                    try:
                        ch = json.loads(raw[5:].strip())
                    except Exception:
                        continue
                    if ch.get("usage"):
                        u_in = ch["usage"].get("prompt_tokens")
                        u_out = ch["usage"].get("completion_tokens")
            end = time.time() - start_wall
    except asyncio.TimeoutError:
        return RequestResult(req.idx, req.arrival_s, dispatched, first, None, None,
                             n, False, f"timeout after {timeout_s}s")
    except Exception as e:
        return RequestResult(req.idx, req.arrival_s, dispatched, first, None, None,
                             n, False, f"{type(e).__name__}: {e}"[:200])

    out = u_out if u_out is not None else n
    ok = first is not None and out > 0
    return RequestResult(req.idx, req.arrival_s, dispatched, first, end, u_in, out,
                         ok, None if ok else "no tokens returned")


async def _run_shard(requests, base_url: str, model: str, start_wall: float,
                     timeout_s: float, conn_limit: int) -> list[RequestResult]:
    import aiohttp

    url = base_url.rstrip("/") + "/v1/chat/completions"
    conn = aiohttp.TCPConnector(limit=conn_limit, force_close=False,
                                enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=conn) as s:
        tasks = [asyncio.create_task(_one(s, url, model, r, start_wall, timeout_s))
                 for r in requests]
        return list(await asyncio.gather(*tasks))


def _worker(requests, base_url, model, start_wall, timeout_s, conn_limit, q) -> None:
    install_fast_loop()
    try:
        res = asyncio.run(_run_shard(requests, base_url, model, start_wall,
                                     timeout_s, conn_limit))
        # Tuples, not dataclasses: cheaper to pickle across the process boundary.
        q.put([(r.idx, r.scheduled_s, r.dispatched_s, r.first_token_s, r.end_s,
                r.prompt_tokens, r.output_tokens, r.ok, r.error) for r in res])
    except Exception as e:
        q.put([("__error__", f"{type(e).__name__}: {e}")])


def run_trace_mp(trace: Trace, base_url: str, model: str, n_workers: int,
                 timeout_s: float = 600.0, lead_in_s: float = 2.0,
                 conn_limit: int = 8192) -> list[RequestResult]:
    """Replay a trace across `n_workers` processes sharing one wall clock.

    Safe to call with an event loop running in the caller: workers are spawned,
    not forked, so they inherit no loop state.
    """
    # Both start methods have a trap here, and neither is universally right:
    #
    #   fork  -- a child of a process with a live asyncio event loop inherits
    #            corrupt loop state and deadlocks. This bit us the first time
    #            this function was tested from a parent running an aiohttp
    #            server.
    #   spawn -- children re-import the caller's __main__, so any module-level
    #            side effect there runs again per worker. This also bit us: a
    #            test without an `if __name__ == "__main__"` guard recursively
    #            spawned itself.
    #
    # fork is chosen because our caller (`bench`) invokes this from synchronous
    # code, and the guard below turns the fork hazard into a loud error instead
    # of a hang. The spawn hazard cannot be guarded from here at all, since it
    # depends on a module we do not control.
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "run_trace_mp() must not be called from inside a running event "
            "loop: forked workers would inherit broken loop state and hang. "
            "Call it from synchronous code, or use run_trace_sp()."
        )
    except RuntimeError as e:
        if "must not be called" in str(e):
            raise
    ctx = mp.get_context("fork")
    shards = [trace.requests[i::n_workers] for i in range(n_workers)]

    # Every worker needs the same origin. lead_in covers process startup so the
    # first request is not already late by the time a worker is running.
    start_wall = time.time() + lead_in_s

    q = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(sh, base_url, model, start_wall, timeout_s,
                               conn_limit, q), daemon=True)
             for sh in shards if sh]
    for p in procs:
        p.start()

    # Drain the queue *before* joining. A child blocks on a full pipe until the
    # parent reads it, so joining first would deadlock on any sizeable result.
    collected: list[RequestResult] = []
    errors: list[str] = []
    deadline = time.time() + timeout_s + 120
    for _ in procs:
        remaining = max(1.0, deadline - time.time())
        try:
            rows = q.get(timeout=remaining)
        except Exception:
            errors.append("worker did not report before deadline")
            break
        if rows and isinstance(rows[0], tuple) and rows[0][0] == "__error__":
            errors.append(rows[0][1])
            continue
        for t in rows:
            collected.append(RequestResult(*t))
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)

    if errors:
        raise RuntimeError("load worker failed: " + "; ".join(errors))
    return sorted(collected, key=lambda r: r.idx)


async def run_trace_sp(trace: Trace, base_url: str, model: str,
                       timeout_s: float = 600.0, lead_in_s: float = 0.5,
                       conn_limit: int = 8192) -> list[RequestResult]:
    """Single-process path, sharing the same wall-clock scheduling as the MP one."""
    start_wall = time.time() + lead_in_s
    res = await _run_shard(trace.requests, base_url, model, start_wall,
                           timeout_s, conn_limit)
    return sorted(res, key=lambda r: r.idx)


def client_health(results, plan_info: dict | None = None) -> dict:
    """Was the *client* healthy enough for these numbers to mean anything?

    Every server metric in a run is conditional on the load generator having
    kept up. When it does not, the failure is silent and looks like server
    latency, so this verdict belongs in every run record rather than being
    checked only when something looks odd.
    """
    if not results:
        return {"verdict": "no data"}
    lags = sorted(r.dispatch_lag_ms for r in results)
    q = lambda p: lags[min(len(lags) - 1, int(len(lags) * p))]
    p99, mx = q(0.99), lags[-1]

    # A lag that grows through the run means the client fell progressively
    # behind -- worse than a uniformly high lag, because it biases late requests.
    half = len(results) // 2
    early = sum(r.dispatch_lag_ms for r in results[:half]) / max(1, half)
    late = sum(r.dispatch_lag_ms for r in results[half:]) / max(1, len(results) - half)
    drifting = late > max(5.0, early * 3)

    if p99 > LAG_P99_FAIL_MS or drifting:
        verdict = "SUSPECT -- discard; the client could not keep up"
    elif p99 > LAG_P99_WARN_MS:
        verdict = "MARGINAL -- client under strain, treat small differences with care"
    else:
        verdict = "OK"

    out = {
        "verdict": verdict,
        "lag_p50_ms": round(q(0.5), 2),
        "lag_p99_ms": round(p99, 2),
        "lag_max_ms": round(mx, 2),
        "lag_early_mean_ms": round(early, 2),
        "lag_late_mean_ms": round(late, 2),
        "drifting": drifting,
    }
    if plan_info:
        out["plan"] = plan_info
    return out


# ── multi-turn ───────────────────────────────────────────────────

async def _one_session(session, base_url: str, model: str, start_wall: float,
                       timeout_s: float, http, out: list) -> None:
    """Run one conversation. Turns are sequential; the prompt grows each turn."""
    import aiohttp

    delay = (start_wall + session.arrival_s) - time.time()
    if delay > 0:
        await asyncio.sleep(delay)

    url = base_url.rstrip("/") + "/v1/chat/completions"
    history: list[dict] = [{"role": "system", "content": session.system}]

    for k, turn in enumerate(session.turns):
        history.append({"role": "user", "content": turn.text})
        hist_tokens = sum(len(m["content"]) for m in history) // 4

        scheduled = time.time() - start_wall
        dispatched = scheduled
        first = end = None
        n = 0
        u_in = u_out = None
        reply: list[str] = []
        err = None
        try:
            async with http.post(url, json={
                "model": model, "messages": history,
                "max_tokens": turn.max_tokens, "temperature": 0.0,
                "stream": True, "stream_options": {"include_usage": True},
                "ignore_eos": True,
            }, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
                if r.status != 200:
                    err = f"HTTP {r.status}"
                else:
                    async for raw in r.content:
                        if not raw.startswith(b"data:"):
                            continue
                        i = raw.find(_CONTENT)
                        if i != -1:
                            j = i + len(_CONTENT)
                            if j < len(raw) and raw[j] != _QUOTE:
                                if first is None:
                                    first = time.time() - start_wall
                                n += 1
                                try:
                                    ch = json.loads(raw[5:].strip())
                                    piece = (ch["choices"][0].get("delta") or {}).get("content")
                                    if piece:
                                        reply.append(piece)
                                except Exception:
                                    pass
                                continue
                        if _DONE in raw:
                            break
                        if _USAGE in raw:
                            try:
                                ch = json.loads(raw[5:].strip())
                            except Exception:
                                continue
                            if ch.get("usage"):
                                u_in = ch["usage"].get("prompt_tokens")
                                u_out = ch["usage"].get("completion_tokens")
                    end = time.time() - start_wall
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:160]

        out_tokens = u_out if u_out is not None else n
        ok = err is None and first is not None and out_tokens > 0
        out.append(RequestResult(
            idx=len(out), scheduled_s=scheduled, dispatched_s=dispatched,
            first_token_s=first, end_s=end, prompt_tokens=u_in,
            output_tokens=out_tokens, ok=ok, error=err,
            session=session.idx, turn=k, history_tokens=hist_tokens))

        if not ok:
            return                      # a failed turn ends the conversation
        history.append({"role": "assistant", "content": "".join(reply)})
        if k < len(session.think_s):
            await asyncio.sleep(session.think_s[k])


async def run_sessions(trace, base_url: str, model: str,
                       timeout_s: float = 600.0, lead_in_s: float = 1.0,
                       conn_limit: int = 8192) -> list[RequestResult]:
    """Replay a SessionTrace.

    Sessions start on schedule regardless of load (open-loop arrivals), but a
    session's later turns wait for its earlier replies (closed-loop within the
    conversation). A slower server therefore receives fewer turns per second
    from the same users, which is real backpressure and is what production
    looks like -- a purely open-loop multi-turn trace would keep firing turn 5
    while turn 2 was still unanswered, which no chat client does.
    """
    import aiohttp

    start_wall = time.time() + lead_in_s
    out: list[RequestResult] = []
    conn = aiohttp.TCPConnector(limit=conn_limit, force_close=False,
                                enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=conn) as http:
        await asyncio.gather(*[
            asyncio.create_task(
                _one_session(s, base_url, model, start_wall, timeout_s, http, out))
            for s in trace.sessions
        ], return_exceptions=True)
    return sorted(out, key=lambda r: (r.session, r.turn))


def summarize_turns(results: list[RequestResult], max_depth: int = 8) -> dict:
    """Break results down by conversation depth.

    The headline question for multi-turn: does TTFT fall as the conversation
    grows? It should. The prompt gets longer every turn, but almost all of it
    was cached by the previous turn, so only the new tail needs prefilling. If
    TTFT instead rises with depth, prefix caching is not working on real
    conversational traffic -- whatever the static-prefix workload reports.
    """
    from .metrics import percentile

    by: dict[int, list[RequestResult]] = {}
    for r in results:
        if r.ok and r.turn >= 0:
            by.setdefault(min(r.turn, max_depth), []).append(r)

    rows = {}
    for d, rs in sorted(by.items()):
        ttfts = [r.ttft_ms for r in rs if r.ttft_ms is not None]
        rows[str(d)] = {
            "n": len(rs),
            "mean_prompt_tokens": round(sum(r.prompt_tokens or 0 for r in rs) / len(rs)),
            "mean_history_tokens": round(sum(r.history_tokens for r in rs) / len(rs)),
            "ttft_p50_ms": round(percentile(ttfts, 50) or 0, 1),
            "ttft_p99_ms": round(percentile(ttfts, 99) or 0, 1),
        }
    return {"by_turn_depth": rows, "n_sessions": len({r.session for r in results})}
