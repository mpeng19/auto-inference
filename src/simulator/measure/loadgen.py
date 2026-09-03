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

So **single-process is the default and there is no multiprocessing path.**
That is a deliberate choice, not an oversight: sharding load across worker
processes buys nothing below the wall and costs a second clock to reconcile,
so the sharded generator was removed rather than carried unused.

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
"""
from __future__ import annotations

import asyncio
import json
import time

from .metrics import RequestResult

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


def client_health(results) -> dict:
    """Was the *client* healthy enough for these numbers to mean anything?

    Every server metric in a run is conditional on the load generator having
    kept up. When it does not, the failure is silent and looks like server
    latency, so this verdict belongs in every run record rather than being
    checked only when something looks odd.
    """
    if not results:
        return {"verdict": "no data"}
    lags = sorted(r.dispatch_lag_ms for r in results)
    def q(p):
        return lags[min(len(lags) - 1, int(len(lags) * p))]

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

    return {
        "verdict": verdict,
        "lag_p50_ms": round(q(0.5), 2),
        "lag_p99_ms": round(p99, 2),
        "lag_max_ms": round(mx, 2),
        "lag_early_mean_ms": round(early, 2),
        "lag_late_mean_ms": round(late, 2),
        "drifting": drifting,
    }


# ── closed-loop concurrency (the SLO frontier) ───────────────────

async def _user_worker(uid: int, make_session, base_url: str, model: str,
                       start_wall: float, deadline: float, timeout_s: float,
                       http, out: list) -> None:
    """One simulated user: conversation, think, another conversation, forever."""
    n = 0
    while time.time() < deadline:
        sess = make_session(uid, n)
        # History is owned by the *worker*, not the Session. Sessions come from
        # a shared pool, so attaching history to one would let two users
        # interleave the same conversation, and would never reset between
        # loops -- prompts would grow without bound and every length in the run
        # would be meaningless.
        history = [{"role": "system", "content": sess.system}]
        for k, turn in enumerate(sess.turns):
            if time.time() >= deadline:
                return
            await _turn(history, k, turn, uid, base_url, model, start_wall,
                        timeout_s, http, out)
            if k < len(sess.think_s):
                await asyncio.sleep(min(sess.think_s[k], max(0.0, deadline - time.time())))
        n += 1


async def run_concurrent_users(make_session, base_url: str, model: str,
                               n_users: int, duration_s: float,
                               timeout_s: float = 600.0,
                               conn_limit: int = 8192,
                               grace_s: float = 3.0) -> list[RequestResult]:
    """Hold exactly `n_users` conversations in flight for `duration_s`.

    This is the marketplace question -- "how many users can we serve at once" --
    and it is genuinely closed-loop: a user cannot send their next message
    before reading the last reply. That differs from the open-loop rate sweeps
    used for scheduler comparison, and deliberately so. Under overload a
    closed-loop population self-limits (users wait longer, so send less), which
    is what real users do and what makes the concurrency axis meaningful.

    **The level ends at the deadline, not when the last reply finishes.** A
    market reply is ~2,000 tokens, 35-60 s at these speeds; letting every
    in-flight request drain made a 120 s level take six minutes and a full
    sweep forty. Requests still streaming at `deadline + grace_s` are
    cancelled (the server sees the disconnect and aborts them). Latency
    samples come only from completed requests, as before; GPU-seconds and
    token counts come from server counter deltas over the same window, so
    aborted work is counted consistently on both sides of the price.
    """
    import aiohttp

    start_wall = time.time()
    deadline = start_wall + duration_s
    out: list[RequestResult] = []
    conn = aiohttp.TCPConnector(limit=conn_limit, force_close=False,
                                enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=conn) as http:
        tasks = [asyncio.create_task(_user_worker(i, make_session, base_url, model,
                                                  start_wall, deadline, timeout_s,
                                                  http, out))
                 for i in range(n_users)]
        await run_until(tasks, duration_s + grace_s)
    return sorted(out, key=lambda r: r.dispatched_s)


async def run_until(tasks: list, timeout_s: float) -> int:
    """Wait for `tasks` up to `timeout_s`, then cancel the rest. Returns
    how many were cancelled. Separate so the cut-off is testable without a
    server."""
    if not tasks:
        return 0
    _done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return len(pending)


async def _turn(history: list[dict], k, turn, uid, base_url, model, start_wall,
                timeout_s, http, out) -> None:
    """One request within a conversation. `history` is caller-owned and mutated."""
    import aiohttp

    history.append({"role": "user", "content": turn.text})
    hist_tokens = sum(len(m["content"]) for m in history) // 4

    dispatched = time.time() - start_wall
    first = end = None
    n = 0
    u_in = u_out = None
    reply: list[str] = []
    err = None
    try:
        async with http.post(
            base_url.rstrip("/") + "/v1/chat/completions",
            json={"model": model, "messages": history,
                  "max_tokens": turn.max_tokens, "temperature": 0.0,
                  "stream": True, "stream_options": {"include_usage": True},
                  "ignore_eos": True},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as r:
            if r.status != 200:
                err = f"HTTP {r.status}"
            else:
                async for raw in r.content:
                    if not raw.startswith(b"data:"):
                        continue
                    i = raw.find(_CONTENT)
                    if i != -1 and i + len(_CONTENT) < len(raw) \
                            and raw[i + len(_CONTENT)] != _QUOTE:
                        if first is None:
                            first = time.time() - start_wall
                        n += 1
                        # A chunk we cannot parse costs one token of the reply
                        # text and nothing else: the token *count* comes from
                        # the byte scan above and the authoritative counts come
                        # from the server's usage frame, so raising here would
                        # discard a whole request's latency sample over a
                        # cosmetic loss.
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
                        # Same trade: a malformed usage frame falls back to the
                        # client's own token count (`out_tokens` below) rather
                        # than failing the request.
                        try:
                            ch = json.loads(raw[5:].strip())
                            u = ch.get("usage") or {}
                            u_in = u.get("prompt_tokens") or u_in
                            u_out = u.get("completion_tokens") or u_out
                        except Exception:
                            pass
                end = time.time() - start_wall
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:160]

    out_tokens = u_out if u_out is not None else n
    ok = err is None and first is not None and out_tokens > 0
    out.append(RequestResult(
        idx=len(out), scheduled_s=dispatched, dispatched_s=dispatched,
        first_token_s=first, end_s=end, prompt_tokens=u_in,
        output_tokens=out_tokens, ok=ok, error=err,
        session=uid, turn=k, history_tokens=hist_tokens))
    if ok:
        history.append({"role": "assistant", "content": "".join(reply)})
