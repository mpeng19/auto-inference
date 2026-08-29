"""Load-generator tests against a fake SSE server with known timings.

The point is to prove the client measures what we believe it measures. A
benchmark client that mis-attributes TTFT, or that silently serialises
requests, will produce numbers that look fine and rank configs wrongly.
"""
import asyncio
import json

import pytest
from aiohttp import web

from autoinf.bench import run_trace, wait_until_ready
from autoinf.config import WorkloadConfig
from autoinf.workload import Request, Trace, build_trace


# Fake server knobs, set per-test.
STATE = {"ttft_s": 0.05, "itl_s": 0.01, "n_tokens": 5, "status": 200, "hang": False}


async def _health(request):
    return web.Response(text="ok")


async def _chat(request):
    if STATE["status"] != 200:
        return web.Response(status=STATE["status"], text="boom")

    body = await request.json()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    if STATE["hang"]:
        await asyncio.sleep(60)

    await asyncio.sleep(STATE["ttft_s"])
    n = STATE["n_tokens"]
    for i in range(n):
        if i:
            await asyncio.sleep(STATE["itl_s"])
        chunk = {"choices": [{"delta": {"content": "x"}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

    usage = {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": n}}
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


@pytest.fixture
async def server(aiohttp_server):
    app = web.Application()
    app.router.add_get("/health", _health)
    app.router.add_post("/v1/chat/completions", _chat)
    return await aiohttp_server(app)


def _trace(n, rate):
    reqs = tuple(
        Request(i, i / rate, "hello", 5, 8, None) for i in range(n)
    )
    return Trace(reqs, WorkloadConfig(n_requests=n, request_rate=rate))


@pytest.mark.asyncio
async def test_measures_ttft_and_tpot(server):
    STATE.update(ttft_s=0.10, itl_s=0.02, n_tokens=6, status=200, hang=False)
    url = f"http://{server.host}:{server.port}"
    res = await run_trace(_trace(4, 50.0), url, "fake")

    assert all(r.ok for r in res), [r.error for r in res]
    for r in res:
        assert 90 < r.ttft_ms < 220, r.ttft_ms          # ~100ms
        assert 12 < r.tpot_ms < 45, r.tpot_ms           # ~20ms
        assert r.output_tokens == 6                     # from usage, not deltas
        assert r.prompt_tokens == 11


@pytest.mark.asyncio
async def test_is_open_loop_not_serialised(server):
    """20 requests at 100/s against a 200ms server must finish in ~200-500ms.

    If the client were closed-loop / serialised it would take 20 * 200ms = 4s.
    """
    STATE.update(ttft_s=0.20, itl_s=0.0, n_tokens=2, status=200, hang=False)
    url = f"http://{server.host}:{server.port}"
    tr = _trace(20, 100.0)

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    res = await run_trace(tr, url, "fake")
    elapsed = loop.time() - t0

    assert all(r.ok for r in res)
    assert elapsed < 1.0, f"serialised? took {elapsed:.2f}s"
    # Client should not have fallen behind schedule.
    assert max(r.dispatch_lag_ms for r in res) < 150


@pytest.mark.asyncio
async def test_http_error_is_recorded_not_raised(server):
    STATE.update(status=500)
    url = f"http://{server.host}:{server.port}"
    res = await run_trace(_trace(3, 50.0), url, "fake")
    STATE.update(status=200)

    assert all(not r.ok for r in res)
    assert all("HTTP 500" in (r.error or "") for r in res)


@pytest.mark.asyncio
async def test_timeout_is_recorded_not_raised(server):
    STATE.update(hang=True, status=200)
    url = f"http://{server.host}:{server.port}"
    res = await run_trace(_trace(2, 50.0), url, "fake", timeout_s=0.5)
    STATE.update(hang=False)

    assert all(not r.ok for r in res)
    assert all("timeout" in (r.error or "") for r in res)


@pytest.mark.asyncio
async def test_wait_until_ready(server):
    url = f"http://{server.host}:{server.port}"
    waited = await wait_until_ready(url, timeout_s=10)
    assert waited < 5.0
