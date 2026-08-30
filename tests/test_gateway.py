"""The recording proxy must stream through untouched and record shape correctly."""
import json
import os
import tempfile

import pytest
from aiohttp import web

from autoinf.gateway import Capture, session_key, summarize_captures


async def _await_captures(path, n, timeout=3.0):
    """Recording happens after the response completes, by design -- a client
    must never wait on our bookkeeping. Tests therefore have to wait for it."""
    import asyncio
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if path.exists() and len(path.read_text().splitlines()) >= n:
            break
        await asyncio.sleep(0.02)
    return [Capture(**json.loads(l)) for l in path.read_text().splitlines()]


@pytest.fixture
def trace_path(tmp_path, monkeypatch):
    p = tmp_path / "cap.jsonl"
    monkeypatch.setenv("TRACE_PATH", str(p))
    return p


async def _upstream(request):
    body = await request.json()
    n = min(body.get("max_tokens", 5), 6)
    r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await r.prepare(request)
    for _ in range(n):
        await r.write(b'data: {"choices":[{"delta":{"content":"tok "}}]}\n\n')
    await r.write(("data: " + json.dumps({"choices": [], "usage": {
        "prompt_tokens": len(json.dumps(body["messages"])) // 4,
        "completion_tokens": n}}) + "\n\n").encode())
    await r.write(b"data: [DONE]\n\n")
    await r.write_eof()
    return r


@pytest.fixture
async def stack(aiohttp_server, aiohttp_client, tmp_path, monkeypatch):
    up = web.Application()
    up.router.add_post("/v1/chat/completions", _upstream)
    up_server = await aiohttp_server(up)

    monkeypatch.setenv("UPSTREAM", f"http://{up_server.host}:{up_server.port}")
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "cap.jsonl"))
    import importlib

    import autoinf.proxy_app as pa
    importlib.reload(pa)                     # pick up patched env
    client = await aiohttp_client(pa.make_app())
    return client, tmp_path / "cap.jsonl", pa


def _msgs(*pairs, system="you are helpful"):
    m = [{"role": "system", "content": system}]
    for u, a in pairs:
        m.append({"role": "user", "content": u})
        if a is not None:
            m.append({"role": "assistant", "content": a})
    return m


@pytest.mark.asyncio
async def test_streams_through_untouched(stack):
    client, path, _ = stack
    r = await client.post("/v1/chat/completions", json={
        "model": "m", "messages": _msgs(("hello", None)),
        "max_tokens": 4, "stream": True})
    body = await r.text()
    assert r.status == 200
    assert body.count("tok ") == 4          # every token reached the client
    assert "[DONE]" in body


@pytest.mark.asyncio
async def test_records_conversation_growth(stack):
    client, path, _ = stack
    convo = [("fix the bug in foo.py", None)]
    for turn in range(4):
        await client.post("/v1/chat/completions", json={
            "model": "m", "messages": _msgs(*convo), "max_tokens": 5, "stream": True})
        convo[-1] = (convo[-1][0], "here is a patch " * 20)
        convo.append((f"now handle case {turn}", None))

    caps = await _await_captures(path, 4)
    assert len(caps) == 4
    # All four turns belong to one session.
    assert len({c.session for c in caps}) == 1
    assert [c.turn for c in caps] == [0, 1, 2, 3]
    # The prompt grows every turn.
    chars = [c.prompt_chars for c in caps]
    assert chars == sorted(chars) and chars[-1] > chars[0] * 2, chars
    # And most of each later prompt is a repeat of the previous one.
    later = caps[-1]
    assert later.new_chars < later.prompt_chars * 0.5


@pytest.mark.asyncio
async def test_separate_conversations_get_separate_sessions(stack):
    client, path, _ = stack
    for first in ("task A", "task B"):
        await client.post("/v1/chat/completions", json={
            "model": "m", "messages": _msgs((first, None)),
            "max_tokens": 3, "stream": True})
    caps = await _await_captures(path, 2)
    assert len({c.session for c in caps}) == 2


@pytest.mark.asyncio
async def test_records_usage_and_ttft(stack):
    client, path, _ = stack
    await client.post("/v1/chat/completions", json={
        "model": "m", "messages": _msgs(("hi", None)), "max_tokens": 6, "stream": True})
    c = (await _await_captures(path, 1))[0]
    assert c.completion_tokens == 6
    assert c.prompt_tokens and c.prompt_tokens > 0
    assert c.ttft_ms is not None and c.ttft_ms >= 0
    assert c.status == 200


def test_summary_reports_prefix_reuse():
    caps = [
        Capture(t=0, session="s", turn=0, n_messages=2, prompt_chars=1000,
                system_chars=100, new_chars=1000, prompt_tokens=250,
                completion_tokens=100, max_tokens=200, ttft_ms=40, total_ms=500,
                think_ms=None, status=200),
        Capture(t=5, session="s", turn=1, n_messages=4, prompt_chars=3000,
                system_chars=100, new_chars=200, prompt_tokens=750,
                completion_tokens=100, max_tokens=200, ttft_ms=30, total_ms=500,
                think_ms=2000, status=200),
    ]
    s = summarize_captures(caps)
    assert s["n_sessions"] == 1 and s["n_requests"] == 2
    # Turn 1 reused 2800/3000 of its prompt.
    assert s["prefix_reuse_frac"]["p50"] > 0.9
