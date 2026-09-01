"""The recording proxy, run inside the container in front of SGLang.

Streams responses through byte-for-byte -- an agent must see exactly what
SGLang produced, or the traffic we record is not the traffic the agent would
have generated. Metadata is teed off the stream as it passes.
"""
from __future__ import annotations

import hmac
import json
import os
import time

import aiohttp
from aiohttp import web

from autoinf.gateway import Capture, Recorder

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:30000")
TRACE_PATH = os.environ.get("TRACE_PATH", "/results/traces/capture.jsonl")
# A public GPU-backed LLM endpoint on a paid account is an open invitation.
# Checked as a standard OpenAI-style bearer token, so any client that can talk
# to the OpenAI API can authenticate by setting its api_key -- no special
# handling needed on the agent side.
API_KEY = os.environ.get("GATEWAY_API_KEY", "")
_rec = Recorder(TRACE_PATH)


@web.middleware
async def _auth(request, handler):
    if not API_KEY or request.path == "/health":
        return await handler(request)
    sent = (request.headers.get("Authorization", "")
            .removeprefix("Bearer ").strip())
    if not hmac.compare_digest(sent, API_KEY):
        return web.json_response(
            {"error": {"message": "invalid api key", "type": "invalid_request_error"}},
            status=401)
    return await handler(request)


async def _models(request):
    async with aiohttp.ClientSession() as s:
        async with s.get(UPSTREAM + "/v1/models") as r:
            return web.json_response(await r.json(), status=r.status)


async def _health(request):
    return web.json_response({"ok": True, "captured": len(_rec._state)})


async def _chat(request):
    body = await request.json()
    messages = body.get("messages") or []
    key, turn, meta = _rec.observe(messages, body)

    t0 = time.time()
    first = None
    usage_in = usage_out = None
    stream = bool(body.get("stream"))

    status = 0
    resp = None
    async with aiohttp.ClientSession() as s:
        try:
            async with s.post(UPSTREAM + "/v1/chat/completions", json=body,
                              timeout=aiohttp.ClientTimeout(total=3600)) as up:
                if not stream:
                    payload = await up.json()
                    u = payload.get("usage") or {}
                    usage_in, usage_out = u.get("prompt_tokens"), u.get("completion_tokens")
                    resp = web.json_response(payload, status=up.status)
                else:
                    resp = web.StreamResponse(
                        status=up.status,
                        headers={"Content-Type": "text/event-stream",
                                 "Cache-Control": "no-cache"})
                    await resp.prepare(request)
                    async for raw in up.content:
                        # Forward verbatim, then inspect. Never the other way
                        # round: the client must not wait on our bookkeeping.
                        await resp.write(raw)
                        if not raw.startswith(b"data:"):
                            continue
                        if first is None and b'"content":"' in raw:
                            j = raw.find(b'"content":"') + 11
                            if j < len(raw) and raw[j] != 0x22:
                                first = time.time()
                        if b'"usage"' in raw:
                            try:
                                ch = json.loads(raw[5:].strip())
                                u = ch.get("usage") or {}
                                usage_in = u.get("prompt_tokens") or usage_in
                                usage_out = u.get("completion_tokens") or usage_out
                            except Exception:
                                pass
                    await resp.write_eof()
                status = up.status
        except Exception as e:
            # A client that disconnects mid-stream still tells us something --
            # how far it got before giving up. Record it rather than dropping
            # it, then re-raise into a 502.
            status = status or 499
            resp = resp or web.json_response({"error": str(e)}, status=502)
        finally:
            _rec.record(key, turn, _capture(
                key, turn, messages, body, meta, t0, first,
                usage_in, usage_out, status, stream), meta["prompt"])
    return resp


def _capture(key, turn, messages, body, meta, t0, first, usage_in, usage_out,
             status, stream) -> Capture:
    return Capture(
        t=round(time.time() - _rec.t0, 3), session=key, turn=turn,
        n_messages=len(messages), prompt_chars=len(meta["prompt"]),
        system_chars=meta["system_chars"], new_chars=meta["new_chars"],
        prompt_tokens=usage_in, completion_tokens=usage_out,
        max_tokens=body.get("max_tokens"),
        ttft_ms=round((first - t0) * 1000, 1) if first else None,
        total_ms=round((time.time() - t0) * 1000, 1),
        think_ms=round(meta["think_ms"], 1) if meta["think_ms"] else None,
        status=status, model=body.get("model", ""), stream=stream,
    )


def make_app() -> web.Application:
    app = web.Application(client_max_size=1024 ** 3, middlewares=[_auth])
    app.router.add_post("/v1/chat/completions", _chat)
    app.router.add_get("/v1/models", _models)
    app.router.add_get("/health", _health)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0",
                port=int(os.environ.get("PROXY_PORT", "8000")))
