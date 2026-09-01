"""Multi-turn conversations: the prefix must actually grow, turns must order."""
import asyncio
import json

import pytest
from aiohttp import web

from autoinf.config import WorkloadConfig
from autoinf.loadgen import run_sessions, summarize_turns
from autoinf.workload import build_sessions

SEEN: list[dict] = []


async def _chat(request):
    body = await request.json()
    SEEN.append({"n_messages": len(body["messages"]),
                 "chars": sum(len(m["content"]) for m in body["messages"])})
    n = min(body.get("max_tokens", 8), 12)
    r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await r.prepare(request)
    for _ in range(n):
        await r.write(b'data: {"choices":[{"delta":{"content":"reply "}}]}\n\n')
    await r.write(("data: " + json.dumps(
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": n}}
    ) + "\n\n").encode())
    await r.write(b"data: [DONE]\n\n")
    await r.write_eof()
    return r


@pytest.fixture
async def server(aiohttp_server):
    SEEN.clear()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _chat)
    return await aiohttp_server(app)


def _cfg(**kw):
    base = dict(name="mt", multi_turn=True, request_rate=40.0, n_requests=12,
                turns_mu=3.0, think_mu=-3.0, think_sigma=0.1,
                shared_prefix_len=60, seed=1)
    base.update(kw)
    return WorkloadConfig(**base)


def test_sessions_are_deterministic():
    a, b = build_sessions(_cfg()), build_sessions(_cfg())
    assert a.digest() == b.digest()
    assert build_sessions(_cfg(seed=2)).digest() != a.digest()


def test_every_session_shares_one_system_prompt():
    tr = build_sessions(_cfg())
    systems = {s.system for s in tr.sessions}
    assert len(systems) == 1, "a shared preamble is what the prefix cache exploits"


@pytest.mark.asyncio
async def test_prompt_grows_with_turn_depth(server):
    url = f"http://{server.host}:{server.port}"
    tr = build_sessions(_cfg(n_requests=8, turns_mu=4.0))
    res = await run_sessions(tr, url, "m")
    assert res and all(r.ok for r in res), [r.error for r in res if not r.ok]

    # Within any session, history must strictly increase turn over turn.
    by_sess: dict[int, list] = {}
    for r in res:
        by_sess.setdefault(r.session, []).append(r)
    grew = 0
    for rs in by_sess.values():
        rs.sort(key=lambda r: r.turn)
        hist = [r.history_tokens for r in rs]
        assert hist == sorted(hist), hist
        if len(hist) > 1:
            assert hist[-1] > hist[0]
            grew += 1
    assert grew >= 1, "no multi-turn session in the trace"


@pytest.mark.asyncio
async def test_turns_are_sequential_within_a_session(server):
    url = f"http://{server.host}:{server.port}"
    res = await run_sessions(build_sessions(_cfg(n_requests=6, turns_mu=4.0)), url, "m")
    by_sess: dict[int, list] = {}
    for r in res:
        by_sess.setdefault(r.session, []).append(r)
    for rs in by_sess.values():
        rs.sort(key=lambda r: r.turn)
        for a, b in zip(rs, rs[1:]):
            # A turn cannot start before the previous one finished.
            assert b.dispatched_s >= (a.end_s or 0), (a.turn, a.end_s, b.dispatched_s)


@pytest.mark.asyncio
async def test_message_count_grows_by_two_per_turn(server):
    """user + assistant appended each turn, on top of one system message."""
    url = f"http://{server.host}:{server.port}"
    await run_sessions(build_sessions(_cfg(n_requests=1, turns_mu=6.0)), url, "m")
    counts = sorted(s["n_messages"] for s in SEEN)
    assert counts[0] == 2                       # system + first user
    if len(counts) > 1:
        assert counts[1] == 4                   # + assistant + user
        assert all((c - 2) % 2 == 0 for c in counts)


@pytest.mark.asyncio
async def test_summarize_turns_reports_growth(server):
    url = f"http://{server.host}:{server.port}"
    res = await run_sessions(build_sessions(_cfg(n_requests=10, turns_mu=4.0)), url, "m")
    s = summarize_turns(res)
    rows = s["by_turn_depth"]
    assert "0" in rows
    depths = sorted(int(k) for k in rows)
    if len(depths) > 1:
        first, last = rows[str(depths[0])], rows[str(depths[-1])]
        assert last["mean_history_tokens"] > first["mean_history_tokens"]


@pytest.mark.asyncio
async def test_concurrent_users_do_not_share_history(server):
    """Sessions come from a shared pool; two users drawing the same one must
    not interleave conversations, and history must reset between loops."""
    from autoinf.loadgen import run_concurrent_users
    url = f"http://{server.host}:{server.port}"
    tr = build_sessions(_cfg(n_requests=3, turns_mu=3.0))
    pool = tr.sessions
    # Every user gets the SAME session object -- the worst case for sharing.
    res = await run_concurrent_users(lambda uid, n: pool[0], url, "m",
                                     n_users=4, duration_s=2.0)
    assert res, "no requests issued"
    by_user: dict[int, list] = {}
    for r in res:
        by_user.setdefault(r.session, []).append(r)
    assert len(by_user) >= 2, "expected several users to have issued requests"
    for rs in by_user.values():
        rs.sort(key=lambda r: r.dispatched_s)
        # Turn indices must restart at 0 for each new conversation loop,
        # never climb without bound.
        assert min(r.turn for r in rs) == 0
        assert max(r.turn for r in rs) < 20, "history leaked across loops"
    # History size must stay bounded rather than growing across conversations.
    assert max(r.history_tokens for r in res) < 5000, \
        "history grew without bound -- shared or never reset"
