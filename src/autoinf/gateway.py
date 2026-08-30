"""Recording gateway: an OpenAI-compatible endpoint that logs request shape.

Purpose is trace capture, not serving. Point a real agent (OpenHands, Claude
Code, anything speaking the OpenAI API) at this endpoint, let it do real work,
and it records the *shape* of the traffic that work produced: how long each
prompt was, how the conversation grew turn over turn, how long the client
thought between turns, how much of each prompt was a repeat of the last one.

Why capture rather than synthesise. Our workloads are chat-shaped: ~500 tokens
in, ~235 out, decode-bound. A coding agent's turn is 8k-90k tokens in and
~99% prefill-bound -- 183x to 768x the work, on the opposite side of the
roofline. The *shape* of that growth (a stable prefix plus an appended tool
result, over and over) is the part that is hard to invent convincingly and the
part that decides whether prefix caching pays off.

The gateway does not shape the load. It streams responses through untouched and
records metadata alongside. Replay happens later, deterministically, from the
recorded trace -- so the agent is never in the measurement loop.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Capture:
    """One request as it passed through the gateway."""
    t: float                    # seconds since gateway start
    session: str                # groups turns of one conversation
    turn: int
    n_messages: int
    prompt_chars: int
    system_chars: int
    new_chars: int              # chars not present in the previous turn
    prompt_tokens: int | None
    completion_tokens: int | None
    max_tokens: int | None
    ttft_ms: float | None
    total_ms: float
    think_ms: float | None      # gap since this session's previous turn ended
    status: int
    model: str = ""
    stream: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def session_key(messages: list[dict]) -> str:
    """Group turns of the same conversation.

    Keyed on the opening exchange, which is stable as the conversation grows.
    A hash of the *whole* prompt would make every turn its own session, and a
    client-supplied id does not exist in the OpenAI API.
    """
    head = ""
    for m in messages:
        if m.get("role") in ("system", "user"):
            head += str(m.get("content", ""))[:400]
            if m.get("role") == "user":
                break
    return hashlib.sha256(head.encode()).hexdigest()[:12]


class Recorder:
    """Accumulates captures and tracks per-session state."""

    def __init__(self, path: str):
        self.path = path
        self.t0 = time.time()
        # session -> (turn index, previous prompt text, when the last turn ended)
        self._state: dict[str, tuple[int, str, float]] = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def observe(self, messages: list[dict], body: dict) -> tuple[str, int, dict]:
        key = session_key(messages)
        turn, prev_prompt, prev_end = self._state.get(key, (0, "", 0.0))
        prompt = "".join(str(m.get("content", "")) for m in messages)
        system = "".join(str(m.get("content", "")) for m in messages
                         if m.get("role") == "system")
        # How much of this prompt is genuinely new. In an agent loop this is
        # small relative to the whole, which is exactly why prefix caching
        # matters so much for this traffic.
        new = max(0, len(prompt) - len(prev_prompt))
        think = (time.time() - prev_end) * 1000.0 if prev_end else None
        return key, turn, {"prompt": prompt, "system_chars": len(system),
                           "new_chars": new, "think_ms": think}

    def record(self, key: str, turn: int, cap: Capture, prompt: str) -> None:
        self._state[key] = (turn + 1, prompt, time.time())
        with open(self.path, "a") as f:
            f.write(cap.to_json() + "\n")

    def load(self) -> list[Capture]:
        if not os.path.exists(self.path):
            return []
        return [Capture(**json.loads(l)) for l in
                open(self.path).read().splitlines() if l.strip()]


def summarize_captures(caps: list[Capture]) -> dict:
    """What did the agent's traffic actually look like?"""
    if not caps:
        return {"n": 0}

    def q(xs, p, nd=1):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[min(len(xs) - 1, int(len(xs) * p))], nd) if xs else None

    sessions: dict[str, list[Capture]] = {}
    for c in caps:
        sessions.setdefault(c.session, []).append(c)

    by_depth: dict[int, list[Capture]] = {}
    for c in caps:
        by_depth.setdefault(min(c.turn, 10), []).append(c)

    pt = [c.prompt_tokens for c in caps if c.prompt_tokens]
    reuse = [1 - c.new_chars / c.prompt_chars for c in caps
             if c.prompt_chars and c.turn > 0]

    return {
        "n_requests": len(caps),
        "n_sessions": len(sessions),
        "turns_per_session_mean": round(len(caps) / len(sessions), 2),
        "turns_per_session_max": max(len(v) for v in sessions.values()),
        "prompt_tokens": {"p50": q(pt, 0.5), "p95": q(pt, 0.95),
                          "max": max(pt) if pt else None},
        "completion_tokens": {
            "p50": q([c.completion_tokens for c in caps], 0.5),
            "p95": q([c.completion_tokens for c in caps], 0.95)},
        "think_ms": {"p50": q([c.think_ms for c in caps], 0.5),
                     "p95": q([c.think_ms for c in caps], 0.95)},
        "ttft_ms": {"p50": q([c.ttft_ms for c in caps], 0.5),
                    "p99": q([c.ttft_ms for c in caps], 0.99)},
        # The headline for agentic traffic: what fraction of each prompt is a
        # verbatim repeat of the previous turn. High reuse is what makes the
        # prefix cache decisive rather than incidental.
        # Rounded to 3 dp on purpose: at one decimal, 0.90 and 0.99 look the
        # same, and that is the difference between the prefix cache being
        # useful and being decisive.
        "prefix_reuse_frac": {"p50": q(reuse, 0.5, 3), "p95": q(reuse, 0.95, 3)},
        "by_turn_depth": {
            str(d): {"n": len(v),
                     "prompt_tokens_p50": q([c.prompt_tokens for c in v], 0.5),
                     "new_chars_p50": q([float(c.new_chars) for c in v], 0.5),
                     "ttft_p50_ms": q([c.ttft_ms for c in v], 0.5)}
            for d, v in sorted(by_depth.items())},
    }
