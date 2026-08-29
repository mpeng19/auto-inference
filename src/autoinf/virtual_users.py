"""LLM-driven virtual users: a realism track alongside the deterministic suite.

Two tracks, two jobs.

The **deterministic suite** exists for A/B resolution. Fixed traces, fixed
seeds, 0.03% run-to-run variance on goodput — that is what lets a 1% scheduler
improvement be believed.

This is the **realism track**. It is deliberately non-deterministic, because it
answers a different question: does our synthetic traffic resemble what people
actually do? Aggregate metrics converge by the law of large numbers, so means
are trustworthy over a long run. Tails converge far more slowly — p99 TTFT is
driven by rare coincidences of long prompts arriving together — so use this
track for aggregates and realism, and the deterministic suite for tail claims.

What this captures that fixed traces cannot:

  * **Multi-turn.** A user's next message depends on the reply. Conversation
    history accumulates, so the shared prefix *grows* — which is the pattern
    prefix caching exists for and which our single-turn suite never exercises.
  * **Think time.** Real sessions have human-scale gaps between turns, so
    arrivals are correlated rather than independent.
  * **Abandonment.** People close the tab mid-answer. The server has already
    committed GPU work to tokens nobody will read, and must reclaim the KV
    blocks. We have never tested that path.
  * **Session shape.** Users arrive, converse, and leave. That is a different
    arrival process from any renewal process, however cleverly parameterised.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Persona:
    """How one kind of user behaves. Shapes traffic, not just wording."""
    name: str
    brief: str                  # steers the user-simulator LLM
    weight: float
    think_mu: float             # lognormal seconds between turns
    think_sigma: float
    turns_mu: float             # mean turns per session
    abandon_p: float            # chance of disconnecting mid-response
    max_tokens: int


PERSONAS: tuple[Persona, ...] = (
    Persona("quick_asker", "Asks short factual questions and moves on. Rarely "
            "follows up. Impatient.", 0.30, 1.6, 0.7, 1.6, 0.18, 200),
    Persona("debugger", "A developer working through a bug. Pastes code, reads "
            "the answer, and asks pointed follow-ups about specific lines.",
            0.22, 2.6, 0.6, 4.5, 0.06, 400),
    Persona("researcher", "Explores a topic in depth over many turns, each "
            "building on the last. Asks for sources and pushes back.",
            0.16, 3.1, 0.5, 7.0, 0.04, 500),
    Persona("summarizer", "Pastes long documents and asks for summaries or "
            "extraction. Few turns, very long inputs.", 0.14, 2.2, 0.6, 2.0,
            0.08, 250),
    Persona("chatter", "Casual open-ended conversation, tangents, jokes. Long "
            "sessions, short messages.", 0.12, 1.9, 0.8, 8.0, 0.10, 300),
    Persona("power_user", "Rapid-fire requests with little think time. Often "
            "cancels and re-asks when the answer starts wrong.", 0.06, 0.6, 0.6,
            6.0, 0.30, 600),
)


@dataclass
class Turn:
    session: int
    turn: int
    persona: str
    dispatched_s: float
    first_token_s: float | None
    end_s: float | None
    prompt_chars: int
    history_chars: int          # shared prefix carried into this turn
    output_tokens: int
    prompt_tokens: int | None
    abandoned: bool
    ok: bool
    error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_s is None:
            return None
        return (self.first_token_s - self.dispatched_s) * 1000.0


class UserSim:
    """Generates each user's next message.

    `claude` uses a small fast model as the user simulator; it is the whole
    point of this track, since the messages are genuinely authored rather than
    slotted into our templates. `template` is a free fallback so the harness
    runs without an API key — degraded, but not broken.
    """

    def __init__(self, backend: str = "claude", model: str = "claude-haiku-4-5-20251001"):
        self.backend = backend
        self.model = model
        self._client = None
        if backend == "claude":
            try:
                import anthropic
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError("ANTHROPIC_API_KEY not set")
                self._client = anthropic.AsyncAnthropic(api_key=key)
            except Exception as e:
                print(f"user-sim falling back to templates: {e}", flush=True)
                self.backend = "template"

    async def next_message(self, p: Persona, history: list[tuple[str, str]],
                           rng: random.Random) -> str:
        if self.backend == "template" or self._client is None:
            from .prompts import ALL_CATEGORIES, make_request, sample_category
            if not history:
                cat = sample_category(rng, ALL_CATEGORIES)
                return make_request(rng, cat.name, int(2.718 ** cat.in_mu))
            return rng.choice([
                "Can you expand on that?", "Why does that work?",
                "What would break if I did the opposite?",
                "Show me a concrete example.", "Is there a simpler way?",
            ])

        if not history:
            instr = (f"You are simulating a user of an AI assistant. Persona: {p.brief}\n\n"
                     "Write ONLY the user's opening message. No preamble, no quotes, "
                     "no explanation. Make it specific and realistic.")
            msgs = [{"role": "user", "content": instr}]
        else:
            convo = "\n\n".join(f"USER: {u[:600]}\nASSISTANT: {a[:900]}"
                                for u, a in history[-3:])
            instr = (f"You are simulating a user of an AI assistant. Persona: {p.brief}\n\n"
                     f"Conversation so far:\n\n{convo}\n\n"
                     "Write ONLY the user's next message. No preamble, no quotes. "
                     "It must follow naturally from the assistant's last reply.")
            msgs = [{"role": "user", "content": instr}]

        r = await self._client.messages.create(
            model=self.model, max_tokens=300, messages=msgs)
        return "".join(b.text for b in r.content if b.type == "text").strip()


async def _one_session(session_idx: int, persona: Persona, sim: UserSim,
                       base_url: str, served_model: str, rng: random.Random,
                       t0: float, out: list[Turn], deadline: float) -> None:
    import aiohttp

    history: list[tuple[str, str]] = []
    n_turns = max(1, int(rng.expovariate(1.0 / persona.turns_mu)))

    async with aiohttp.ClientSession() as s:
        for turn in range(n_turns):
            if time.perf_counter() - t0 > deadline:
                return
            try:
                msg = await sim.next_message(persona, history, rng)
            except Exception as e:
                print(f"user-sim error: {type(e).__name__}: {e}", flush=True)
                return

            # Full conversation is resent each turn -- exactly how a chat client
            # works, and the reason the shared prefix grows turn over turn.
            messages = []
            for u, a in history:
                messages += [{"role": "user", "content": u},
                             {"role": "assistant", "content": a}]
            messages.append({"role": "user", "content": msg})
            hist_chars = sum(len(u) + len(a) for u, a in history)

            # Abandon partway through reading the reply, as people do.
            abandon = rng.random() < persona.abandon_p
            abandon_after = rng.randint(3, 40) if abandon else None

            dispatched = time.perf_counter() - t0
            first_tok, end_s, n_delta, usage_in, usage_out = None, None, 0, None, None
            reply_parts: list[str] = []
            err = None
            try:
                async with s.post(
                    base_url.rstrip("/") + "/v1/chat/completions",
                    json={"model": served_model, "messages": messages,
                          "max_tokens": persona.max_tokens, "temperature": 0.7,
                          "stream": True, "stream_options": {"include_usage": True}},
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as r:
                    if r.status != 200:
                        err = f"HTTP {r.status}"
                    else:
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
                                usage_in = ch["usage"].get("prompt_tokens")
                                usage_out = ch["usage"].get("completion_tokens")
                            for c in ch.get("choices", []):
                                piece = (c.get("delta") or {}).get("content")
                                if piece:
                                    if first_tok is None:
                                        first_tok = time.perf_counter() - t0
                                    n_delta += 1
                                    reply_parts.append(piece)
                            if abandon_after and n_delta >= abandon_after:
                                # Drop the connection mid-stream. The server has
                                # committed GPU work to tokens nobody reads and
                                # must reclaim the KV blocks.
                                break
                        end_s = time.perf_counter() - t0
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:160]

            out.append(Turn(
                session=session_idx, turn=turn, persona=persona.name,
                dispatched_s=dispatched, first_token_s=first_tok, end_s=end_s,
                prompt_chars=len(msg), history_chars=hist_chars,
                output_tokens=usage_out if usage_out is not None else n_delta,
                prompt_tokens=usage_in,
                abandoned=bool(abandon_after and n_delta >= (abandon_after or 0)),
                ok=err is None and first_tok is not None, error=err))

            if err or abandon_after:
                return
            history.append((msg, "".join(reply_parts)))
            await asyncio.sleep(min(60.0, rng.lognormvariate(
                persona.think_mu, persona.think_sigma)))


def pick_persona(rng: random.Random) -> Persona:
    r = rng.random() * sum(p.weight for p in PERSONAS)
    acc = 0.0
    for p in PERSONAS:
        acc += p.weight
        if r <= acc:
            return p
    return PERSONAS[-1]


async def run_virtual_users(base_url: str, served_model: str, n_users: int = 40,
                            duration_s: float = 300.0, arrival_rps: float = 2.0,
                            seed: int = 0, backend: str = "claude") -> list[Turn]:
    """Sessions arrive over time, converse, and leave."""
    sim = UserSim(backend=backend)
    rng = random.Random(seed)
    out: list[Turn] = []
    t0 = time.perf_counter()
    tasks = []

    for i in range(n_users):
        # Stagger arrivals; users do not all appear at t=0.
        delay = rng.expovariate(arrival_rps) if arrival_rps > 0 else 0.0
        await asyncio.sleep(min(delay, 5.0))
        if time.perf_counter() - t0 > duration_s:
            break
        tasks.append(asyncio.create_task(_one_session(
            i, pick_persona(rng), sim, base_url, served_model,
            random.Random(seed * 10000 + i), t0, out, duration_s)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return out


def summarize_sessions(turns: list[Turn]) -> dict:
    """Aggregate view. Per-persona and per-turn-depth, because that is where
    the multi-turn effect lives."""
    if not turns:
        return {"n_turns": 0}

    def q(xs, p):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[min(len(xs) - 1, int(len(xs) * p))], 1) if xs else None

    ok = [t for t in turns if t.ok]
    by_depth: dict[int, list[Turn]] = {}
    for t in ok:
        by_depth.setdefault(min(t.turn, 5), []).append(t)

    by_persona: dict[str, list[Turn]] = {}
    for t in ok:
        by_persona.setdefault(t.persona, []).append(t)

    return {
        "n_turns": len(turns),
        "n_sessions": len({t.session for t in turns}),
        "n_ok": len(ok),
        "n_abandoned": sum(1 for t in turns if t.abandoned),
        "abandon_rate": round(sum(1 for t in turns if t.abandoned) / len(turns), 3),
        "ttft_p50_ms": q([t.ttft_ms for t in ok], 0.5),
        "ttft_p99_ms": q([t.ttft_ms for t in ok], 0.99),
        # The headline for this track: does TTFT improve with conversation
        # depth? It should, because the shared prefix grows and the cache can
        # serve more of it. If it does not, prefix caching is not working on
        # realistic multi-turn traffic regardless of what the synthetic
        # prefix_heavy workload says.
        "by_turn_depth": {
            str(d): {
                "n": len(ts),
                "ttft_p50_ms": q([t.ttft_ms for t in ts], 0.5),
                "mean_history_chars": round(sum(t.history_chars for t in ts) / len(ts)),
                "mean_prompt_tokens": round(
                    sum(t.prompt_tokens or 0 for t in ts) / len(ts)),
            } for d, ts in sorted(by_depth.items())
        },
        "by_persona": {
            name: {
                "n": len(ts),
                "ttft_p50_ms": q([t.ttft_ms for t in ts], 0.5),
                "mean_out_tokens": round(sum(t.output_tokens for t in ts) / len(ts)),
            } for name, ts in sorted(by_persona.items())
        },
        "errors": {},
    }
