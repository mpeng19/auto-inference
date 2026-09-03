"""Filler text for TraceLab replay.

TraceLab is sanitised: it carries token counts and prefix structure but no
message text, so the replay needs plausible prose to fill each turn to its
recorded length. `_pad_to` cycles a reshuffled deck of English paragraphs so
nothing repeats back to back -- identical repeated spans would hand the prefix
cache an unrealistically easy time and read as machine-generated. The system
prompt is the shared prefix real deployments put in front of every turn.
"""
from __future__ import annotations

import random

CHARS_PER_TOKEN = 4.0

SYSTEM_PROMPT = (
    "You are an experienced code reviewer. Point out correctness bugs first, "
    "then performance, then style. Quote the specific line you mean. Do not "
    "restate what the code does unless it is unclear.")

_PROSE = [
    "The team reviewed the rollout plan and agreed the migration should proceed in stages.",
    "Latency rose sharply during the evening peak, though error rates stayed flat.",
    "Several customers reported that the export job silently produced empty files.",
    "We considered sharding by tenant, but the largest tenant alone exceeds one node.",
    "The postmortem identified three contributing factors and one root cause.",
    "Adoption has been slower than forecast, mostly in the self-serve segment.",
    "Throughput improved after the batch size was raised, at the cost of tail latency.",
    "The new schema removes two joins from the hot path and adds a nullable column.",
    "Support volume doubled in the week following the pricing change.",
    "Nobody could reproduce the issue locally until we matched the production timezone.",
]


def _pad_to(text: str, target_tokens: int, rng: random.Random,
            pool: list[str] | None = None, sep: str = " ") -> str:
    """Extend `text` with plausible material until it reaches ~target_tokens."""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    pool = pool if pool is not None else _PROSE
    parts = [text] if text else []
    size = len(text)
    # Cycle a reshuffled deck rather than sampling with replacement. Immediate
    # repeats read as machine-generated, and identical repeated spans would also
    # give the prefix cache an unrealistically easy time.
    deck: list[str] = []
    while size < target_chars:
        if not deck:
            deck = list(pool)
            rng.shuffle(deck)
        chunk = deck.pop()
        parts.append(chunk)
        size += len(chunk) + len(sep)
    return sep.join(parts)[:max(target_chars, len(text))]
