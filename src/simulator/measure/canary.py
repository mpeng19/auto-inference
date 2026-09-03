"""Correctness canaries: did the serving change alter what the model says?

Once serving code is editable, "goodput improved" and "we broke something"
look identical from the outside. An optimizer maximising goodput can win by
truncating outputs, dropping requests, or quietly changing sampling. These
canaries make that visible: six fixed prompts, run on the idle server before
load, with the outputs and their digests kept in the run record.

They are evidence, not a gate. Greedy decoding is not bitwise deterministic
across batch compositions -- reduction order changes, and a near-tie between
two tokens can flip -- so an exact-match test against stock would fail
constantly. The gates are `quality` (accuracy) and `equivalence` (logits).
"""
from __future__ import annotations

import hashlib

# Fixed, deterministic prompts spanning the behaviours a broken scheduler is
# most likely to damage: long context, multi-step reasoning, exact recall,
# instruction following, and a long generation.
CANARIES: tuple[tuple[str, str, int], ...] = (
    ("arith", "What is 17 * 24? Reply with only the number.", 16),
    ("recall", "List the first 8 prime numbers, comma separated, nothing else.", 32),
    ("instruct", "Reply with exactly the word BANANA and nothing else.", 8),
    ("short_gen", "Write one sentence about scheduling latency.", 48),
    ("long_gen", "Count from 1 to 40, separated by spaces. Only the numbers.", 128),
    ("long_ctx", ("Remember this token: ZEBRA7. " + "filler text here. " * 400
                  + " What token were you asked to remember?"), 24),
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def run(base_url: str, model: str) -> dict:
    """Fire every canary at an otherwise idle server and digest the outputs.

    Idle on purpose: batch composition perturbs greedy decoding, so running
    these under load would measure the load, not the code. That also means the
    record is weaker than it looks -- divergence introduced only under
    concurrency will not show up here. It is a cheap floor, not a proof.
    """
    from .server import complete
    outs = {}
    for name, prompt, max_tokens in CANARIES:
        outs[name] = await complete(base_url, model, prompt, max_tokens)
    digests = {k: digest(v) for k, v in outs.items()}
    return {"outputs": outs, "digests": digests,
            "summary": f"{len(digests)}/{len(CANARIES)} returned"}
