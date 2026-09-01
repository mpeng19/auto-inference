"""Correctness canaries: did the serving change alter what the model says?

Once serving code is editable, "goodput improved" and "we broke something"
look identical from the outside. An optimizer maximising goodput can win by
truncating outputs, dropping requests, or quietly changing sampling. These
canaries make that visible.

The subtlety: **identical output is not guaranteed even for identical code.**
Greedy decoding is not bitwise deterministic across different batch
compositions — reduction order changes, and a near-tie between two tokens can
flip. So an exact-match test would fail constantly and teach us to ignore it.

Instead we measure the *baseline* divergence first (same config, two runs) and
judge every other config against that floor. A config that diverges much more
than the floor is suspicious; one that diverges the same amount is normal.
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


def compare(a: dict[str, str], b: dict[str, str]) -> dict:
    """Compare two canary result sets. Returns divergence detail, not a verdict.

    `first_divergence` is the character index where the two outputs part, which
    localises whether a change nudged one token late in a long generation or
    derailed the response from the start.
    """
    keys = sorted(set(a) | set(b))
    per_key, exact = {}, 0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            per_key[k] = {"status": "missing"}
            continue
        if x == y:
            exact += 1
            per_key[k] = {"status": "identical", "len": len(x)}
            continue
        i = next((j for j, (cx, cy) in enumerate(zip(x, y)) if cx != cy),
                 min(len(x), len(y)))
        per_key[k] = {
            "status": "diverged",
            "first_divergence": i,
            "frac_identical": round(i / max(len(x), len(y), 1), 3),
            "len_a": len(x), "len_b": len(y),
            "a_tail": x[i:i + 60], "b_tail": y[i:i + 60],
        }
    return {
        "n": len(keys),
        "n_identical": exact,
        "exact_match_rate": round(exact / len(keys), 3) if keys else None,
        "per_canary": per_key,
    }


def verdict(observed: dict, floor: dict | None) -> str:
    """Judge a comparison against the measured same-config baseline.

    Without a floor we cannot say anything: we would be calling ordinary
    batching non-determinism a correctness bug.
    """
    if floor is None:
        return "no baseline — run the same config twice first to establish the floor"
    o, f = observed["exact_match_rate"], floor["exact_match_rate"]
    if o is None or f is None:
        return "insufficient data"
    if o >= f:
        return "OK — diverges no more than the same-config baseline"
    if o >= f - 0.2:
        return "MARGINAL — slightly more divergence than baseline; re-run before trusting"
    return "SUSPECT — materially more divergence than baseline; treat goodput gains as invalid"


async def run(base_url: str, model: str) -> dict:
    """Fire every canary at an otherwise idle server and digest the outputs.

    Idle on purpose: batch composition perturbs greedy decoding, so running
    these under load would measure the load, not the code. That also means the
    gate is weaker than it looks -- divergence introduced only under
    concurrency will not show up here. It is a cheap floor, not a proof.
    """
    from .server import complete
    outs = {}
    for name, prompt, max_tokens in CANARIES:
        outs[name] = await complete(base_url, model, prompt, max_tokens)
    digests = {k: digest(v) for k, v in outs.items()}
    return {"outputs": outs, "digests": digests,
            "summary": f"{len(digests)}/{len(CANARIES)} returned"}
