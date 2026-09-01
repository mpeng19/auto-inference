"""Replay real coding-agent traffic from TraceLab.

TraceLab (SyFI Lab, University of Washington, CC-BY-4.0) is a sanitised trace of
665,453 Claude Code and Codex invocations across 8,058 sessions, published
specifically for LLM serving research. https://tracelab.cs.washington.edu/

Why this replaces guessing. Our synthetic workloads were built from plausible
assumptions and are wrong by two orders of magnitude on the axis that matters:

                              ours    TraceLab
    median input tokens        537     132,092
    input:output ratio         2.3       291.4
    cache hit rate            0.96       0.956

Output length we happened to get right (233 vs 249), which is precisely why the
ratio came out so badly. The traffic is not chat with a bit of context -- it is
a ~130k-token context resent every turn with a ~1k increment, hundreds of times
per session. That is a different serving problem: overwhelmingly prefill,
overwhelmingly cached, and bounded by KV capacity rather than compute.

The trace carries no message text (sanitised), which is fine: we need the
*shape*, not the content. Token counts, prefix reuse and session structure are
reproduced exactly, and filler text is generated to match.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

REPO = "UW-SyFI/TraceLab"
REVISION = "v0.0.2"
ROUNDS = "data/v0.0.2/rounds/train.parquet"

# Attribution required by CC-BY-4.0.
CITATION = ("TraceLab (SyFI Lab, University of Washington), "
            "https://tracelab.cs.washington.edu/ , CC-BY-4.0")


@dataclass(frozen=True)
class TraceRound:
    """One LLM invocation from a real coding-agent session."""
    session: str
    index: int
    input_tokens: int
    cached_tokens: int          # provider-reported prefix reuse
    new_tokens: int             # non-cached input appended this round
    output_tokens: int
    model: str
    provider: str

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / max(1, self.input_tokens)


def download_rounds(local_dir: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO, repo_type="dataset",
                           filename=ROUNDS, revision=REVISION,
                           local_dir=local_dir)


def load_sessions(path: str | None = None, min_rounds: int = 4,
                  max_rounds: int = 60, max_sessions: int | None = 400,
                  seed: int = 0, provider: str | None = None) -> list[list[TraceRound]]:
    """Load sessions, longest-context first, with the documented correction.

    TraceLab warns that a small fraction of adjacent rounds report
    `prefix_tokens` larger than the reconstructable previous context, and
    advises capping the reusable prefix at
    `previous.input_tokens_total + previous.output_tokens` for session-local
    KV replay. Applied here -- without it the replay would claim cache hits the
    server could not possibly serve, and inflate the very number we are trying
    to measure.
    """
    import pandas as pd

    df = pd.read_parquet(path or download_rounds())
    if provider:
        df = df[df.provider == provider]
    df = df.sort_values(["session_id", "round_index"])

    out: list[list[TraceRound]] = []
    for sid, g in df.groupby("session_id", sort=False):
        if not (min_rounds <= len(g) <= max_rounds):
            continue
        rounds: list[TraceRound] = []
        prev_ctx = 0
        for i, (_, r) in enumerate(g.iterrows()):
            inp = int(r.input_tokens_total)
            # Cap reusable prefix at what the previous round could have left
            # behind (TraceLab issue #22).
            cached = min(int(r.prefix_tokens), prev_ctx, inp)
            rounds.append(TraceRound(
                session=str(sid), index=i, input_tokens=inp,
                cached_tokens=max(0, cached),
                new_tokens=max(0, inp - max(0, cached)),
                output_tokens=int(r.output_tokens),
                model=str(r.model), provider=str(r.provider)))
            prev_ctx = inp + int(r.output_tokens)
        out.append(rounds)

    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:max_sessions] if max_sessions else out


def describe(sessions: list[list[TraceRound]]) -> dict:
    """Summarise a replay set, so a scaled-down subset is not silently
    unrepresentative of the traffic it claims to model."""
    rounds = [r for s in sessions for r in s]
    if not rounds:
        return {"n_sessions": 0}

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    tin = sum(r.input_tokens for r in rounds)
    tc = sum(r.cached_tokens for r in rounds)
    to = sum(r.output_tokens for r in rounds)
    return {
        "source": CITATION,
        "n_sessions": len(sessions),
        "n_rounds": len(rounds),
        "rounds_per_session_p50": q([len(s) for s in sessions], 0.5),
        "input_tokens_p50": q([r.input_tokens for r in rounds], 0.5),
        "input_tokens_p90": q([r.input_tokens for r in rounds], 0.9),
        "new_tokens_p50": q([r.new_tokens for r in rounds], 0.5),
        "output_tokens_p50": q([r.output_tokens for r in rounds], 0.5),
        "aggregate_hit_rate": round(tc / max(1, tin), 4),
        "input_output_ratio": round(tin / max(1, to), 1),
        "total_input_tokens": tin,
        "total_output_tokens": to,
    }


def to_sessions(traces: list[list[TraceRound]], system_tokens: int = 400,
                think_mu: float = 0.3, think_sigma: float = 0.6,
                seed: int = 0):
    """Convert trace rounds into replayable Sessions.

    The trace is sanitised, so there is no message text -- only the shape. That
    is what we need: token counts, prefix reuse and turn structure drive cache
    and KV behaviour, not the words. Filler is generated to match each round's
    `new_tokens`, and because the client resends the full conversation each
    turn, the server sees the same growing-prefix pattern the real agent
    produced.

    Returns `simulator.workload.sessions.Session` objects so the existing multi-turn
    runner replays them unchanged.
    """
    import random as _r

    from . import prompts as _p
    from .sessions import Session, Turn

    rng = _r.Random(seed)
    system = _p._pad_to(_p.SYSTEM_PROMPTS[2][1], system_tokens, _r.Random(seed * 31))

    out = []
    for i, rounds in enumerate(traces):
        turns, think = [], []
        for r in rounds:
            # Only the *new* tokens are authored each turn; the rest of the
            # prompt is the conversation the client resends, which the runner
            # rebuilds from prior replies.
            turns.append(Turn(_p._pad_to("", max(1, r.new_tokens), rng),
                              max(1, r.output_tokens), "agentic"))
            think.append(min(60.0, rng.lognormvariate(think_mu, think_sigma)))
        out.append(Session(idx=i, arrival_s=0.0, system=system,
                           turns=tuple(turns), think_s=tuple(think)))
    return out


def scale_sessions(sessions: list[list[TraceRound]], factor: float,
                   out_factor: float | None = None
                   ) -> list[list[TraceRound]]:
    """Shrink every context by `factor`, preserving cache structure.

    Needed because the real traffic will not fit. At a 132k median context and
    144 KiB/token of KV, one conversation is ~19GB -- an L40S holds two, which
    is too few to study batching at all. Scaling keeps the *ratios* that drive
    serving behaviour (hit rate, input:output, increment size) while fitting
    the memory available.

    This is a compromise and should be recorded as one: a scaled replay tests
    the same *structure* at a different absolute scale, and results from it do
    not automatically transfer to full-size contexts, where KV pressure and
    eviction dominate.

    `out_factor` scales output independently. Uniform scaling preserves
    TraceLab's 291:1 input:output ratio, but the marketplace this model
    actually serves runs at **9.9:1** -- see `scale_to_market`.
    """
    of = factor if out_factor is None else out_factor
    out = []
    for s in sessions:
        rs = []
        for r in s:
            inp = max(16, int(r.input_tokens * factor))
            cached = min(int(r.cached_tokens * factor), inp)
            rs.append(TraceRound(
                session=r.session, index=r.index, input_tokens=inp,
                cached_tokens=cached, new_tokens=max(1, inp - cached),
                output_tokens=max(1, int(r.output_tokens * of)),
                model=r.model, provider=r.provider))
        out.append(rs)
    return out


# What OpenRouter's traffic for qwen/qwen3.8-27b actually looks like, from 17
# days of `model_chart` totals (`scripts/market_pull.py`). TraceLab is Claude
# Code traffic specifically; the marketplace mixes it with agents that generate
# far more output (pi, Hermes Agent, LangChain, DeepSeek Harness).
#
#                    TraceLab   marketplace
#   input tokens/req  132,092        20,583
#   output tokens/req     454         2,076
#   input:output        291:1         9.9:1
#
# The ratio matters more than either level: under TraceLab's mix only 12% of
# modelled serving cost falls on output tokens, against 70-81% under the real
# one, which changes what is worth optimising.
MARKET_IN_PER_REQ = 20_583
MARKET_OUT_PER_REQ = 2_076


def scale_to_market(sessions: list[list[TraceRound]],
                    in_tokens: int = MARKET_IN_PER_REQ,
                    out_tokens: int = MARKET_OUT_PER_REQ
                    ) -> tuple[list[list[TraceRound]], dict]:
    """Rescale a TraceLab pool to the marketplace's input and output sizes.

    Factors are derived from the pool that was actually loaded rather than
    hard-coded, so this stays correct if the sampling parameters change.
    Prefix structure -- which turn reuses what -- is untouched; only the
    absolute token counts move.
    """
    rounds = [r for s in sessions for r in s]
    if not rounds:
        return sessions, {"scaled": False}
    mean_in = sum(r.input_tokens for r in rounds) / len(rounds)
    mean_out = sum(r.output_tokens for r in rounds) / len(rounds)
    f_in = in_tokens / mean_in
    f_out = out_tokens / mean_out
    scaled = scale_sessions(sessions, f_in, f_out)
    return scaled, {"scaled": True, "in_factor": round(f_in, 5),
                    "out_factor": round(f_out, 5),
                    "source_in": round(mean_in), "source_out": round(mean_out),
                    "target_in": in_tokens, "target_out": out_tokens}
