"""Synthetic multi-turn session generation, and the suite calibration guard.

Retired from the product 2026-09-01: the settled method replays real
coding-agent traffic rescaled to the marketplace mix, so there is nothing
left for a synthetic conversation to stand in for. Our synthetic sessions
were also wrong by ~250x on input length (HANDOFF SS2), which is what sent
us to TraceLab in the first place.
"""
from __future__ import annotations

import random
from research.autoinf.workload_config import WorkloadConfig
from research.autoinf.eval_suite import SessionTrace
from simulator.workload import prompts as _prompts
from simulator.workload.sessions import Session, Turn, _filler, _lognormal_int


CALIBRATED_FOR = ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "H100", 1)


def check_calibration(cfg) -> str | None:
    """Warn when the suite is run against hardware it was not calibrated for."""
    got = (cfg.model, cfg.gpu, cfg.n_gpu)
    if got == CALIBRATED_FOR:
        return None
    return (f"suite rates were calibrated for {CALIBRATED_FOR[0].split('/')[-1]} "
            f"on {CALIBRATED_FOR[2]}x{CALIBRATED_FOR[1]}, but this run uses "
            f"{got[0].split('/')[-1]} on {got[2]}x{got[1]}. Absolute goodput is "
            f"not comparable across the two, and the load levels may sit in a "
            f"different regime entirely (the small dense models are "
            f"prefill-bound where the 30B MoE is decode-bound). Re-derive with "
            f"`make staircase` before trusting these numbers.")


_FOLLOWUPS = [
    "Can you expand on that?",
    "Why does that work?",
    "What would break if I did the opposite?",
    "Show me a concrete example.",
    "Is there a simpler way?",
    "What are the failure modes?",
    "How would I test that?",
    "Does that still hold at scale?",
    "What would you do differently in production?",
    "Summarise that as bullet points.",
]


def build_sessions(cfg: WorkloadConfig) -> SessionTrace:
    """Generate conversations. Deterministic given the seed.

    Only the *structure* is fixed here -- who arrives when, how many turns, what
    they say, how long they think. The prompt actually sent at turn k is
    assembled at run time from the replies the server gave, because that is what
    a chat client does and it is what makes the prefix grow the way a real one
    does.
    """
    rng = random.Random(cfg.seed)
    system = _prompts.SYSTEM_PROMPTS[0][1]
    if cfg.shared_prefix_len:
        system = _prompts._pad_to(system, cfg.shared_prefix_len,
                                  random.Random(cfg.seed * 7717))

    # Sessions arrive as a Poisson process at `request_rate`.
    n_sessions = cfg.n_requests or 200
    duration = cfg.duration_s
    times, t = [], 0.0
    for _ in range(n_sessions):
        t += rng.expovariate(max(cfg.request_rate, 1e-9))
        if duration is not None and t > duration:
            break
        times.append(t)

    mix = cfg.category_mix or _prompts.ALL_CATEGORIES
    sessions = []
    for i, at in enumerate(times):
        k = max(1, min(cfg.turns_max, int(rng.expovariate(1.0 / cfg.turns_mu)) + 1))
        cat = _prompts.sample_category(rng, mix)

        turns = [Turn(_prompts.make_request(rng, cat.name,
                                            _lognormal_int(rng, cat.in_mu, cat.in_sigma,
                                                           cfg.input_len_cap)),
                      _lognormal_int(rng, cat.out_mu, cat.out_sigma, cfg.output_len_cap),
                      cat.name)]
        for _ in range(k - 1):
            # Follow-ups are short; the growth comes from accumulated history,
            # not from the user typing more.
            turns.append(Turn(rng.choice(_FOLLOWUPS),
                              _lognormal_int(rng, cat.out_mu, cat.out_sigma,
                                             cfg.output_len_cap),
                              cat.name))
        think = tuple(min(60.0, rng.lognormvariate(cfg.think_mu, cfg.think_sigma))
                      for _ in range(k))
        sessions.append(Session(i, at, system, tuple(turns), think))

    return SessionTrace(tuple(sessions), cfg)
