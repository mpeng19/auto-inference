"""WorkloadConfig and the original single-percentile SLO.

Retired 2026-09-01. The product replays market-scaled TraceLab only, so
the synthetic-trace knobs have no consumer; `SLO` was replaced by
`simulator.slo`, which lets TTFT and TPOT sit at different quantiles --
the market publishes a TTFT tail and only a TPOT middle (HANDOFF SS6d).
"""
from __future__ import annotations

import hashlib, json
from dataclasses import asdict, dataclass, field
from simulator.config import _digest


@dataclass(frozen=True)
class WorkloadConfig:
    """Defines the request trace. Same config + same seed = same trace, always.

    The arrival process matters as much as the length distributions. A serving
    system that looks fine under smooth Poisson load can fall over under bursts
    of the same *mean* rate, because queueing, admission control and batch
    formation all respond to variance rather than to the mean.
    """
    name: str = "sustained"

    # ── arrival process ──────────────────────────────────────────
    # poisson  : memoryless, the standard baseline
    # constant : deterministic gaps; zero arrival variance, a control
    # bursty   : on/off square wave at the same mean rate
    # ramp     : rate climbs linearly, to locate the saturation knee
    # spike    : flat, then a sudden step, to test overload + recovery
    arrival: str = "poisson"
    request_rate: float = 4.0           # base rate, requests/sec

    # Trace length. duration_s wins if both are set; with only n_requests, the
    # duration is derived from the mean rate.
    n_requests: int | None = 300
    duration_s: float | None = None

    # bursty: peak = request_rate * burst_factor during burst_on_s. The idle
    # gap is derived so the *mean* rate is preserved exactly, which is what
    # makes "sustained" and "bursty" a fair comparison.
    burst_factor: float = 4.0
    burst_on_s: float = 2.0

    # ramp: request_rate -> ramp_end_rate, linearly over the trace
    ramp_end_rate: float = 20.0

    # spike: step to spike_rate for spike_dur_s, starting at spike_at_s
    spike_at_s: float = 20.0
    spike_rate: float = 40.0
    spike_dur_s: float = 5.0

    # staircase: hold each level for `stair_step_s`, then step up by
    # `stair_step_pct` of the full rate. `request_rate` is the *peak*. Unlike a
    # smooth ramp this holds each level long enough to reach steady state, so
    # the level at which the system breaks is read directly off the plateau
    # rather than inferred from a moving target.
    stair_start_pct: float = 5.0
    stair_step_pct: float = 5.0
    stair_step_s: float = 60.0

    # ── multi-turn conversations ─────────────────────────────────
    # When set, `request_rate` becomes the *session* arrival rate and each
    # session runs several turns. The conversation is resent in full each turn,
    # so the shared prefix **grows** -- 500 tokens at turn 1, several thousand
    # by turn 8. That is a different regime from `prefix_heavy`, which shares a
    # *static* system prompt: there the cached prefix is constant, here it
    # extends every turn and each request is a partial match against the
    # previous turn's entire context.
    #
    # Turns within a session are closed-loop (a user waits for the reply, then
    # thinks, then replies) while sessions arrive open-loop. That hybrid is what
    # production actually looks like, and it means a slower server receives less
    # load from the same users -- real backpressure that a fixed trace cannot
    # express.
    multi_turn: bool = False
    turns_mu: float = 4.0               # mean turns per session
    turns_max: int = 12
    think_mu: float = 1.1               # lognormal seconds between turns
    think_sigma: float = 0.6

    # Human-plausible request mix. When set, prompts are generated from these
    # categories (see prompts.py) and each request's length comes from its own
    # category profile rather than one global distribution. When None, the
    # lognormal fields above govern and prompts are generic prose.
    category_mix: tuple[str, ...] | None = None

    # ── length distributions ─────────────────────────────────────
    # Lognormal; these are the underlying normal's mu/sigma, so median = exp(mu).
    input_len_mu: float = 6.0           # median ~403 tokens
    input_len_sigma: float = 0.8
    output_len_mu: float = 5.2          # median ~181 tokens
    output_len_sigma: float = 0.7
    input_len_cap: int = 8192
    output_len_cap: int = 2048

    # ── prefix sharing ───────────────────────────────────────────
    # Fraction of requests drawn from a pool of shared system prefixes. This is
    # what exercises the radix/prefix cache; a workload with zero sharing
    # cannot tell a good cache policy from a bad one.
    prefix_share_frac: float = 0.0
    n_shared_prefixes: int = 8
    shared_prefix_len: int = 512

    seed: int = 0

    def mean_rate(self) -> float:
        """Average arrival rate over the trace, for sizing and sanity checks."""
        if self.arrival == "ramp":
            return (self.request_rate + self.ramp_end_rate) / 2.0
        if self.arrival == "staircase":
            lv = self.stair_levels()
            return self.request_rate * sum(lv) / len(lv) / 100.0
        return self.request_rate

    def stair_levels(self) -> list[float]:
        """Percentages of the peak rate, one per plateau."""
        out, pct = [], self.stair_start_pct
        while pct <= 100.0 + 1e-9:
            out.append(min(100.0, pct))
            pct += self.stair_step_pct
        return out or [100.0]

    def stair_duration(self) -> float:
        return len(self.stair_levels()) * self.stair_step_s

    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class SLO:
    """Latency targets. Goodput counts only requests meeting *both*.

    `percentile` is which order statistic the frontier is judged at. It is a
    real choice, not a detail: a p99 on a 90-second window with few completions
    is effectively a maximum (we measured "p99" on 35 samples), whereas a p90
    is a genuine percentile at the same sample count. Tightening the level and
    loosening the percentile can describe the same service quality with far
    less measurement noise.
    """
    # Agreed 2026-09-01. TTFT is judged at p90 -- a per-request measurement
    # whose p99 our window lengths cannot resolve (at 90 s the "p99" is the
    # single worst request). TPOT is judged at p99 because it is ALREADY a mean
    # over ~2,000 tokens within a request, so its across-request p99 is a
    # meaningful tail rather than an extreme order statistic on raw samples.
    ttft_ms: float = 1000.0             # p99 TTFT
    tpot_ms: float = 50.0               # p99 TPOT -- the binding decode target
    percentile: int = 99

    # A second, tighter tier. Industry guidance for interactive serving sets
    # BOTH: a p90 the typical user feels, and a looser p99 for the tail. A
    # single p99 threshold cannot express "usually snappy, occasionally slow",
    # which is the actual product requirement. For a mid-size model the quoted
    # bands are p90 TTFT 300-500 ms / p99 800-1500 ms, and p90 TPOT 15-30 ms /
    # p99 35-50 ms -- with p99 TPOT above 60 ms said to visibly stutter.
    ttft_p90_ms: float | None = 300.0   # the binding prefill target
    tpot_p90_ms: float | None = 25.0

    def tiers(self) -> list[tuple[int, float, float]]:
        """(percentile, ttft_ms, tpot_ms), tightest first."""
        out = []
        if self.ttft_p90_ms is not None and self.tpot_p90_ms is not None:
            out.append((90, self.ttft_p90_ms, self.tpot_p90_ms))
        out.append((self.percentile, self.ttft_ms, self.tpot_ms))
        return out

    def digest(self) -> str:
        return _digest(asdict(self))
