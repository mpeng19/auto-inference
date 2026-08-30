"""Typed configuration.

`ServingConfig` *is* the search space: every field here is a knob an optimizer
is allowed to turn. Anything not in this dataclass is held fixed, and anything
held fixed must be recorded in the run record so results stay comparable.

Frozen + hashable on purpose. The digest is what dedupes a sweep and what keys
a cached result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


# Flag names verified against sglang 0.5.18 by `probe.py::probe_env` on
# 2026-08-29: all 12 present. Re-run that probe after any SGLang upgrade —
# SGLang renames flags between minor versions, and a silently-ignored flag
# produces a run that looks successful while measuring the wrong config.
@dataclass(frozen=True)
class ServingConfig:
    # ── fixed within a study (recorded, not searched) ────────────
    #
    # Default is the small dense model on a cheap GPU: $1.10/hr against $3.95,
    # so harness iteration costs a third as much. Be clear about what that
    # trades away -- Qwen3-4B on an A10G is **prefill-bound** at our request
    # shapes, while Qwen3-30B-A3B on an H100 is **decode-bound**. Opposite sides
    # of the roofline. Use the small setup to develop and debug the harness; use
    # the 30B (or the 235B on 8xH100) whenever a serving-policy result is meant
    # to transfer.
    #
    #   Qwen/Qwen3-4B-Instruct-2507-FP8    A10G   $1.10/hr   262k ctx
    #   Qwen/Qwen3-8B-FP8                  L40S   $1.95/hr    41k ctx
    #   Qwen/Qwen3-30B-A3B-Instruct-2507-FP8  H100  $3.95/hr  MoE, decode-bound
    #   Qwen/Qwen3-235B-A22B-Instruct-2507-FP8  8xH100  $31.60/hr
    model: str = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    gpu: str = "A10G"
    n_gpu: int = 1

    # ── parallelism ──────────────────────────────────────────────
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int | None = None          # expert parallelism; None = disabled

    # ── memory ───────────────────────────────────────────────────
    mem_fraction_static: float = 0.85   # VRAM share for weights + KV pool
    max_total_tokens: int | None = None

    # ── batching / scheduling (the interesting knobs) ────────────
    max_running_requests: int = 256
    chunked_prefill_size: int = 8192
    # Verified against sglang 0.5.18 --help by probe_env. Seven policies,
    # not the four originally assumed: lof, priority and routing-key are
    # extra search-space dimensions we would otherwise have missed.
    schedule_policy: str = "fcfs"
    # {lpm, random, fcfs, dfs-weight, lof, priority, routing-key}
    schedule_conservativeness: float = 1.0

    # ── caching ──────────────────────────────────────────────────
    enable_prefix_caching: bool = True

    # ── observability ────────────────────────────────────────────
    # Server-side histograms (TTFT, ITL, queue time, cache hit rate). These are
    # measured from request arrival inside the inference system, so they are
    # independent of where load is generated -- which is what makes an
    # agent-driven client on another machine comparable to the in-container one.
    enable_metrics: bool = True

    # ── misc ─────────────────────────────────────────────────────
    context_length: int | None = None
    extra_args: tuple[str, ...] = ()

    def to_sglang_args(self) -> list[str]:
        """Render as `sglang.launch_server` CLI arguments."""
        a = [
            "--model-path", self.model,
            "--tp-size", str(self.tp_size),
            "--dp-size", str(self.dp_size),
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--max-running-requests", str(self.max_running_requests),
            "--chunked-prefill-size", str(self.chunked_prefill_size),
            "--schedule-policy", self.schedule_policy,
            "--schedule-conservativeness", str(self.schedule_conservativeness),
        ]
        if self.ep_size is not None:
            a += ["--ep-size", str(self.ep_size)]
        if self.max_total_tokens is not None:
            a += ["--max-total-tokens", str(self.max_total_tokens)]
        if self.context_length is not None:
            a += ["--context-length", str(self.context_length)]
        if not self.enable_prefix_caching:
            a += ["--disable-radix-cache"]
        if self.enable_metrics:
            a += ["--enable-metrics"]
        a += list(self.extra_args)
        return a

    def digest(self) -> str:
        return _digest(asdict(self))


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
    """Latency targets. Goodput counts only requests meeting *both*."""
    ttft_ms: float = 500.0
    tpot_ms: float = 40.0               # mean inter-token latency per request

    def digest(self) -> str:
        return _digest(asdict(self))


def _digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
