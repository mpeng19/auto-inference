"""Request trace generation and the eval suite.

Three properties matter more than realism:

1. **Open-loop.** Arrival times are drawn *before* the run and the client fires
   at those times whether or not earlier requests finished. A closed-loop
   generator silently converts overload into slowdown, which flatters a bad
   scheduler and hides queueing.

2. **Deterministic.** Same config + seed gives a byte-identical trace, and
   `Trace.digest` goes into the run record. Comparing two configs against two
   different traces is the easiest way to fool yourself.

3. **Variance, not just mean.** Arrival *shape* stresses a serving system as
   much as arrival rate. `bursty` and `sustained` carry the same mean rate on
   purpose, so any difference between them is attributable to burstiness alone.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Callable

from .config import WorkloadConfig


@dataclass(frozen=True)
class Request:
    idx: int
    arrival_s: float        # seconds after trace start
    prompt: str
    prompt_tokens_est: int  # approximate; the server reports the true count
    max_tokens: int
    prefix_id: int | None   # which shared prefix, if any
    tag: str = ""           # which suite component produced it (for mixtures)


@dataclass(frozen=True)
class Trace:
    requests: tuple[Request, ...]
    config: WorkloadConfig

    @property
    def duration_s(self) -> float:
        return self.requests[-1].arrival_s if self.requests else 0.0

    @property
    def observed_rate(self) -> float:
        return len(self.requests) / self.duration_s if self.duration_s > 0 else 0.0

    def digest(self) -> str:
        blob = json.dumps(
            [(r.idx, round(r.arrival_s, 6), r.prompt_tokens_est, r.max_tokens,
              r.prefix_id, r.tag) for r in self.requests],
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def describe(self) -> dict:
        ins = [r.prompt_tokens_est for r in self.requests]
        outs = [r.max_tokens for r in self.requests]
        gaps = [b.arrival_s - a.arrival_s
                for a, b in zip(self.requests, self.requests[1:])]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        # Coefficient of variation of inter-arrival gaps: 0 = clockwork,
        # 1 = Poisson, >1 = bursty. The single most useful shape statistic.
        if gaps and mean_gap > 0:
            var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            cv = (var ** 0.5) / mean_gap
        else:
            cv = 0.0
        return {
            "name": self.config.name,
            "arrival": self.config.arrival,
            "n_requests": len(self.requests),
            "duration_s": round(self.duration_s, 2),
            "observed_rate_rps": round(self.observed_rate, 3),
            "interarrival_cv": round(cv, 3),
            "input_tokens": {"mean": round(sum(ins) / len(ins), 1) if ins else 0,
                             "min": min(ins) if ins else 0, "max": max(ins) if ins else 0},
            "output_tokens": {"mean": round(sum(outs) / len(outs), 1) if outs else 0,
                              "min": min(outs) if outs else 0, "max": max(outs) if outs else 0},
            "total_input_tokens": sum(ins),
            "total_output_tokens": sum(outs),
            "prefix_shared_frac": round(
                sum(1 for r in self.requests if r.prefix_id is not None) / len(self.requests), 3
            ) if self.requests else 0.0,
            "digest": self.digest(),
        }


# A token is ~4 characters of English. Filler text is generated to a target
# token count rather than tokenized for real, so trace building needs no GPU
# and no tokenizer. True token counts come back from the server.
_CHARS_PER_TOKEN = 4
_WORDS = (
    "system latency throughput scheduler batch cache prefix decode prefill "
    "tensor expert routing kernel memory request token attention parallel"
).split()


def _filler(n_tokens: int, rng: random.Random) -> str:
    target = max(1, n_tokens * _CHARS_PER_TOKEN)
    out, size = [], 0
    while size < target:
        w = rng.choice(_WORDS)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)


def _lognormal_int(rng: random.Random, mu: float, sigma: float, cap: int) -> int:
    return max(1, min(cap, int(rng.lognormvariate(mu, sigma))))


# ── arrival processes ────────────────────────────────────────────

def _rate_fn(cfg: WorkloadConfig, duration_s: float) -> tuple[Callable[[float], float], float]:
    """Return (instantaneous rate at t, an upper bound on it)."""
    r0 = cfg.request_rate

    if cfg.arrival in ("poisson", "constant"):
        return (lambda t: r0), max(r0, 1e-9)

    if cfg.arrival == "bursty":
        peak = r0 * cfg.burst_factor
        on = cfg.burst_on_s
        # Idle gap chosen so the mean rate is exactly preserved:
        #   peak * on / (on + off) = r0  =>  off = on * (burst_factor - 1)
        off = on * (cfg.burst_factor - 1.0)
        period = on + off
        if period <= 0:
            return (lambda t: r0), max(r0, 1e-9)
        return (lambda t: peak if (t % period) < on else 0.0), max(peak, 1e-9)

    if cfg.arrival == "ramp":
        r1 = cfg.ramp_end_rate
        D = max(duration_s, 1e-9)
        return (lambda t: r0 + (r1 - r0) * min(t / D, 1.0)), max(r0, r1, 1e-9)

    if cfg.arrival == "spike":
        a, b = cfg.spike_at_s, cfg.spike_at_s + cfg.spike_dur_s
        return (lambda t: cfg.spike_rate if a <= t < b else r0), max(r0, cfg.spike_rate, 1e-9)

    raise ValueError(f"unknown arrival process: {cfg.arrival!r}")


def _arrival_times(cfg: WorkloadConfig, rng: random.Random) -> list[float]:
    # Duration is either given, or derived from the mean rate with headroom so
    # that a count-based trace reliably reaches n_requests.
    if cfg.duration_s is not None:
        duration = cfg.duration_s
        hard_cap = cfg.n_requests
    else:
        n = cfg.n_requests or 100
        duration = (n / max(cfg.mean_rate(), 1e-9)) * 1.5
        hard_cap = n

    if cfg.arrival == "constant":
        gap = 1.0 / max(cfg.request_rate, 1e-9)
        times = []
        t = gap
        while t <= duration:
            times.append(t)
            t += gap
    else:
        # Lewis-Shedler thinning: sample at the upper bound, keep with
        # probability rate(t)/bound. Exact for any bounded rate function.
        rate_at, bound = _rate_fn(cfg, duration)
        times, t = [], 0.0
        while True:
            t += rng.expovariate(bound)
            if t > duration:
                break
            if rng.random() < rate_at(t) / bound:
                times.append(t)

    if hard_cap is not None:
        times = times[:hard_cap]
    return times


def build_trace(cfg: WorkloadConfig) -> Trace:
    rng = random.Random(cfg.seed)

    shared = [
        _filler(cfg.shared_prefix_len, random.Random(cfg.seed * 1000 + i))
        for i in range(cfg.n_shared_prefixes)
    ]

    times = _arrival_times(cfg, rng)

    reqs: list[Request] = []
    for i, t in enumerate(times):
        in_len = _lognormal_int(rng, cfg.input_len_mu, cfg.input_len_sigma, cfg.input_len_cap)
        out_len = _lognormal_int(rng, cfg.output_len_mu, cfg.output_len_sigma, cfg.output_len_cap)

        prefix_id = None
        if shared and rng.random() < cfg.prefix_share_frac:
            prefix_id = rng.randrange(len(shared))
            body = max(1, in_len - cfg.shared_prefix_len)
            prompt = shared[prefix_id] + " " + _filler(body, rng)
            in_len = cfg.shared_prefix_len + body
        else:
            prompt = _filler(in_len, rng)

        reqs.append(Request(i, t, prompt, in_len, out_len, prefix_id, cfg.name))

    return Trace(tuple(reqs), cfg)


def merge_traces(traces: list[Trace], name: str = "mixed") -> Trace:
    """Interleave several traces into one heterogeneous stream.

    Rates add: merging 2 rps and 3 rps yields 5 rps. Each request keeps a `tag`
    naming its source component, so results can be sliced by traffic class
    afterwards — which is the point of a mixed workload.
    """
    merged = sorted(
        (r for t in traces for r in t.requests), key=lambda r: r.arrival_s
    )
    reindexed = tuple(replace(r, idx=i) for i, r in enumerate(merged))
    base = traces[0].config if traces else WorkloadConfig()
    return Trace(reindexed, replace(base, name=name))


# ── the eval suite ───────────────────────────────────────────────
# Each entry stresses a different part of the serving stack. Run the whole
# suite against a config; a change that helps one pattern and wrecks another is
# a trade-off to see explicitly, not to average away.

def suite(seed: int = 0, scale: float = 1.0) -> dict[str, WorkloadConfig]:
    """Named workloads. `scale` multiplies request counts for longer runs."""
    n = lambda k: max(20, int(k * scale))

    return {
        # Baseline: smooth, memoryless. Everything else is compared to this.
        "sustained": WorkloadConfig(
            name="sustained", arrival="poisson", request_rate=4.0,
            n_requests=n(300), seed=seed),

        # Control: zero arrival variance. Any gap between this and `sustained`
        # is the cost of randomness alone, independent of the scheduler.
        "constant": WorkloadConfig(
            name="constant", arrival="constant", request_rate=4.0,
            n_requests=n(300), seed=seed),

        # Same mean rate as `sustained`, delivered in 4x bursts. Isolates the
        # effect of burstiness on queueing and batch formation.
        "bursty": WorkloadConfig(
            name="bursty", arrival="bursty", request_rate=4.0,
            burst_factor=4.0, burst_on_s=2.0, n_requests=n(300), seed=seed),

        # Climbs until the system breaks. Locates the saturation knee, which is
        # the number that actually matters for capacity planning.
        "ramp": WorkloadConfig(
            name="ramp", arrival="ramp", request_rate=2.0, ramp_end_rate=32.0,
            duration_s=180.0, n_requests=None, seed=seed),

        # Sudden 10x step. Tests admission control and, more importantly,
        # whether the system *recovers* once the spike passes.
        "spike": WorkloadConfig(
            name="spike", arrival="spike", request_rate=4.0, spike_rate=40.0,
            spike_at_s=30.0, spike_dur_s=10.0, duration_s=90.0,
            n_requests=None, seed=seed),

        # Prefill-dominated: long prompts, short answers. Stresses chunked
        # prefill and prefill/decode interference.
        "prefill_heavy": WorkloadConfig(
            name="prefill_heavy", arrival="poisson", request_rate=1.5,
            input_len_mu=8.1, input_len_sigma=0.5,      # median ~3300 tok
            output_len_mu=3.4, output_len_sigma=0.4,    # median ~30 tok
            n_requests=n(150), seed=seed),

        # Decode-dominated: short prompts, long answers. Stresses KV growth,
        # batch residency and inter-token latency.
        "decode_heavy": WorkloadConfig(
            name="decode_heavy", arrival="poisson", request_rate=2.0,
            input_len_mu=3.9, input_len_sigma=0.4,      # median ~50 tok
            output_len_mu=6.7, output_len_sigma=0.5,    # median ~810 tok
            n_requests=n(150), seed=seed),

        # Heavy prefix reuse, as in multi-turn chat with a fixed system prompt.
        # Without this, a good radix-cache policy is indistinguishable from a
        # bad one.
        "prefix_heavy": WorkloadConfig(
            name="prefix_heavy", arrival="poisson", request_rate=6.0,
            prefix_share_frac=0.9, n_shared_prefixes=6, shared_prefix_len=1024,
            input_len_mu=7.2, input_len_sigma=0.4,
            output_len_mu=4.6, output_len_sigma=0.5,
            n_requests=n(400), seed=seed),

        # Many tiny requests: per-request scheduling overhead dominates, not
        # GPU work.
        "short_chat": WorkloadConfig(
            name="short_chat", arrival="poisson", request_rate=24.0,
            input_len_mu=3.5, input_len_sigma=0.5,      # median ~33 tok
            output_len_mu=3.9, output_len_sigma=0.5,    # median ~50 tok
            n_requests=n(800), seed=seed),
    }


def mixed_trace(seed: int = 0, scale: float = 1.0) -> Trace:
    """Heterogeneous production-like stream: several classes sharing one server.

    This is where schedulers usually fail. Optimising each class alone tends to
    produce a policy that starves one of them once they compete.
    """
    s = suite(seed=seed, scale=scale)
    parts = [build_trace(s[k]) for k in
             ("sustained", "prefill_heavy", "decode_heavy", "prefix_heavy")]
    return merge_traces(parts, name="mixed")
