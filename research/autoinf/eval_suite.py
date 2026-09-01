"""The nine-workload eval suite and its open-loop trace builder.

Retired from the product 2026-09-01. It varies arrival *shape* at fixed
mean rate, which is the right tool for scheduler comparison and the wrong
one for pricing: the settled method needs a closed-loop population sweep
on real traffic (HANDOFF SS6b). Kept for the congestion-collapse work.
"""
from __future__ import annotations

import math, random
from collections.abc import Callable
from dataclasses import dataclass, field
from research.autoinf.workload_config import WorkloadConfig
from simulator.workload.sessions import Session, Turn, _filler, _lognormal_int


@dataclass(frozen=True)
class Request:
    idx: int
    arrival_s: float        # seconds after trace start
    prompt: str
    prompt_tokens_est: int  # approximate; the server reports the true count
    max_tokens: int
    prefix_id: int | None   # which shared prefix, if any
    tag: str = ""           # which suite component produced it (for mixtures)
    category: str = ""      # human-request category, when a mix is configured


@dataclass(frozen=True)
class SessionTrace:
    sessions: tuple[Session, ...]
    config: WorkloadConfig

    @property
    def duration_s(self) -> float:
        return self.sessions[-1].arrival_s if self.sessions else 0.0

    @property
    def n_turns(self) -> int:
        return sum(s.n_turns for s in self.sessions)

    def digest(self) -> str:
        blob = json.dumps(
            [(s.idx, round(s.arrival_s, 6), len(s.turns),
              [(len(t.text), t.max_tokens) for t in s.turns]) for s in self.sessions],
            sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def describe(self) -> dict:
        tp = [s.n_turns for s in self.sessions]
        firsts = [len(s.turns[0].text) // 4 for s in self.sessions if s.turns]
        return {
            "name": self.config.name,
            "multi_turn": True,
            "n_sessions": len(self.sessions),
            "n_turns": self.n_turns,
            "duration_s": round(self.duration_s, 2),
            "session_rate_rps": round(len(self.sessions) / self.duration_s, 3)
            if self.duration_s else 0.0,
            "turns_per_session_mean": round(sum(tp) / len(tp), 2) if tp else 0,
            "turns_per_session_max": max(tp) if tp else 0,
            "first_turn_tokens_mean": round(sum(firsts) / len(firsts)) if firsts else 0,
            "system_prompt_tokens": len(self.sessions[0].system) // 4 if self.sessions else 0,
            "digest": self.digest(),
        }


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

    if cfg.arrival == "staircase":
        levels = cfg.stair_levels()
        step = cfg.stair_step_s

        def rate_at(t: float) -> float:
            i = min(int(t / step), len(levels) - 1)
            return r0 * levels[i] / 100.0

        return rate_at, max(r0, 1e-9)

    raise ValueError(f"unknown arrival process: {cfg.arrival!r}")


def _arrival_times(cfg: WorkloadConfig, rng: random.Random) -> list[float]:
    # Duration is either given, or derived from the mean rate with headroom so
    # that a count-based trace reliably reaches n_requests.
    if cfg.arrival == "staircase" and cfg.duration_s is None:
        # The plateau schedule defines the length; a request count would
        # truncate it mid-climb and hide the level we are looking for.
        duration = cfg.stair_duration()
        hard_cap = None
    elif cfg.duration_s is not None:
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

    # Real deployments put a stable system preamble in front of every request;
    # that is what a prefix cache exists to exploit. Use real ones when a human
    # mix is configured, padded to the requested prefix length.
    if cfg.category_mix:
        base = [t for _, t in _prompts.SYSTEM_PROMPTS]
        shared = [
            _prompts._pad_to(base[i % len(base)], cfg.shared_prefix_len,
                             random.Random(cfg.seed * 1000 + i))
            for i in range(cfg.n_shared_prefixes)
        ]
    else:
        shared = [
            _filler(cfg.shared_prefix_len, random.Random(cfg.seed * 1000 + i))
            for i in range(cfg.n_shared_prefixes)
        ]

    times = _arrival_times(cfg, rng)

    mix = cfg.category_mix
    reqs: list[Request] = []
    for i, t in enumerate(times):
        cat_name = ""
        if mix:
            # Human mix: the category picks the length profile, because intent
            # and length are correlated in real traffic -- a summarize request
            # is long-in/short-out, a creative one is the reverse. Drawing both
            # from one global distribution erases exactly the prefill/decode
            # asymmetry that serving decisions turn on.
            cat = _prompts.sample_category(rng, mix)
            cat_name = cat.name
            in_len = _lognormal_int(rng, cat.in_mu, cat.in_sigma, cfg.input_len_cap)
            out_len = _lognormal_int(rng, cat.out_mu, cat.out_sigma, cfg.output_len_cap)
        else:
            in_len = _lognormal_int(rng, cfg.input_len_mu, cfg.input_len_sigma,
                                    cfg.input_len_cap)
            out_len = _lognormal_int(rng, cfg.output_len_mu, cfg.output_len_sigma,
                                     cfg.output_len_cap)

        body_target = in_len
        prefix_id = None
        prefix_text = ""
        if shared and rng.random() < cfg.prefix_share_frac:
            prefix_id = rng.randrange(len(shared))
            prefix_text = shared[prefix_id] + "\n\n"
            body_target = max(1, in_len - cfg.shared_prefix_len)

        if mix:
            body = _prompts.make_request(rng, cat_name, body_target)
            in_len = int(len(prefix_text + body) / _prompts.CHARS_PER_TOKEN)
        else:
            body = _filler(body_target, rng)
            in_len = (cfg.shared_prefix_len + body_target) if prefix_id is not None \
                else body_target

        reqs.append(Request(i, t, prefix_text + body, in_len, out_len,
                            prefix_id, cfg.name, cat_name))

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


def roofline_rps(model=None, hw=None, in_tok: int | None = None,
                 out_tok: int | None = None, batch: int = 146) -> float:
    """Ceiling request rate for the human mix, from the capacity model.

    Sizing the suite as a *fraction of theoretical capacity* rather than an
    absolute rps makes the same suite meaningful on different hardware. An
    "80% of roofline" workload means the same thing on 1xH100 and 8xH100; "32
    rps" does not.
    """
    import math

    from .flops import H100, QWEN3_30B_A3B, capacity
    from .prompts import CATEGORIES

    model = model or QWEN3_30B_A3B
    hw = hw or H100
    if in_tok is None or out_tok is None:
        tw = sum(c.weight for c in CATEGORIES)
        in_tok = int(sum(c.weight * math.exp(c.in_mu + c.in_sigma ** 2 / 2)
                         for c in CATEGORIES) / tw)
        out_tok = int(sum(c.weight * math.exp(c.out_mu + c.out_sigma ** 2 / 2)
                          for c in CATEGORIES) / tw)
    return capacity(model, hw, in_tok, out_tok, batch=batch)["max_rps_roofline"]


def suite(seed: int = 0, scale: float = 1.0,
          minutes: float | None = None) -> dict[str, WorkloadConfig]:
    """Named workloads. `scale` multiplies request counts for longer runs.

    Rates are calibrated against the **measured** behaviour of
    Qwen3-30B-A3B-FP8 on one H100 (see docs/HANDOFF.md):

        offered 35.6 rps -> 100% met SLO       max throughput 34.3 rps
        offered 49.1 rps ->  24% met SLO

    **Two corrections got us here.** The first calibration guessed rates ~10x
    too low: every workload passed trivially, measuring an idle server. The
    second overcorrected to ~32 rps, just under the knee — which put the suite
    inside the *metastable* region. Two identical runs there produced 30.47 and
    19.62 goodput, and one of them escalated from 96ms to 2704ms TTFT mid-trace
    at unchanged throughput. At ~88% utilisation there is no slack to drain a
    transient, so a run measures which basin it fell into.

    The general-purpose workloads now sit near **60% of measured max
    throughput**, comfortably outside that region, which is what makes A/B
    comparison meaningful. Load near and past the knee is still covered, by
    `ramp`, `spike` and `stress` — but there the honest metric is *whether and
    when it collapses* (`metrics.detect_collapse`), not goodput at a point.

    Per-workload capacity differs sharply because the bottleneck differs:
    prefill_heavy is prefill-bound (~8300 prefill tok/s, so ~2 req/s at 3825
    tokens each), decode_heavy is decode-bound, short_chat is barely bound at
    all. Rates are set near each one's own limit, not to a single global number.

    **Re-derive these after any change to model, hardware or GPU count.** A
    rate calibrated for 1xH100 means nothing on 8xH100.

    `minutes` sets the total wall time of the traces, divided across workloads
    in proportion to their default sizes. Longer runs are not just more data:
    they expose drift a short run cannot -- KV fragmentation, cache growth,
    thermal effects -- which is exactly what a consistency check needs to see.
    """
    if minutes is not None:
        # Default traces total ~591s. Scale request counts to hit the target.
        scale = scale * (minutes * 60.0) / 591.0
    n = lambda k: max(20, int(k * scale))

    return {
        # Just below the knee: sensitive to scheduling without being a
        # foregone collapse. Everything else is compared to this.
        "sustained": WorkloadConfig(
            name="sustained", arrival="poisson", request_rate=20.0,
            n_requests=n(1600), seed=seed),

        # Control: zero arrival variance at the same mean rate. The gap to
        # `sustained` is the cost of randomness alone.
        "constant": WorkloadConfig(
            name="constant", arrival="constant", request_rate=20.0,
            n_requests=n(1600), seed=seed),

        # Same mean rate as `sustained`, delivered in 4x bursts that peak at
        # 128 rps — well past the knee. Isolates burstiness: identical load,
        # different shape, and the bursts should now actually hurt.
        "bursty": WorkloadConfig(
            name="bursty", arrival="bursty", request_rate=20.0,
            burst_factor=4.0, burst_on_s=2.0, n_requests=n(1600), seed=seed),

        # Brackets the knee (10 -> 64) so the bend is inside the trace.
        "ramp": WorkloadConfig(
            name="ramp", arrival="ramp", request_rate=10.0, ramp_end_rate=64.0,
            duration_s=120.0, n_requests=None, seed=seed),

        # Sits safely below the knee, then steps 4x past it for 10s. Tests
        # admission control and, more importantly, whether it *recovers*.
        "spike": WorkloadConfig(
            name="spike", arrival="spike", request_rate=24.0, spike_rate=96.0,
            spike_at_s=30.0, spike_dur_s=10.0, duration_s=90.0,
            n_requests=None, seed=seed),

        # Prefill-dominated and prefill-bound: ~3825 tokens in, ~30 out. At
        # ~8300 prefill tok/s the ceiling is ~2.2 req/s, so 1.8 is near the
        # limit — this workload was already close to saturated at 1.5.
        "prefill_heavy": WorkloadConfig(
            name="prefill_heavy", arrival="poisson", request_rate=1.8,
            input_len_mu=8.1, input_len_sigma=0.5,
            output_len_mu=3.4, output_len_sigma=0.4,
            n_requests=n(140), seed=seed),

        # Decode-dominated: ~55 in, ~890 out. Bound by aggregate decode
        # throughput (~7800 output tok/s), so ~8.8 req/s; 7.0 sits under it.
        "decode_heavy": WorkloadConfig(
            name="decode_heavy", arrival="poisson", request_rate=7.0,
            input_len_mu=3.9, input_len_sigma=0.4,
            output_len_mu=6.7, output_len_sigma=0.5,
            n_requests=n(420), seed=seed),

        # 89% of requests share a 1024-token prefix, so effective prefill is
        # only the ~315-token unique body. That is a *partial* match, the
        # regime where the cache genuinely helps (1.76x measured), which is why
        # this can run far above prefill_heavy's rate.
        "prefix_heavy": WorkloadConfig(
            name="prefix_heavy", arrival="poisson", request_rate=20.0,
            prefix_share_frac=0.9, n_shared_prefixes=6, shared_prefix_len=1024,
            input_len_mu=7.2, input_len_sigma=0.4,
            output_len_mu=4.6, output_len_sigma=0.5,
            n_requests=n(1000), seed=seed),

        # Tiny requests where per-request scheduling overhead dominates rather
        # than GPU work. Ran clean at 23 rps, so push well past that.
        "short_chat": WorkloadConfig(
            name="short_chat", arrival="poisson", request_rate=80.0,
            input_len_mu=3.5, input_len_sigma=0.5,
            output_len_mu=3.9, output_len_sigma=0.5,
            n_requests=n(3200), seed=seed),

        # Realistic mixed traffic: ten request categories at their real-world
        # shares, human-plausible prompts, and 40% behind a shared system
        # prompt. Held at the same stable rate as `sustained` so the two are
        # directly comparable -- the difference is then traffic *composition*,
        # not load.
        # Deliberately inside the metastable region. Goodput here is not a
        # reproducible number and must not be A/B'd; the metric is
        # `detect_collapse` -- did it run away, and when. Run it repeatedly and
        # report the collapse *rate*.
        "stress": WorkloadConfig(
            name="stress", arrival="poisson", request_rate=32.0,
            category_mix=_prompts.ALL_CATEGORIES,
            prefix_share_frac=0.4, n_shared_prefixes=3, shared_prefix_len=180,
            n_requests=n(1600), seed=seed),

        # ── multi-turn ───────────────────────────────────────────
        # Conversations, not independent requests. `request_rate` is the
        # *session* arrival rate; each session runs ~4 turns with human think
        # time, resending the whole conversation each time. The shared prefix
        # therefore grows turn over turn, which is the pattern real chat has and
        # `prefix_heavy` (static shared prompt) does not.
        "chat_multiturn": WorkloadConfig(
            name="chat_multiturn", multi_turn=True, request_rate=4.0,
            turns_mu=4.0, think_mu=1.1, think_sigma=0.6,
            category_mix=_prompts.ALL_CATEGORIES,
            shared_prefix_len=180, n_requests=n(300), seed=seed),

        # Long conversations: 10 turns, so the prefix reaches several thousand
        # tokens. This is where growing-prefix caching either pays off or does
        # not, and it is the closest thing in the suite to agentic traffic.
        "chat_deep": WorkloadConfig(
            name="chat_deep", multi_turn=True, request_rate=1.5,
            turns_mu=10.0, turns_max=20, think_mu=0.7, think_sigma=0.5,
            category_mix=("code_debug", "analysis", "explain", "rag"),
            shared_prefix_len=400, n_requests=n(80), seed=seed),

        "human": WorkloadConfig(
            name="human", arrival="poisson",
            request_rate=20.0,
            category_mix=_prompts.ALL_CATEGORIES,
            prefix_share_frac=0.4, n_shared_prefixes=3, shared_prefix_len=180,
            n_requests=n(1500), seed=seed),
    }


def staircase(seed: int = 0, peak_fraction: float = 1.0, step_pct: float = 5.0,
              step_s: float = 60.0, start_pct: float = 5.0) -> WorkloadConfig:
    """Step to `peak_fraction` of roofline in `step_pct` increments.

    Each level is held long enough (60s) to reach steady state, so the level at
    which the system breaks is read straight off a plateau. A smooth ramp
    cannot do that: the rate is still moving while the queue is filling, so the
    reported break point lags the true one by however long the queue takes to
    build.

    Defaults give 20 levels of 60s = 20 minutes, stepping 5% -> 100% of the
    theoretical ceiling.
    """
    return WorkloadConfig(
        name="staircase", arrival="staircase",
        request_rate=round(roofline_rps() * peak_fraction, 1),
        stair_start_pct=start_pct, stair_step_pct=step_pct, stair_step_s=step_s,
        category_mix=_prompts.ALL_CATEGORIES,
        prefix_share_frac=0.4, n_shared_prefixes=3, shared_prefix_len=180,
        n_requests=None, duration_s=None, seed=seed)


def mixed_trace(seed: int = 0, scale: float = 1.0) -> Trace:
    """Heterogeneous production-like stream: several classes sharing one server.

    This is where schedulers usually fail. Optimising each class alone tends to
    produce a policy that starves one of them once they compete.
    """
    s = suite(seed=seed, scale=scale)
    parts = [build_trace(s[k]) for k in
             ("sustained", "prefill_heavy", "decode_heavy", "prefix_heavy")]
    return merge_traces(parts, name="mixed")


def staircase_levels(seed: int = 0, peak_fraction: float = 1.0,
                     step_pct: float = 5.0, step_s: float = 60.0,
                     start_pct: float = 5.0,
                     minutes: float | None = None) -> list[WorkloadConfig]:
    """The staircase as one workload per plateau.

    Each level is an independent Poisson workload at a fixed rate. Splitting it
    this way means every plateau gets its own server-metrics slice and its own
    client-health verdict, and the sequence can be cut short the moment the
    system breaks -- which a single pre-built trace cannot do.
    """
    if minutes is not None:
        n_levels = max(1, int((100.0 - start_pct) / step_pct) + 1)
        step_s = minutes * 60.0 / n_levels
    peak = roofline_rps() * peak_fraction
    out, pct = [], start_pct
    while pct <= 100.0 + 1e-9:
        rate = peak * min(100.0, pct) / 100.0
        out.append(WorkloadConfig(
            name=f"L{int(pct):03d}pct", arrival="poisson", request_rate=rate,
            duration_s=step_s, n_requests=None,
            category_mix=_prompts.ALL_CATEGORIES,
            prefix_share_frac=0.4, n_shared_prefixes=3, shared_prefix_len=180,
            seed=seed + int(pct)))
        pct += step_pct
    return out
