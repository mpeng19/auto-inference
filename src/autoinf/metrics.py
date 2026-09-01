"""Per-request records and the metrics derived from them.

Store per-request rows, not aggregates. Aggregates cannot be re-sliced, and
you will want to re-slice (by prompt length, by prefix hit, by arrival phase)
long after the GPU has been released.

The headline number is SLO-constrained **goodput**, not throughput. A server
can post excellent tokens/sec while missing every latency target; goodput is
the metric that refuses to reward that.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import SLO


@dataclass
class RequestResult:
    idx: int
    scheduled_s: float          # when the trace said to send it
    dispatched_s: float         # when the client actually sent it
    first_token_s: float | None
    end_s: float | None
    prompt_tokens: int | None
    output_tokens: int
    ok: bool
    error: str | None = None
    # Populated for multi-turn runs. Turn depth is the axis that matters there:
    # TTFT should improve with depth as the cached conversation prefix grows,
    # even though the prompt is getting longer.
    session: int = -1
    turn: int = -1
    history_tokens: int = 0

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_s is None:
            return None
        return (self.first_token_s - self.dispatched_s) * 1000.0

    @property
    def tpot_ms(self) -> float | None:
        """Mean inter-token latency, excluding the first token."""
        if self.first_token_s is None or self.end_s is None or self.output_tokens < 2:
            return None
        return (self.end_s - self.first_token_s) * 1000.0 / (self.output_tokens - 1)

    @property
    def e2e_ms(self) -> float | None:
        if self.end_s is None:
            return None
        return (self.end_s - self.dispatched_s) * 1000.0

    @property
    def dispatch_lag_ms(self) -> float:
        """How late the *client* was firing this request.

        This is a self-check, not a server metric. If lag grows through the
        run, the load generator is the bottleneck and every server number in
        the run is suspect.
        """
        return (self.dispatched_s - self.scheduled_s) * 1000.0


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]. Matches numpy's default."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
    }


def summarize(results: list[RequestResult], slo: SLO, window_s: float | None = None,
              warmup_s: float = 0.0) -> dict:
    """Aggregate per-request records.

    `warmup_s` discards requests scheduled in the opening seconds of a trace.
    This matters more than it sounds: a 10-minute suite and a 30-minute suite
    of the *same* workload at the same offered rate differed by 17% in goodput
    (26.13 vs 30.53), because a fixed startup transient is amortised over more
    requests in a longer trace. Workloads also run sequentially against one
    server, so whichever goes first sees the coldest cache.

    Excluding the transient makes a result depend on the workload rather than
    on how long the trace happened to be or where it sat in the sequence.
    """
    measured = [r for r in results if r.scheduled_s >= warmup_s]
    excluded = len(results) - len(measured)
    results = measured or results
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    e2es = [r.e2e_ms for r in ok if r.e2e_ms is not None]

    # Measurement window: first dispatch to last completion.
    if window_s is None:
        starts = [r.dispatched_s for r in results]
        ends = [r.end_s for r in ok if r.end_s is not None]
        window_s = (max(ends) - min(starts)) if (starts and ends) else 0.0

    # A request is "good" only if it met both targets. Requests that failed or
    # never produced a token are never good.
    good = 0
    for r in ok:
        t, p = r.ttft_ms, r.tpot_ms
        if t is None or t > slo.ttft_ms:
            continue
        # Single-token outputs have no TPOT to violate.
        if p is not None and p > slo.tpot_ms:
            continue
        good += 1

    out_tokens = sum(r.output_tokens for r in ok)
    lags = [r.dispatch_lag_ms for r in results]

    return {
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "n_excluded_warmup": excluded,
        "warmup_s": warmup_s,
        "window_s": window_s,
        # ── headline ──
        "goodput_rps": good / window_s if window_s > 0 else 0.0,
        "n_good": good,
        "good_frac": good / len(results) if results else 0.0,
        # ── conventional throughput, for context only ──
        "throughput_rps": len(ok) / window_s if window_s > 0 else 0.0,
        "output_tok_per_s": out_tokens / window_s if window_s > 0 else 0.0,
        "total_output_tokens": out_tokens,
        # ── latency ──
        "ttft_ms": _stats(ttfts),
        "tpot_ms": _stats(tpots),
        "e2e_ms": _stats(e2es),
        # ── harness self-check ──
        "client_dispatch_lag_ms": _stats(lags),
        "slo": asdict(slo),
        "errors": _error_counts(failed),
    }


def _error_counts(failed: list[RequestResult]) -> dict:
    counts: dict[str, int] = {}
    for r in failed:
        key = (r.error or "unknown")[:120]
        counts[key] = counts.get(key, 0) + 1
    return counts


def detect_collapse(results: list["RequestResult"], n_buckets: int = 12,
                    escalation_factor: float = 3.0) -> dict:
    """Did latency run away during the trace?

    Near saturation a continuous-batching server is **metastable**: the same
    offered load either holds steady or enters an unbounded escalation,
    depending on whether an early transient happened to build a backlog that
    could not drain. Observed directly -- two identical 30 rps runs, one flat at
    40ms TTFT for 152s, the other climbing 96 -> 2704ms and never recovering,
    at the same throughput.

    Single-point goodput cannot express this: it reports which basin the run
    fell into, and averaging across runs invents a middle value that never
    occurs. So collapse is measured as its own property.

    A stable run has flat TTFT across the trace. A collapsed run has TTFT that
    climbs monotonically and ends far above where it started.
    """
    ok = [r for r in results if r.ok and r.ttft_ms is not None]
    if len(ok) < 4 * n_buckets:
        return {"available": False, "reason": "too few requests to bucket"}

    span = max(r.scheduled_s for r in ok) - min(r.scheduled_s for r in ok)
    if span <= 0:
        return {"available": False, "reason": "zero-duration trace"}
    t0 = min(r.scheduled_s for r in ok)

    buckets: list[list[float]] = [[] for _ in range(n_buckets)]
    for r in ok:
        i = min(n_buckets - 1, int((r.scheduled_s - t0) / span * n_buckets))
        buckets[i].append(r.ttft_ms)

    med = [percentile(b, 50) for b in buckets if b]
    if len(med) < n_buckets // 2:
        return {"available": False, "reason": "sparse buckets"}

    first, last = med[0], med[-1]
    ratio = last / first if first > 0 else float("inf")

    # Escalation must be sustained rather than a spike that recovered. Two
    # conditions: the final third sits well above the first, and the trace
    # *ends* near its worst.
    #
    # An earlier version also required most bucket-to-bucket transitions to
    # rise. That was wrong: a collapse beginning midway through has a flat
    # prefix contributing no rises, so the check rejected precisely the shape it
    # was meant to catch. "Ends near its worst" separates runaway from spike
    # without penalising a late onset.
    third = max(1, len(med) // 3)
    early = sum(med[:third]) / third
    late = sum(med[-third:]) / third
    peak = max(med)
    ends_high = med[-1] >= 0.7 * peak
    rises = sum(1 for a, b in zip(med, med[1:]) if b > a)
    monotonic_frac = rises / max(1, len(med) - 1)

    collapsed = (late > early * escalation_factor) and ends_high

    onset = None
    if collapsed:
        for i, v in enumerate(med):
            if v > early * escalation_factor:
                onset = round(t0 + span * i / n_buckets, 1)
                break

    return {
        "available": True,
        "collapsed": collapsed,
        "ttft_first_bucket_ms": round(first, 1),
        "ttft_last_bucket_ms": round(last, 1),
        "escalation_ratio": round(ratio, 2),
        "early_third_ms": round(early, 1),
        "late_third_ms": round(late, 1),
        "monotonic_frac": round(monotonic_frac, 2),
        "ends_near_peak": ends_high,
        "onset_s": onset,
        "bucket_medians_ms": [round(m, 1) for m in med],
        "note": ("A collapsed run is not a slower run: throughput is typically "
                 "unchanged. It has entered a backlog it cannot drain, so "
                 "goodput from it measures the basin, not the configuration."),
    }


def littles_law(n_users: int, throughput_rps: float, batch: float,
                queued: float = 0.0) -> dict:
    """Cross-check three independently measured numbers against L = lambda*W.

    Concurrency, throughput and residence time are not free of one another:
    Little's Law ties them by an identity that holds for any stable system,
    whatever the scheduling discipline. So it costs nothing and catches a class
    of error nothing else here does.

    Two boundaries:

        server: running + queued = throughput * time_in_server
        client: n_users          = throughput * (time_in_server + think)

    `think` is a property of the workload, so it must not vary with load. If
    it climbs as concurrency rises, the load generator is not keeping `n_users`
    requests in flight -- which is precisely the failure that once moved N*
    from 128 to 32 when nvidia-smi polling starved the client, and which was
    diagnosed then only by accident.

    Compare `implied_think_s` across the levels of one sweep; a single level in
    isolation says little, because the true think time is not known here.
    """
    if throughput_rps <= 0:
        return {"available": False, "reason": "no throughput"}
    in_server = (batch + queued) / throughput_rps
    cycle = n_users / throughput_rps
    return {"available": True,
            "time_in_server_s": round(in_server, 2),
            "cycle_s": round(cycle, 2),
            "implied_think_s": round(cycle - in_server, 2),
            "server_utilisation_hint": round(in_server / cycle, 3)}


def littles_law_drift(levels: list[dict]) -> dict:
    """Implied think time across a sweep. It should be flat; drift is a bug.

    `levels` need `n_users`, `throughput_rps`, and a batch (running, and
    optionally queued).
    """
    rows = []
    for lv in levels:
        r = littles_law(lv["n_users"], lv["throughput_rps"], lv["batch"],
                        lv.get("queued", 0.0))
        if r.get("available"):
            rows.append({"n_users": lv["n_users"], **r})
    if len(rows) < 2:
        return {"available": False}
    th = [r["implied_think_s"] for r in rows]
    lo, hi = min(th), max(th)
    return {"available": True, "rows": rows,
            "think_min_s": lo, "think_max_s": hi,
            "drift": round(hi / lo, 2) if lo > 0 else None,
            "consistent": bool(lo > 0 and hi / lo < 1.5)}
