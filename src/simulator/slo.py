"""Latency targets, and the arithmetic of judging a level against them.

Two things here are not obvious and were both learned the expensive way.

**A bound names its own order statistic.** TTFT and TPOT do not live at the
same quantile. Published market latency gives a usable p90 for TTFT, but
throughput percentiles run backwards -- `TPOT = 1/throughput`, so `p99` of
throughput is the *fastest* 1%, mapping to p1 TPOT -- and nobody publishes the
slow tail. So the market pins a TTFT tail and only a TPOT middle. An SLO that
forces both metrics onto one percentile cannot express that, and the version
that did sent us chasing a p90 TTFT bound of 300 ms that no provider on the
board meets and that our own workload cannot reach at any load.

**A percentile needs samples to be a percentile.** At 45 completions in a
window, "p99" is the single worst request -- we measured p99 TTFT 1017.9 ms
against a max of 1030.7 ms. A frontier decided by one request is not a
frontier, so `judge` reports under-sampling rather than quietly returning a
number that looks like a tail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STATS = ("mean", "p50", "p90", "p95", "p99", "max")


@dataclass(frozen=True)
class Bound:
    """One constraint: `<stat>` of `<metric>` must not exceed `limit_ms`."""
    metric: str          # "ttft" | "tpot"
    stat: str            # one of STATS
    limit_ms: float
    note: str = ""

    def __post_init__(self):
        if self.metric not in ("ttft", "tpot"):
            raise ValueError(f"metric must be ttft or tpot, got {self.metric!r}")
        if self.stat not in STATS:
            raise ValueError(f"stat must be one of {STATS}, got {self.stat!r}")

    @property
    def label(self) -> str:
        return f"{self.stat} {self.metric.upper()}"

    @classmethod
    def parse(cls, spec: str) -> Bound:
        """`'ttft:p90:2818'` -> Bound. For CLI and config files."""
        metric, stat, limit = spec.split(":")
        return cls(metric.strip().lower(), stat.strip().lower(), float(limit))

    def min_samples(self) -> int:
        """Completions needed before this stat is meaningfully estimated.

        Three observations beyond the quantile: at p90 that is 30, at p99 it is
        300. Means and medians are fine at any sample size we run.
        """
        if not self.stat.startswith("p"):
            return 0
        q = int(self.stat[1:])
        return 0 if q <= 50 else round(3 / (1 - q / 100))


# Measured from OpenRouter's own published percentiles for this model
# (`price.market`), not guessed, except where noted.
MARKET_SLO: tuple[Bound, ...] = (
    Bound("ttft", "p90", 2818.0, "median provider's published p90 latency"),
    Bound("tpot", "p90", 25.0, "interactive-serving guidance -- our choice"),
    Bound("tpot", "mean", 20.0, "median provider's published p50 throughput"),
)


@dataclass
class Verdict:
    """Did a level hold, and if not, which bound gave way."""
    ok: bool
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def binding(self) -> str | None:
        """The bound that failed by the largest relative margin."""
        bad = [c for c in self.checks if not c["ok"] and c["value"] is not None]
        if not bad:
            return None
        return max(bad, key=lambda c: c["value"] / c["limit"])["label"]


@dataclass(frozen=True)
class SLO:
    """A set of bounds, all of which must hold, plus an error floor."""
    bounds: tuple[Bound, ...] = MARKET_SLO
    # A blow-up detector, not a second SLO. The per-request test uses the
    # loosest bound on each metric (see per_request_limits), and a p90 bound
    # already tolerates 10% of requests over that limit -- so 0.90 is the floor
    # the percentile bounds imply, and anything below it is stragglers the
    # percentiles cannot see (joint TTFT+TPOT misses, or collapse). 0.99 was
    # wrong: one straggler in 37 completions rejected a level whose p90 sat
    # 30% inside its limit, and 0.99 is unmeasurable below ~100 samples.
    min_good_frac: float = 0.90
    max_failed: int = 0

    @classmethod
    def parse(cls, specs: str, **kw) -> SLO:
        """`'ttft:p90:2818,tpot:mean:20'` -> SLO."""
        return cls(bounds=tuple(Bound.parse(s) for s in specs.split(",") if s.strip()),
                   **kw)

    def judge(self, level: dict) -> Verdict:
        """Score one measured level. `level` is a row from the sweep record.

        One caveat that matters when re-judging a stored sweep: the percentile
        bounds are recomputed from the stored statistics and are exact, but
        `good_frac` was computed **at run time** against whatever SLO the sweep
        used, and cannot be recovered without per-request records. It is kept
        as a blow-up detector -- a level at 0.97 had real stragglers under any
        threshold -- but do not read it as evidence about *this* SLO.
        """
        v = Verdict(ok=True)
        n = (level.get("ttft_ms") or {}).get("n", 0)
        for b in self.bounds:
            val = (level.get(f"{b.metric}_ms") or {}).get(b.stat)
            ok = val is not None and val <= b.limit_ms
            v.ok = v.ok and ok
            v.checks.append({"label": b.label, "stat": b.stat, "metric": b.metric,
                             "value": val, "limit": b.limit_ms, "ok": ok,
                             "note": b.note})
            need = b.min_samples()
            if need and n and n < need:
                v.warnings.append(
                    f"{b.label} estimated from {n} completions; "
                    f"~{need} needed before it is a percentile rather than a maximum")
        gf = level.get("good_frac")
        if gf is not None and gf < self.min_good_frac:
            v.ok = False
            v.checks.append({"label": "good_frac", "stat": "frac", "metric": "slo",
                             "value": gf, "limit": self.min_good_frac, "ok": False,
                             "note": "share of requests meeting every bound"})
        if level.get("n_failed", 0) > self.max_failed:
            v.ok = False
            v.warnings.append(f"{level['n_failed']} request(s) errored")
        return v

    def per_request_limits(self) -> tuple[float, float]:
        """(ttft_ms, tpot_ms) for judging a *single* request.

        The **loosest** bound on each metric, deliberately. `good_frac` is a
        blow-up detector -- it catches the congestion-collapse regime where
        TTFT runs to 35 s -- not a second copy of the percentile test. Using
        the tightest bound instead would fail ~40% of requests at a level whose
        p90 sits comfortably inside its limit, and would double-count the
        constraint the percentile bounds already enforce.
        """
        inf = float("inf")
        t = [b.limit_ms for b in self.bounds if b.metric == "ttft"]
        p = [b.limit_ms for b in self.bounds if b.metric == "tpot"]
        return (max(t) if t else inf, max(p) if p else inf)

    def describe(self) -> str:
        return "  and  ".join(f"{b.label} <= {b.limit_ms:g} ms" for b in self.bounds)

    def as_dict(self) -> dict:
        return {"bounds": [{"metric": b.metric, "stat": b.stat,
                            "limit_ms": b.limit_ms, "note": b.note}
                           for b in self.bounds],
                "min_good_frac": self.min_good_frac,
                "max_failed": self.max_failed}

    @classmethod
    def from_dict(cls, d: dict) -> SLO:
        return cls(bounds=tuple(Bound(x["metric"], x["stat"], x["limit_ms"],
                                      x.get("note", "")) for x in d["bounds"]),
                   min_good_frac=d.get("min_good_frac", 0.90),
                   max_failed=d.get("max_failed", 0))
