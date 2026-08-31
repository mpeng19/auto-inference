"""What could we sell this inference for?

The objective is an OpenRouter listing price, not a latency number. Getting
there needs three things the harness can measure and one it cannot:

  1. **The SLO frontier.** The largest offered load that still meets the
     marketplace's latency targets. Measured (`sweep_concurrency`).
  2. **Cost attribution.** How many GPU-seconds an uncached input token, a
     cached input token, and an output token each consume. Measured, by
     regression across workloads with deliberately different token mixes.
  3. **A cost basis.** Dollars per GPU-hour. *Not* measured — an input.
  4. **Utilisation.** What fraction of paid-for capacity carries traffic. Not
     measurable here at all; it depends on how much traffic the marketplace
     sends. It is the single largest lever on the answer, so it is an explicit
     parameter that must be stated with every result.

Modal's $3.95/hr is serverless retail and is deliberately excluded from the
default bases: nobody serving at scale pays it. The optimisation target is
therefore the **hardware-independent** quantity -- SLO-constrained tokens per
GPU-second -- and price is a reporting layer over it. A stack improvement shows
up in the former; a procurement decision only moves the latter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Reference costs, $/GPU-hour. Nebius list prices fetched 2026-08-30; the
# committed figure applies their stated "up to 35% off on-demand".
COST_BASES: dict[str, tuple[float, str]] = {
    "nebius-h100-committed": (2.50, "on-demand less 35% commitment discount"),
    "nebius-h100-preemptible": (2.15, "preemptible; cheapest, can be reclaimed"),
    "nebius-h100-ondemand": (3.85, "list on-demand"),
    "nebius-h200-committed": (2.93, "on-demand less 35%"),
    "nebius-h200-preemptible": (2.45, "preemptible"),
    "modal-h100": (3.95, "serverless retail -- for our own spend, NOT a serving cost basis"),
}
DEFAULT_BASIS = "nebius-h100-committed"


@dataclass(frozen=True)
class Observation:
    """One workload run at a known operating point.

    `cached_in` comes from SGLang's own `sglang:cached_tokens_total` counter,
    so the split between cached and uncached input is the server's accounting,
    not our inference from prompt text.
    """
    name: str
    uncached_in: float
    cached_in: float
    out: float
    gpu_seconds: float


@dataclass(frozen=True)
class Attribution:
    """GPU-seconds per token, by class. The ratio is the interesting part."""
    per_uncached_in: float
    per_cached_in: float
    per_out: float
    r2: float
    n: int
    residual: float

    @property
    def cache_discount(self) -> float | None:
        """How much cheaper a cached input token is than an uncached one.

        This is the number the whole business case rests on. The market prices
        cached input at up to 10x below input; if our ratio is near 1, we cannot
        follow, whatever the hit rate.
        """
        if self.per_uncached_in <= 0:
            return None
        return self.per_cached_in / self.per_uncached_in


def _nnls(A, b, iters: int = 500):
    """Non-negative least squares by multiplicative updates.

    Negative coefficients are physically meaningless here -- a token cannot
    give GPU-seconds back -- and an unconstrained fit will happily produce them
    when the workloads are collinear, which they partly are.
    """
    import numpy as np

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.full(A.shape[1], max(b.mean() / max(A.mean(), 1e-9), 1e-12))
    AtA, Atb = A.T @ A, A.T @ b
    for _ in range(iters):
        denom = AtA @ x
        x = x * Atb / np.maximum(denom, 1e-30)
        x = np.maximum(x, 0.0)
    return x


def attribute(obs: list[Observation]) -> Attribution:
    """Solve gpu_seconds = a*uncached_in + b*cached_in + c*out.

    One workload gives one equation and three unknowns, so this needs several
    workloads whose token mixes differ substantially -- `prefill_heavy`,
    `decode_heavy` and `prefix_heavy` are in the suite precisely because they
    span this space. Feeding it near-identical workloads yields a fit that looks
    fine and means nothing, which is why `r2` alone is not a sufficient check.
    """
    import numpy as np

    if len(obs) < 3:
        raise ValueError(f"need >=3 workloads with differing token mixes, got {len(obs)}")

    A = np.array([[o.uncached_in, o.cached_in, o.out] for o in obs], dtype=float)
    b = np.array([o.gpu_seconds for o in obs], dtype=float)
    x = _nnls(A, b)

    pred = A @ x
    ss_res = float(((b - pred) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return Attribution(float(x[0]), float(x[1]), float(x[2]),
                       round(r2, 4), len(obs), round(ss_res ** 0.5, 3))


def attribute_saturated(obs: list[Observation]) -> Attribution:
    """Solve for per-token cost from *rates*, not totals.

    Fixed-duration experiments cannot identify per-token costs. Every mix ran
    ~107 GPU-seconds because each was given the same 90s window, while token
    counts varied 35x -- so the regression was asked to fit a near-constant
    target from wildly varying features, which no linear model can do. Three
    successive attempts failed this way for what looked like three different
    reasons (r2 of -2.9, then zeros, then -36.7); the cause was the experimental
    design, not the denominator.

    At saturation the server is busy throughout, so what a mix reveals is its
    *throughput composition*: how many uncached, cached and output tokens per
    GPU-second. Dividing the cost equation through by GPU-seconds

        gpu_s = a*U + b*C + c*O   ->   1 = a*(U/gpu_s) + b*(C/gpu_s) + c*(O/gpu_s)

    leaves the same coefficients with duration cancelled. The target is now
    constant 1 by construction, so r2 is meaningless here and fit quality is
    the residual |1 - prediction| instead.
    """
    import numpy as np

    if len(obs) < 3:
        raise ValueError(f"need >=3 saturated mixes with differing "
                         f"compositions, got {len(obs)}")

    rows = []
    for o in obs:
        g = max(o.gpu_seconds, 1e-9)
        rows.append([o.uncached_in / g, o.cached_in / g, o.out / g])
    A = np.array(rows, dtype=float)
    b = np.ones(len(rows), dtype=float)

    x = _nnls(A, b)
    pred = A @ x
    resid = np.abs(pred - 1.0)
    # Reported in the r2 slot as 1 - worst relative residual, so a single
    # threshold still works: 1.0 is perfect, below 0.9 means >10% off.
    quality = float(1.0 - resid.max())
    return Attribution(float(x[0]), float(x[1]), float(x[2]),
                       round(quality, 4), len(obs), round(float(resid.max()), 4))


def cross_validate(obs: list[Observation]) -> dict:
    """Leave-one-out: fit on the rest, predict the held-out workload.

    In-sample r2 only says the fit reproduces the points it was fitted to. It
    cannot tell us whether the *structure* is right -- whether GPU time really
    is additive across token classes, or whether prefill and decode interact in
    ways a linear model cannot express. Held-out prediction can.

    This is the strongest fidelity check available without being a serving
    provider: it does not require knowing anyone's utilisation, margin or
    hardware cost. If held-out error is large while in-sample r2 is high, the
    model is memorising four points rather than describing the system.
    """
    if len(obs) < 4:
        return {"available": False,
                "reason": "need >=4 workloads to hold one out and still fit 3 unknowns"}

    rows, errs = [], []
    for i, held in enumerate(obs):
        rest = obs[:i] + obs[i + 1:]
        try:
            a = attribute(rest)
        except ValueError:
            continue
        pred = (a.per_uncached_in * held.uncached_in
                + a.per_cached_in * held.cached_in
                + a.per_out * held.out)
        err = abs(pred - held.gpu_seconds) / max(held.gpu_seconds, 1e-9)
        errs.append(err)
        rows.append({
            "held_out": held.name,
            "actual_gpu_s": round(held.gpu_seconds, 2),
            "predicted_gpu_s": round(pred, 2),
            "rel_error": round(err, 4),
            # How much the held-out point moved the cache ratio -- if dropping
            # one workload swings it wildly, the ratio is not identified.
            "cache_discount_without_it": (round(a.cache_discount, 3)
                                          if a.cache_discount is not None else None),
        })

    if not errs:
        return {"available": False, "reason": "no fold could be fitted"}
    worst, mean = max(errs), sum(errs) / len(errs)
    ratios = [r["cache_discount_without_it"] for r in rows
              if r["cache_discount_without_it"] is not None]
    spread = (max(ratios) - min(ratios)) if len(ratios) > 1 else None
    return {
        "available": True,
        "folds": rows,
        "mean_rel_error": round(mean, 4),
        "worst_rel_error": round(worst, 4),
        "cache_discount_spread": round(spread, 3) if spread is not None else None,
        # A model that predicts held-out GPU time to within ~20% is describing
        # the system; one that misses by 2x is fitting noise.
        "structure_holds": worst < 0.35,
        "ratio_stable": (spread is not None and spread < 0.25),
    }


MIN_R2 = 0.90


def usable(attr: Attribution, cond: dict) -> tuple[bool, str]:
    """Is this fit good enough to price from?

    Learned the hard way: attributing against *wall* GPU-seconds instead of
    compute-seconds produced r2 = -2.9 — worse than predicting the mean — and
    still yielded entirely plausible-looking prices ($0.11/M in, $0.31/M out).
    Plausible output from an invalid model is the failure mode that matters, so
    the gate is enforced rather than reported.
    """
    if attr.r2 < MIN_R2:
        return False, (f"r2 {attr.r2} < {MIN_R2}: the cost model does not "
                       f"describe these observations. A negative r2 means worse "
                       f"than predicting the mean — usually the wrong left-hand "
                       f"side (wall time rather than compute time).")
    if not cond["well_conditioned"]:
        return False, cond["note"]
    return True, "ok"


def conditioning(obs: list[Observation]) -> dict:
    """Are these workloads actually different enough to identify three costs?

    Collinear observations give a confident-looking fit with arbitrary
    coefficients. Reported alongside every attribution so a degenerate design
    is visible rather than silent.
    """
    import numpy as np

    A = np.array([[o.uncached_in, o.cached_in, o.out] for o in obs], dtype=float)
    scale = A.max(axis=0)
    scale[scale == 0] = 1.0
    cond = float(np.linalg.cond(A / scale))
    return {
        "condition_number": round(cond, 1),
        "well_conditioned": cond < 50,
        "note": ("condition number over ~50 means the workloads are too alike to "
                 "separate the three token costs; add a workload with a "
                 "different in/out or cached/uncached ratio"),
    }


def prices(attr: Attribution, basis: str = DEFAULT_BASIS, n_gpu: int = 1,
           utilization: float = 0.6, margin: float = 0.25) -> dict:
    """Convert GPU-seconds per token into $/M tokens.

    `utilization` is the fraction of paid capacity actually serving traffic.
    Idle GPUs still bill, so cost per token scales as 1/utilisation -- at 40%
    versus 80% the price doubles for an identical stack. There is no way to
    measure it here; it is a property of the traffic a marketplace sends.
    """
    if basis not in COST_BASES:
        raise KeyError(f"unknown cost basis {basis!r}; have {sorted(COST_BASES)}")
    usd_hr, note = COST_BASES[basis]
    usd_per_gpu_second = usd_hr / 3600.0
    # Attribution is measured in GPU-seconds already aggregated over the node,
    # so n_gpu multiplies the dollar rate, not the token cost.
    rate = usd_per_gpu_second * n_gpu

    def to_price(gpu_s_per_token: float) -> float:
        cost = gpu_s_per_token * rate / max(utilization, 1e-9)
        return cost * (1.0 + margin) * 1e6      # $/M tokens

    # Full precision on purpose. `effective_in` blends two of these, and at a
    # 95% hit rate a cached price rounded to 4dp carries most of the weight --
    # rounding in the data layer would put the error where it hurts most.
    # Round at display time instead.
    return {
        "basis": basis, "usd_per_gpu_hour": usd_hr, "basis_note": note,
        "n_gpu": n_gpu, "utilization": utilization, "margin": margin,
        "price_in_per_mtok": to_price(attr.per_uncached_in),
        "price_cached_in_per_mtok": to_price(attr.per_cached_in),
        "price_out_per_mtok": to_price(attr.per_out),
        "cache_discount": attr.cache_discount,
    }


def fmt_prices(p: dict, nd: int = 4) -> dict:
    """Display-rounded copy. Never feed this back into arithmetic."""
    out = dict(p)
    for k in ("price_in_per_mtok", "price_cached_in_per_mtok",
              "price_out_per_mtok", "cache_discount"):
        if out.get(k) is not None:
            out[k] = round(out[k], nd)
    return out


def effective_in(price_in: float, price_cached: float, hit_rate: float) -> float:
    """Blended input price at a given cache hit rate -- the marketplace metric."""
    return hit_rate * price_cached + (1.0 - hit_rate) * price_in


def rank_vs_market(eff: float, market: list[tuple[str, float, float, float | None]],
                   hit_rate: float) -> dict:
    """Where would we sit on the live provider table?

    `market` rows are (provider, input $/M, output $/M, cache_read $/M).
    """
    scored = []
    for name, pin, pout, pcache in market:
        e = pin if pcache is None else effective_in(pin, pcache, hit_rate)
        scored.append((name, e))
    scored.sort(key=lambda r: r[1])
    rank = sum(1 for _, e in scored if e < eff) + 1
    return {
        "hit_rate": hit_rate,
        "our_effective_in_per_mtok": round(eff, 5),
        "rank": rank,
        "of": len(scored) + 1,
        "best_competitor": scored[0][0],
        "best_competitor_price": round(scored[0][1], 4),
        "table": [{"provider": n, "eff_in": round(e, 4)} for n, e in scored],
    }
