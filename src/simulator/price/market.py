"""What the market actually charges, and what share a deployment can serve.

Two kinds of thing live here and they must not be confused.

**Listed prices** are what a provider posts. **Realised effective prices**
are what buyers actually pay once prompt caching is accounted for, which
OpenRouter publishes alongside each provider's realised hit rate. Ranking
uses the realised figures at each provider's OWN hit rate: re-blending
everyone to a common rate would erase the thing being optimised, since
Novita realises 87.4% and Cloudflare 0.0% on the same model and the same
traffic (HANDOFF SS5e).

The share economics are the other half. A price quoted without the demand
it assumes is meaningless: below saturation, utilisation is set by demand
rather than chosen, and price runs as 1/share regardless of how good the
serving stack is. The stack sets only where the curve stops falling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
import pathlib


# qwen/qwen3.8-27b provider table, from the OpenRouter model page 2026-08-30.
# (provider, input $/M, output $/M, cache read $/M)
MARKET_QWEN38_27B = [
    ("Reka", 0.350, 2.550, 0.050), ("AkashML", 0.350, 2.550, 0.050),
    ("Chutes", 0.350, 2.750, 0.035), ("Parasail", 0.350, 3.200, 0.050),
    ("Phala", 0.400, 3.000, 0.150), ("CoreWeave", 0.400, 3.000, 0.150),
    ("Novita", 0.420, 3.000, 0.085), ("Alibaba", 0.425, 2.550, 0.085),
    ("Cloudflare", 0.450, 3.200, 0.050), ("Venice", 0.450, 3.200, None),
    ("Io Net", 0.480, 3.400, 0.250),
]

# What providers *actually* realise, which OpenRouter publishes alongside the
# list prices. This is ground truth for two things we previously had to assume:
#
#   1. The effective-price formula. eff = h*cache_read + (1-h)*listed reproduces
#      their published effective price to within 0.1% on 9 of 11 providers.
#   2. Achievable cache hit rate. Not 95% -- the range is 0% to 82%, and it is
#      plainly a *serving-system* property: Novita realises 81.8% and Chutes
#      69.5% while Venice and Cloudflare realise 0.0% on the same model and the
#      same marketplace traffic. Failing to implement prompt caching costs them
#      3x on effective input price.
#
# Dated snapshots of what providers *realise*, which OpenRouter publishes but
# does not expose through the public API (`/api/v1/models/.../endpoints` gives
# listed prices and uptime only). Captured by hand from the model page.
#
# Two snapshots are kept because the comparison is evidence in its own right:
# listed prices did not move at all between them, so every change in effective
# price is a change in *cache hit rate*. The "price history" chart on the model
# page is therefore a cache-hit-rate history in disguise.
#
# (provider, effective in $/M, realised cache hit rate, share of 1d tokens)
MARKET_SNAPSHOTS = {
    "2026-08-29": [
        ("Chutes", 0.1310, 0.695, 0.095), ("Novita", 0.1459, 0.818, 0.143),
        ("Alibaba", 0.1961, 0.662, 0.132), ("Parasail", 0.2263, 0.412, 0.053),
        ("Phala", 0.2281, 0.688, 0.206), ("AkashML", 0.2285, 0.405, 0.104),
        ("CoreWeave", 0.2336, 0.665, 0.067), ("Reka", 0.2622, 0.293, 0.167),
        ("Venice", 0.4500, 0.000, 0.017), ("Cloudflare", 0.4500, 0.000, 0.009),
        ("Io Net", 0.4624, 0.076, 0.006),
    ],
    "2026-08-31": [
        ("Novita", 0.1272, 0.874, 0.157), ("Chutes", 0.1439, 0.654, 0.075),
        ("Parasail", 0.1773, 0.575, 0.101), ("CoreWeave", 0.1988, 0.805, 0.101),
        ("Alibaba", 0.2160, 0.613, 0.122), ("Reka", 0.2257, 0.414, 0.130),
        ("Phala", 0.2526, 0.589, 0.126), ("AkashML", 0.2725, 0.258, 0.159),
        ("Venice", 0.4499, 0.000, 0.022), ("Cloudflare", 0.4499, 0.000, 0.004),
        ("Io Net", 0.4645, 0.067, 0.005),
    ],
}

MARKET_AS_OF = "2026-08-31"

MARKET_REALISED = MARKET_SNAPSHOTS[MARKET_AS_OF]

# Weighted average price actually paid across the market.
MARKET_WEIGHTED_IN = 0.2116

MARKET_WEIGHTED_OUT = 2.868

# The target: beat the best realised effective input price. This MOVES -- it was
# Chutes at $0.1310 two days earlier, and the leader changed because Chutes' hit
# rate fell 69.5%->65.4% while Novita's rose 81.8%->87.4%. Nobody changed a
# posted price. Any "we beat the market" claim is against a moving target.
MARKET_BEST_EFF_IN = 0.1272          # Novita, 87.4% hit

# Real traffic for this model is overwhelmingly input-dominated: 17.6B prompt
# tokens against 448M completion + 508M reasoning in one day, so roughly 18:1
# counting reasoning as output, 39:1 counting only completion. Our synthetic
# workloads run about 2:1, which models chat rather than the agentic coding
# traffic this model actually serves -- the top apps on it are pi, Hermes
# Agent, Claude Code and DeepSeek Harness.
#
# This matters for the objective: at 18:1, input is roughly 60% of revenue at
# market prices, so effective *input* price really is the number that decides
# competitiveness, and prefill is where the cost sits.
MARKET_INPUT_OUTPUT_RATIO = 18.4


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


def burst_utilisation(daily_volumes: list[float], sigma: float = 2.0) -> dict:
    """Utilisation a single-model deployment achieves, sized for peak.

    A provider serving one model must provision for its peak or shed traffic
    at the peak, so it runs at roughly mean/peak. Measured on this model's
    OpenRouter volume (17 days) that is **48%** -- derived from the traffic,
    not assumed.

    A multi-model fleet does better for a reason that has nothing to do with
    how much of *this* model it serves: GPUs are fungible, so what matters is
    fleet utilisation, and uncorrelated demands add in quadrature. With N
    similar, weakly-correlated models the aggregate coefficient of variation
    falls as 1/sqrt(N), so peak/mean approaches 1 and utilisation approaches
    100%. **That, not scale in any single model, is the incumbent advantage.**

        models on fleet    aggregate cv    peak/mean    utilisation
                      1            0.51         2.03            49%
                     10            0.16         1.33            75%
                    100            0.05         1.10            91%

    Two caveats, both making our 48% optimistic: the series is daily, so
    intra-day bursts are invisible; and real model demands are positively
    correlated (they share business hours), so a fleet's benefit is smaller
    than 1/sqrt(N) suggests -- but so is ours if we ever multiplex.
    """
    import statistics as st
    if len(daily_volumes) < 3:
        return {"available": False, "reason": "need >= 3 days"}
    mean = st.mean(daily_volumes)
    peak = max(daily_volumes)
    cv = st.pstdev(daily_volumes) / mean
    return {"available": True, "mean": mean, "peak": peak,
            "peak_over_mean": round(peak / mean, 2), "cv": round(cv, 3),
            "single_model_utilisation": round(mean / peak, 3),
            "sigma": sigma}


def fleet_utilisation(cv_single: float, n_models: int, sigma: float = 2.0) -> float:
    """Utilisation an N-model fleet holds, given one model's burstiness."""
    cv = cv_single / (n_models ** 0.5)
    return 1.0 / (1.0 + sigma * cv)


# ── the market snapshot, as scraped ──────────────────────────────────────

SNAPSHOT_NAME = "market-qwen-qwen3.8-27b.json"


def _find_snapshot() -> pathlib.Path:
    """The market snapshot is data, not code, so it may sit outside the package.

    Checked in order: an explicit AUTOINF_MARKET_DATA, the repo checkout, and
    the working directory. Failing all three, say which paths were tried --
    a missing denominator silently turning into a wrong market share would be
    much worse than a stack trace.
    """
    import os
    tried = []
    env = os.environ.get("AUTOINF_MARKET_DATA")
    if env:
        tried.append(pathlib.Path(env))
    here = pathlib.Path(__file__).resolve()
    tried += [here.parents[3] / "data" / SNAPSHOT_NAME,
              pathlib.Path.cwd() / "data" / SNAPSHOT_NAME]
    for t in tried:
        if t.is_file():
            return t
    raise FileNotFoundError(
        "no market snapshot found; tried " + ", ".join(str(t) for t in tried)
        + ".  Refresh it with `make market`, or set AUTOINF_MARKET_DATA.")


@dataclass(frozen=True)
class Provider:
    name: str
    effective_in_per_m: float      # realised, at their own hit rate
    hit_rate: float
    listed_out_per_m: float
    token_share: float


@dataclass(frozen=True)
class Market:
    """Everything about the demand side that a price has to be quoted against."""
    slug: str
    as_of: str
    requests_per_day: float
    in_per_request: float
    out_per_request: float
    providers: tuple[Provider, ...]
    utilisation_ceiling: float
    daily_requests: tuple[float, ...] = ()

    @classmethod
    def load(cls, path: str | pathlib.Path | None = None,
             window_days: int = 7) -> "Market":
        """Read a `market_pull` snapshot.

        `window_days` is the trailing window used for the demand denominator.
        Daily volume swings 829k-1.88M requests on this model, so a single day
        is not a denominator and the 17-day mean lags real growth; a trailing
        week is the compromise, and the full series is kept so the choice can
        be revisited without re-scraping.
        """
        d = json.loads(pathlib.Path(path or _find_snapshot()).read_text())
        s = d["summary"]
        daily = [float(r["count"]) for r in d.get("daily", [])]
        win = daily[-window_days:] if daily else []
        listed_out = {n: o for n, _, o, _ in MARKET_QWEN38_27B}
        provs = tuple(
            Provider(n, e, h, listed_out.get(n, MARKET_WEIGHTED_OUT), sh)
            for n, e, h, sh in MARKET_REALISED)
        return cls(
            slug=d.get("slug", "unknown"), as_of=MARKET_AS_OF,
            requests_per_day=(sum(win) / len(win)) if win
            else s["requests"] / max(s["days"], 1),
            in_per_request=s["in_per_request"], out_per_request=s["out_per_request"],
            providers=provs,
            utilisation_ceiling=(burst_utilisation(daily)["single_model_utilisation"]
                                 if len(daily) >= 3 else 0.50),
            daily_requests=tuple(daily))

    # ── scoring a price against the board ────────────────────────────────
    def bill_per_1k(self, eff_in_per_m: float, out_per_m: float) -> float:
        """What a buyer pays for 1,000 average requests. The honest score."""
        return (eff_in_per_m * self.in_per_request
                + out_per_m * self.out_per_request) / 1e6 * 1000

    def blended_per_m(self, eff_in_per_m: float, out_per_m: float) -> float:
        return self.bill_per_1k(eff_in_per_m, out_per_m) / 1000 / (
            self.in_per_request + self.out_per_request) * 1e6

    def leaderboard(self, eff_in_per_m: float, out_per_m: float,
                    label: str = "us") -> list[dict]:
        """Providers plus us, ranked by whole bill AND by effective input.

        Both, always, because they disagree: on the 1xH100 baseline we are 1st
        on effective input and 9th on the bill. OpenRouter sorts on the former
        and buyers pay the latter, so quoting one without the other is a way to
        be accidentally dishonest.
        """
        rows = [{"provider": p.name, "eff_in": p.effective_in_per_m,
                 "out": p.listed_out_per_m, "hit": p.hit_rate, "us": False,
                 "bill_1k": self.bill_per_1k(p.effective_in_per_m, p.listed_out_per_m)}
                for p in self.providers]
        rows.append({"provider": label, "eff_in": eff_in_per_m, "out": out_per_m,
                     "hit": None, "us": True,
                     "bill_1k": self.bill_per_1k(eff_in_per_m, out_per_m)})
        by_bill = sorted(rows, key=lambda r: r["bill_1k"])
        by_eff = sorted(rows, key=lambda r: r["eff_in"])
        for i, r in enumerate(by_bill, 1):
            r["rank_bill"] = i
        for i, r in enumerate(by_eff, 1):
            r["rank_eff_in"] = i
        return by_bill


# ── deployment economics: price and share are one object ─────────────────

@dataclass(frozen=True)
class Economics:
    """The SS6d formula. Price and capacity derive from one measured quantity.

    `gpu_s_per_request` is GPU-seconds of *forward time* for one market-sized
    request, summed across the node. Everything else follows, and the identity
    `price_per_request x capacity_per_node_per_day == n_gpu x 24 x rate` holds
    exactly -- which is the check that nothing double-counts utilisation.

    Deriving capacity from measured goodput instead is wrong at low
    concurrency: goodput counts only requests that finish inside the window, so
    a level whose requests each take 90 s is undercounted, and the 1xH100 N=4
    level came out 2.2x off.
    """
    gpu_s_per_request: float
    n_gpu: int = 1
    rate_per_gpu_hour: float = 3.00
    utilisation: float = 0.50

    @property
    def price_per_1k(self) -> float:
        return (self.gpu_s_per_request * self.rate_per_gpu_hour / 3600
                / self.utilisation * 1000)

    def capacity_per_node_per_day(self) -> float:
        return 86400 * self.n_gpu * self.utilisation / self.gpu_s_per_request

    def share_per_node(self, m: Market) -> float:
        return self.capacity_per_node_per_day() / m.requests_per_day

    def nodes_for_share(self, share: float, m: Market) -> float:
        return share * m.requests_per_day / self.capacity_per_node_per_day()

    def price_at_share(self, share: float, m: Market, nodes: int = 1) -> float:
        """$/1k at a share the fleet may be too big for.

        Below saturation the node idles, utilisation is whatever demand grants,
        and price is a pure hyperbola that the serving stack does not enter:

            $/request = nodes x n_gpu x 24 x rate / (share x M)

        Above it, add nodes and the price is flat. The stack sets the kink.
        """
        sat = self.share_per_node(m) * nodes
        if share <= 0:
            return float("inf")
        if share >= sat:
            return self.price_per_1k
        return nodes * self.n_gpu * 24 * self.rate_per_gpu_hour / (
            share * m.requests_per_day) * 1000
