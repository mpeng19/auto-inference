"""Figures written into a run's `root_dir`. One question per file.

Deliberately plain matplotlib. These are working plots read by people deciding
what to optimise next, not published exhibits, and the default look is the one
everyone already knows how to read.

**The SLO plots put performance on y and the constraint on x.** That is the
whole tradeoff in one frame: every point is a concurrency level, moving right
buys throughput and spends latency, and the shaded region is where the SLO
stops you. Plotting the metric against concurrency instead -- which an earlier
version did -- hides the thing you actually want, because concurrency is a knob
and throughput is the payoff.

Each figure stands alone at full size rather than sharing a canvas: three
panels crammed into one image are three plots nobody can read.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless: these render on a GPU box with no display

# Imported after the backend choice above, deliberately.
import matplotlib.pyplot as plt

DPI = 150
FIGSIZE = (8.0, 5.4)
SERIES = "#1f77b4"        # matplotlib default blue
LIMIT = "#d62728"         # limit lines and the violating region
HELD = "#2ca02c"          # the last level that held
LABEL = "#444444"


def _fig():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _points(ax, xs, ys, ns):
    """The curve, its markers, and a small concurrency label on each."""
    ax.plot(xs, ys, "-o", color=SERIES, lw=1.8, markersize=6, zorder=3)
    for x, y, n in zip(xs, ys, ns, strict=True):
        ax.annotate(str(n), (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=LABEL)


def _violating(ax, limit):
    """Shade where the SLO is broken. Every bound here is an upper bound."""
    ax.axvline(limit, color=LIMIT, lw=1.5, ls="--", zorder=2,
               label=f"SLO limit {limit:g}")
    lo, hi = ax.get_xlim()
    ax.axvspan(limit, max(hi, limit * 1.02), color=LIMIT, alpha=0.06, zorder=1)
    ax.set_xlim(lo, hi)


def _held(ax, x, y, n, tps):
    ax.plot([x], [y], "o", markersize=15, markerfacecolor="none",
            markeredgecolor=HELD, markeredgewidth=2, zorder=6,
            label=f"last held: N={n}, {tps:.0f} tok/s/gpu")


def _save(fig, ax, path, sim, res, title, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("aggregate output tok/s/gpu")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="best")
    fig.text(0.01, 0.005, _caption(sim, res), fontsize=7, color=LABEL)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def slo_plot(sim, res, bound, path: pathlib.Path) -> str:
    """Aggregate throughput against one SLO metric, one point per concurrency."""
    pts = [p for p in res.curve
           if next((c["value"] for c in p.checks if c["label"] == bound.label), None)
           is not None]
    xs = [next(c["value"] for c in p.checks if c["label"] == bound.label) for p in pts]
    ys = [p.output_tps_per_gpu for p in pts]
    ns = [p.n_users for p in pts]

    fig, ax = _fig()
    _points(ax, xs, ys, ns)
    _violating(ax, bound.limit_ms)
    star = res.best
    if star is not None and star in pts:
        i = pts.index(star)
        _held(ax, xs[i], ys[i], star.n_users, star.output_tps_per_gpu)

    failing = [p for p in res.curve if not p.meets_slo]
    binds = bool(failing) and min(failing, key=lambda p: p.n_users).binding == bound.label
    verdict = ("binds first — this sets N*" if binds
               else ("violated, but only past N*" if any(x > bound.limit_ms for x in xs)
                     else "never violated at these levels"))
    return _save(fig, ax, path, sim, res,
                 f"throughput vs {bound.label}  —  {verdict}",
                 f"{bound.label} (ms)")


def price_vs_share(sim, res, path: pathlib.Path) -> str:
    """What each concurrency costs, against the share one node can serve there."""
    m = res.market
    xs = [p.share_per_node * 100 for p in res.curve]
    ys = [p.bill_per_1k for p in res.curve]
    ns = [p.n_users for p in res.curve]

    fig, ax = _fig()
    ax.plot(xs, ys, "-", color=SERIES, lw=1.8, zorder=3)
    for p, x, y, n in zip(res.curve, xs, ys, ns, strict=True):
        ax.plot([x], [y], "o", markersize=6, color=SERIES,
                markerfacecolor=SERIES if p.meets_slo else "white",
                markeredgecolor=SERIES, markeredgewidth=1.6, zorder=4)
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=LABEL)
    best = min(m.providers,
               key=lambda q: m.bill_per_1k(q.effective_in_per_m, q.listed_out_per_m))
    ref = m.bill_per_1k(best.effective_in_per_m, best.listed_out_per_m)
    # Red means "the SLO forbids operating here", nothing else. Shading
    # everything above the cheapest provider would paint the whole chart and
    # say only that we are expensive, which the y axis already says.
    ax.axhline(ref, color=LIMIT, lw=1.5, ls="--", zorder=2,
               label=f"cheapest provider: {best.name} ${ref:.2f}/1k")
    star = res.best
    if star is not None:
        lo, hi = ax.get_xlim()
        ax.axvspan(star.share_per_node * 100, hi, color=LIMIT, alpha=0.06, zorder=1)
        ax.set_xlim(lo, hi)
    if star is not None:
        ax.plot([star.share_per_node * 100], [star.bill_per_1k], "o", markersize=15,
                markerfacecolor="none", markeredgecolor=HELD, markeredgewidth=2,
                zorder=6, label=f"last held: N={star.n_users}, "
                                f"${star.bill_per_1k:.2f}/1k")
    ax.set_xlabel("market share ONE node can serve (%)")
    ax.set_ylabel("$ per 1,000 requests")
    ax.set_title("Price falls with concurrency — until the SLO stops it  "
                 "(hollow = violates SLO)", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    fig.text(0.01, 0.005, _caption(sim, res), fontsize=7, color=LABEL)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def price_vs_demand(sim, res, path: pathlib.Path) -> str:
    """Below saturation the node idles and price is a hyperbola in share."""
    import numpy as np

    from ..price.market import Economics
    m, star = res.market, res.best
    if star is None:
        raise ValueError("no priced point")
    e = Economics(gpu_s_per_request=star.gpu_s_per_request, n_gpu=sim.n_gpu,
                  rate_per_gpu_hour=sim.rate_per_gpu_hour, utilisation=sim.util)
    smax = star.share_per_node
    s = np.logspace(np.log10(max(smax / 30, 1e-5)), np.log10(smax * 12), 400)
    y = [e.price_at_share(x, m) for x in s]

    fig, ax = _fig()
    ax.plot(s * 100, y, "-", color=SERIES, lw=1.8, zorder=3)
    ax.plot([smax * 100], [star.bill_per_1k], "o", markersize=15,
            markerfacecolor="none", markeredgecolor=HELD, markeredgewidth=2,
            zorder=6, label=f"one node saturates: {smax*100:.2f}% share, "
                            f"${star.bill_per_1k:.2f}/1k")
    # Grey, not red: idling is an economic problem, not an SLO violation, and
    # red is reserved for "you may not operate here".
    ax.axvspan(s[0] * 100, smax * 100, color="#888888", alpha=0.10, zorder=1)
    ax.axvline(smax * 100, color="#555555", lw=1.5, ls="--", zorder=2,
               label="left of here the node idles")
    best = min(m.providers,
               key=lambda q: m.bill_per_1k(q.effective_in_per_m, q.listed_out_per_m))
    ref = m.bill_per_1k(best.effective_in_per_m, best.listed_out_per_m)
    ax.axhline(ref, color="#888888", lw=1.2, ls=":", zorder=2,
               label=f"{best.name} ${ref:.2f}/1k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(_num))
    ax.set_xlabel("market share actually captured (%, log)")
    ax.set_ylabel("$ per 1,000 requests (log)")
    ax.set_title("Below the kink, price is set by demand, not by the stack",
                 fontsize=11)
    ax.legend(fontsize=9, loc="best")
    fig.text(0.01, 0.005, _caption(sim, res), fontsize=7, color=LABEL)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def _num(v, _pos=None):
    """Log axes with plain numbers: 0.05, 0.3, 3 — never 10^-1."""
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:g}" if v >= 1 else f"{v:.3g}"


def _caption(sim, res) -> str:
    m = res.market or sim.market
    return (f"{sim.model} · {sim.n_gpu}x{sim.gpu} · {sim.cost_basis} · "
            f"utilisation {sim.util:.0%} · break-even · stack {sim.stack.describe()[:50]}\n"
            f"SLO {sim.slo.describe()} · market {m.requests_per_day:,.0f} req/day at "
            f"{m.in_per_request:,.0f} in / {m.out_per_request:,.0f} out per request")


def _slug(label: str) -> str:
    return label.lower().replace(" ", "-")


def render_all(sim, res, root: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not res.curve:
        return out
    for b in sim.slo.bounds:
        key = f"slo_{_slug(b.label).replace('-', '_')}"
        out[key] = slo_plot(sim, res, b, root / f"slo-{_slug(b.label)}.png")
    if res.market is not None:
        out["price_vs_share"] = price_vs_share(sim, res, root / "price-vs-share.png")
        if res.best is not None:
            out["price_vs_demand"] = price_vs_demand(
                sim, res, root / "price-vs-demand.png")
    return out
