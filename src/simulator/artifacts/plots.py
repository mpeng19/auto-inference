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

**Fitted as a power law, not joined by segments.** Concurrency is swept
multiplicatively (1, 2, 4, 8, 16, 32) and the relationships here saturate;
straight lines between measured points assert a linearity the data does not
have. The exponent is reported with the fit, because it *is* the finding --
`b < 1` on throughput-versus-latency is the shape of a system running out of
headroom.

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


def powerlaw(xs, ys):
    """Fit y = a * x**b by least squares in log space.

    A power law, not a line: throughput against a latency bound is a
    saturating relationship over a multiplicative range of concurrency, and
    straight segments between measured points imply a linearity that is not
    there. Returns (a, b, r2) or None when the data cannot support a fit.
    """
    import numpy as np

    x = np.asarray([v for v in xs], dtype=float)
    y = np.asarray([v for v in ys], dtype=float)
    ok = (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(set(x.tolist())) < 3:
        return None
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    pred = la + b * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(np.exp(la)), float(b), r2


def _fit_curve(ax, xs, ys, label_prefix="") -> str:
    """Draw the fitted power law across the measured range. Returns a legend label."""
    import numpy as np

    fit = powerlaw(xs, ys)
    if fit is None:
        ax.plot(xs, ys, "-", color=SERIES, lw=1.2, alpha=0.5, zorder=2)
        return ""
    a, b, r2 = fit
    grid = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 200)
    ax.plot(grid, a * grid ** b, "-", color=SERIES, lw=1.8, zorder=2)
    return f"{label_prefix}fit y = {a:.3g}·x^{b:.2f}  (r²={r2:.3f})"


def _points(ax, xs, ys, ns):
    """Markers and a concurrency label on each, placed clear of the curve.

    Labels alternate above and below: with a fitted curve running through the
    markers, a fixed offset puts every label on the line it is meant to
    annotate.
    """
    ax.plot(xs, ys, "o", color=SERIES, markersize=6, zorder=4)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    for rank, i in enumerate(order):
        above = rank % 2 == 0
        ax.annotate(f"N={ns[i]}", (xs[i], ys[i]), textcoords="offset points",
                    xytext=(0, 11 if above else -17), ha="center",
                    fontsize=8, color=LABEL)


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
    fit_label = _fit_curve(ax, xs, ys)
    _points(ax, xs, ys, ns)
    _violating(ax, bound.limit_ms)
    if fit_label:
        ax.plot([], [], "-", color=SERIES, lw=1.8, label=fit_label)
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
    fit_label = _fit_curve(ax, xs, ys)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    for rank, i in enumerate(order):
        pt = res.curve[i]
        ax.plot([xs[i]], [ys[i]], "o", markersize=6, color=SERIES,
                markerfacecolor=SERIES if pt.meets_slo else "white",
                markeredgecolor=SERIES, markeredgewidth=1.6, zorder=4)
        ax.annotate(f"N={ns[i]}", (xs[i], ys[i]), textcoords="offset points",
                    xytext=(0, 11 if rank % 2 == 0 else -17), ha="center",
                    fontsize=8, color=LABEL)
    if fit_label:
        ax.plot([], [], "-", color=SERIES, lw=1.8, label=fit_label)
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
