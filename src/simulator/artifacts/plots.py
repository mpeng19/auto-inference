"""Figures written into a run's `root_dir`.

Two figures, because there are two questions a result has to answer.

**Where does the SLO stop us?** One panel per bound, each with its own limit,
so the binding constraint is visible rather than inferred. Which bound gives
way first is the whole diagnosis: on 1xH100 the p90 TTFT bound never binds and
mean TPOT does, which is why prefill work would have been the wrong thing to
optimise.

**What does that cost, and how much of the market does it buy?** Price and
share are one object, not two, and quoting either alone is misleading -- below
saturation, utilisation is set by demand rather than chosen, so price runs as
1/share no matter how good the stack is.

Style follows the two house rules that matter here: no dual axes ever, and
identity is never carried by colour alone -- filled vs hollow markers do the
SLO encoding, so the figures survive greyscale and every form of CVD without
needing a palette check.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _num(v, _pos=None):
    """Log axes with plain numbers: 0.05, 0.3, 3 -- never 10^-1."""
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


DPI = 150
SERIES = "#2a78d6"        # validated categorical slot 1
LIMIT = "#b3402f"         # limit lines only; never used for a data series
INK = "dimgrey"


def _theme():
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")


def _finish(ax, xlabel="", ylabel="", title="", sub=""):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.tick_params(labelcolor=INK, colors=INK)
    if title:
        ax.set_title(title, loc="left", color="black", pad=18 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK)


def slo_frontier(sim, res, path: pathlib.Path) -> str:
    """One panel per SLO bound: the measured statistic against concurrency."""
    _theme()
    bounds = sim.slo.bounds
    fig, axes = plt.subplots(1, len(bounds), figsize=(4.6 * len(bounds), 4.4),
                             dpi=DPI, squeeze=False)
    star = res.best
    for ax, b in zip(axes[0], bounds, strict=True):
        xs = [p.n_users for p in res.curve]
        ys = [next((c["value"] for c in p.checks if c["label"] == b.label), None)
              for p in res.curve]
        okm = [p.meets_slo for p in res.curve]
        ax.plot(xs, ys, color=SERIES, lw=2, zorder=3)
        for x, y, ok in zip(xs, ys, okm, strict=True):
            if y is None:
                continue
            ax.scatter([x], [y], s=90, zorder=5, linewidth=2,
                       color=SERIES if ok else "white",
                       edgecolor="white" if ok else SERIES)
        ax.axhline(b.limit_ms, color=LIMIT, lw=1.6, ls=(0, (5, 3)), zorder=2)
        # Headroom above whichever is higher, the limit or the data, so the
        # N* label has somewhere to sit that is not on top of the limit line.
        vals = [v for v in ys if v is not None]
        top = max([*vals, b.limit_ms])
        ax.set_ylim(min([*vals, b.limit_ms]) * 0.92, top * 1.14)
        ax.annotate(f"limit {b.limit_ms:g} ms", (xs[-1], b.limit_ms), color=LIMIT,
                    fontsize=8.5, ha="right", textcoords="offset points",
                    xytext=(0, 6))
        if star is not None:
            ax.axvline(star.n_users, color=INK, lw=1, ls=":", zorder=1)
            ax.annotate(f"N* = {star.n_users}", (star.n_users, 1.0),
                        xycoords=("data", "axes fraction"), color=INK, fontsize=8.5,
                        textcoords="offset points", xytext=(4, -12))
        # Annotate the insight, not the axis. "Binds" and "is violated" are
        # different claims and were conflated in an earlier version: p90 TPOT
        # crosses its limit at high load but mean TPOT crosses first, so p90 is
        # violated without ever being the binding constraint.
        failing = [p for p in res.curve if not p.meets_slo]
        binds_first = bool(failing) and min(
            failing, key=lambda p: p.n_users).binding == b.label
        violated = any(v is not None and v > b.limit_ms for v in ys)
        _finish(ax, "concurrent conversations", f"{b.label}  (ms)", b.label,
                "BINDS FIRST — this sets N*" if binds_first
                else ("violated, but only past N*" if violated
                      else "never violated at these levels"))
    sns.despine(left=True, bottom=True)
    fig.suptitle("Where offered load stops meeting the SLO", x=0.005, ha="left",
                 fontsize=12.5)
    fig.text(0.005, 0.005, _caption(sim, res), fontsize=7.4, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def price_vs_share(sim, res, path: pathlib.Path) -> str:
    """Price at each feasible concurrency, and what happens below saturation."""
    from ..price.market import Economics
    _theme()
    m, star = res.market, res.best
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=DPI)

    # ── left: the curve the sweep traced ──
    best_provider = min(m.providers, key=lambda q: m.bill_per_1k(
        q.effective_in_per_m, q.listed_out_per_m))
    ref = m.bill_per_1k(best_provider.effective_in_per_m,
                        best_provider.listed_out_per_m)
    xs = [p.share_per_node * 100 for p in res.curve]
    ys = [p.bill_per_1k for p in res.curve]
    ax.plot(xs, ys, color=SERIES, lw=2, zorder=3, alpha=.45)
    ok = [p for p in res.curve if p.meets_slo]
    if ok:
        ax.plot([p.share_per_node * 100 for p in ok], [p.bill_per_1k for p in ok],
                color=SERIES, lw=2.2, zorder=4)
    for p in res.curve:
        ax.scatter([p.share_per_node * 100], [p.bill_per_1k], s=95, zorder=5,
                   linewidth=2, color=SERIES if p.meets_slo else "white",
                   edgecolor="white" if p.meets_slo else SERIES)
    # Levels bunch up in share -- 1xH100 spans 0.27-0.55% across the whole
    # sweep -- so a fixed label offset overlaps. Alternate sides whenever two
    # points sit within a tenth of the axis of each other.
    span = (max(xs) - min(xs)) or 1.0
    prev_x, above = None, True
    for p, x, y in sorted(zip(res.curve, xs, ys, strict=True), key=lambda t: t[1]):
        if prev_x is not None and (x - prev_x) < 0.12 * span:
            above = not above
        else:
            above = True
        ax.annotate(f"N={p.n_users}", (x, y), textcoords="offset points",
                    xytext=(0, 13 if above else -19), ha="center",
                    fontsize=8.5, color=INK)
        prev_x = x
    lo, hi = min([*ys, ref]), max(ys)
    ax.set_ylim(lo - 0.13 * (hi - lo), hi + 0.11 * (hi - lo))
    ax.axhline(ref, color=LIMIT, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.annotate(f"{best_provider.name} ${ref:.2f} — cheapest provider",
                (min(xs), ref), textcoords="offset points", xytext=(0, 6),
                fontsize=8.5, color=LIMIT)
    _finish(ax, "market share ONE node can serve  (%)", "$ per 1,000 requests",
            "Price falls with concurrency — until the SLO stops it",
            "filled = meets SLO   ·   hollow = violates it")

    # ── right: below saturation, demand sets the price ──
    if star is not None:
        e = Economics(gpu_s_per_request=star.gpu_s_per_request, n_gpu=sim.n_gpu,
                      rate_per_gpu_hour=sim.rate_per_gpu_hour, utilisation=sim.util)
        smax = star.share_per_node
        s = np.logspace(np.log10(max(smax / 30, 1e-4)), np.log10(smax * 12), 400)
        y = [e.price_at_share(x, m) for x in s]
        bx.plot(s * 100, y, color=SERIES, lw=2.2, zorder=4)
        bx.scatter([smax * 100], [star.bill_per_1k], s=110, color=SERIES,
                   edgecolor="white", linewidth=2, zorder=6)
        bx.annotate(f"one node saturates\n{smax*100:.2f}% share  "
                    f"${star.bill_per_1k:.2f}/1k",
                    (smax * 100, star.bill_per_1k), fontsize=8.5, color=INK,
                    textcoords="offset points", xytext=(12, 16))
        bx.axhline(ref, color=LIMIT, lw=1.6, ls=(0, (5, 3)), zorder=2)
        bx.annotate(f"{best_provider.name} ${ref:.2f}", (s[0] * 100, ref),
                    textcoords="offset points", xytext=(0, 6), fontsize=8.5,
                    color=LIMIT)
        bx.set_xscale("log")
        bx.set_yscale("log")
        bx.minorticks_off()
        for axis in (bx.xaxis, bx.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(_num))
        bx.set_ylim(min(ref, star.bill_per_1k) * 0.72, max(y) * 1.15)
        _finish(bx, "market share actually captured  (%, log)",
                "$ per 1,000 requests  (log)",
                "Below the kink, price is set by demand",
                "left of the marker the node idles; right of it, add nodes")
    sns.despine(left=True, bottom=True)
    fig.text(0.005, 0.005, _caption(sim, res), fontsize=7.4, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _caption(sim, res) -> str:
    m = res.market or sim.market
    return (f"{sim.model} · {sim.n_gpu}x{sim.gpu} · {sim.cost_basis} · "
            f"utilisation {sim.util:.0%} · break-even · stack {sim.stack.describe()[:60]}\n"
            f"SLO {sim.slo.describe()} · market {m.requests_per_day:,.0f} req/day at "
            f"{m.in_per_request:,.0f} in / {m.out_per_request:,.0f} out per request")


def render_all(sim, res, root: pathlib.Path) -> dict[str, str]:
    out = {}
    if res.curve:
        out["slo_frontier"] = slo_frontier(sim, res, root / "slo-frontier.png")
    if res.curve and res.market is not None:
        out["price_vs_share"] = price_vs_share(sim, res, root / "price-vs-share.png")
    return out
