"""Render the SLO frontier: aggregate throughput against per-user quality.

Form: connected scatter. The data is an ordered sweep (concurrency), and the
question is a trade-off with a hard boundary -- so points must stay identifiable
(each is a concurrency level) while the path shows direction. A bar chart would
lose the ordering; a plain scatter would lose the trajectory.

Colour: one series (categorical slot 1) plus one status colour for the SLO
boundary. `status-good` was dropped after the palette validator flagged
good vs critical at CVD delta-E 4.1 (deutan) -- the classic red/green failure.
The SLO-holding point is now carried by marker shape and a direct label, so
identity never rests on hue alone.

    node scripts/validate_palette.js "#2a78d6,#d03b3b" --mode light   -> all pass
    node scripts/validate_palette.js "#3987e5,#d03b3b" --mode dark    -> all pass
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 8xH100, market workload (run 1788203369): users, goodput rps, TTFT p99, TPOT p99
SWEEP = [(8, 0.29, 684, 8.1), (16, 0.55, 649, 10.2), (32, 0.88, 778, 21.1),
         (64, 1.58, 1066, 37.8), (128, 2.20, 1061, 82.9)]
N_GPU, OUT_TOK, WINDOW_S = 8, 2076, 120.0
TTFT_MAX, TPOT_MAX = 2000.0, 50.0

# A p99 needs ~100 completions to be a percentile rather than a maximum. At 8
# users only ~35 requests finish in the window, so "p99 TTFT" there is the
# single worst request -- an extreme order statistic. That is why the TTFT
# series wanders (684 -> 649 -> 778) instead of rising: the low-concurrency
# points are noise. Marked hollow on the chart rather than silently plotted.
MIN_COMPLETIONS_FOR_P99 = 100


def _reliable(goodput: float) -> bool:
    return goodput * WINDOW_S >= MIN_COMPLETIONS_FOR_P99

THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  axis="#c3c2b7", series="#2a78d6", crit="#d03b3b", band="#d03b3b12", hatch="#d03b3b55"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  axis="#383835", series="#3987e5", crit="#d03b3b", band="#d03b3b1f", hatch="#d03b3b66"),
}


def _panel(ax, xs, ys, labels, bound, side, xlabel, held_i, t, solid=None):
    """One trade-off panel with a constraint boundary."""
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or hi * 0.2
    # Include the bound in the axis only when it is near the data; a far bound
    # would squeeze every point into a corner and waste the panel.
    near = lo - span * 1.6 <= bound <= hi + span * 1.6
    x0, x1 = (min(lo, bound), max(hi, bound)) if near else (lo, hi)
    pad = (x1 - x0) * 0.13 or 1
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(0, max(ys) * 1.20)

    if near:
        # Violation region: hatch as well as fill, so it survives CVD and print.
        if side == "right":
            # facecolor, not color: `color` overrides edgecolor and the hatch
            # would lose its stroke, taking the CVD/print fallback with it.
            ax.axvspan(bound, x1 + pad, facecolor=t["band"], hatch="///",
                       edgecolor=t["hatch"], linewidth=0.0, zorder=0)
        else:
            ax.axvspan(x0 - pad, bound, facecolor=t["band"], hatch="///",
                       edgecolor=t["hatch"], linewidth=0.0, zorder=0)
        ax.axvline(bound, color=t["crit"], lw=2, ls=(0, (5, 3)), zorder=2)
        ax.annotate(f"SLO  {bound:,.0f}", (bound, ax.get_ylim()[1] * 0.955),
                    xytext=(7 if side == "right" else -7, 0),
                    textcoords="offset points", color=t["crit"], fontsize=8.5,
                    ha="left" if side == "right" else "right", va="top",
                    fontweight="bold")
    else:
        ax.annotate(f"SLO {bound:,.0f} — off scale, never approached",
                    (0.98, 0.955), xycoords="axes fraction", color=t["ink2"],
                    fontsize=8.5, ha="right", va="top")

    order = sorted(range(len(xs)), key=lambda i: xs[i])   # path in x order
    ax.plot([xs[i] for i in order], [ys[i] for i in order],
            color=t["series"], lw=2, zorder=3, solid_capstyle="round")
    # Hollow marker = too few completions for p99 to mean anything (see
    # MIN_COMPLETIONS_FOR_P99). Shape, not colour, so it survives CVD.
    solid = [True]*len(xs) if solid is None else solid
    for x, y, ok in zip(xs, ys, solid):
        ax.scatter([x], [y], s=64,
                   color=t["series"] if ok else t["surface"],
                   edgecolor=t["surface"] if ok else t["series"],
                   linewidth=2, zorder=4)                  # 2px surface ring

    # The SLO-holding level: distinct SHAPE, not a second hue.
    ax.scatter([xs[held_i]], [ys[held_i]], s=290, facecolor="none",
               edgecolor=t["series"], linewidth=2.4, zorder=5)
    # Short label only: a long one overflowed the right panel and collided with
    # the y-axis on the left. The full sentence lives in the subtitle instead.
    ax.annotate(f"N* = {labels[held_i]}", (xs[held_i], ys[held_i]),
                xytext=(0, -26), textcoords="offset points", ha="center",
                color=t["ink"], fontsize=9, fontweight="bold")

    for i, (x, y) in enumerate(zip(xs, ys)):              # selective direct labels
        if i == held_i:
            continue
        ax.annotate(f"{labels[i]}", (x, y), xytext=(0, 11 if i % 2 else -17),
                    textcoords="offset points", ha="center",
                    color=t["muted"], fontsize=8.5)

    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9.5, labelpad=8)
    ax.grid(True, axis="y", color=t["axis"], lw=0.8, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for k, sp in ax.spines.items():
        sp.set_visible(k == "bottom")
        sp.set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)


def render(mode: str, out: pathlib.Path) -> pathlib.Path:
    t = THEME[mode]
    agg = [r[1] * OUT_TOK / N_GPU for r in SWEEP]          # aggregate tok/s per GPU
    users = [r[0] for r in SWEEP]
    held = max(i for i, r in enumerate(SWEEP)
               if r[2] <= TTFT_MAX and r[3] <= TPOT_MAX)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), facecolor=t["surface"])
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.22, left=0.075, right=0.975)
    for ax in axes:
        ax.set_facecolor(t["surface"])

    solid = [_reliable(r[1]) for r in SWEEP]
    # TPOT averages over hundreds of tokens per request, so it is stable even
    # at 35 completions; only the TTFT percentile needs the caveat.
    _panel(axes[0], [1000 / r[3] for r in SWEEP], agg, users,
           1000 / TPOT_MAX, "left", "p99 decode rate per user  (tokens/second)",
           held, t)
    _panel(axes[1], [r[2] for r in SWEEP], agg, users,
           TTFT_MAX, "right", "p99 time to first token  (ms)", held, t,
           solid=solid)
    axes[1].annotate("hollow: <100 completions,\nso p99 is a maximum",
                     (0.97, 0.05), xycoords="axes fraction", color=t["muted"],
                     fontsize=8, ha="right", va="bottom")

    axes[0].set_ylabel("aggregate output tokens/second per GPU",
                       color=t["ink2"], fontsize=9.5, labelpad=10)
    axes[0].set_title("Decode rate is what binds", color=t["ink"], fontsize=11.5,
                      fontweight="bold", loc="left", pad=10)
    axes[1].set_title("Time to first token never does", color=t["ink"],
                      fontsize=11.5, fontweight="bold", loc="left", pad=10)

    fig.suptitle(
        f"Throughput costs per-user quality — {agg[held]:.0f} tok/s/GPU at "
        f"{users[held]} concurrent users",
        color=t["ink"], fontsize=14, fontweight="bold", x=0.075, ha="left", y=0.955)
    fig.text(0.075, 0.885,
             "Qwen3.8-27B-FP8 · 8×H100 · market workload · each point is one "
             "concurrency level · ringed = last level holding both SLOs",
             color=t["ink2"], fontsize=9.5, ha="left")

    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    d = pathlib.Path("plots"); d.mkdir(exist_ok=True)
    for m in ("light", "dark"):
        print("wrote", render(m, d / f"slo-frontier-{m}.png"))
