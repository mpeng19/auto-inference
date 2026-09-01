import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SP = "/private/tmp/claude-501/-Users-michaelpeng-Desktop-auto-inference/1f1ac11b-8118-4087-9409-0bea081bdd79/scratchpad"
d = json.load(open(f"{SP}/curve.json"))
M, U, RATE = d["market"], d["u_max"], d["rate"]
CH, CHN = 8.67, "Chutes"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992", "#e6e5e0"
COL = {"1xH100": "#2a78d6", "2xH100": "#eb6834"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.6, 5.8), facecolor=SURF)
for a in (ax, bx):
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID); a.spines[s].set_linewidth(1.0)
    a.tick_params(colors=INK2, labelsize=9, length=3, width=1.0)
    a.grid(True, color=GRID, linewidth=0.8)
    a.set_axisbelow(True)

# ── A: price at every SLO-feasible concurrency ──────────────────────────
# label offsets tuned per point; the 1xH100 levels crowd into 0.27-0.55%
# share, so alternating above/below is the only way they all fit.
# N=12/N=16 and N=24/N=32 nearly coincide, so each label is pushed into its
# own quadrant by hand; there is no automatic placement that survives this.
OFF = {("1xH100", 4): (0, 14), ("1xH100", 8): (-23, 4), ("1xH100", 12): (-27, -4),
       ("1xH100", 16): (24, 5), ("1xH100", 24): (25, 3),
       ("2xH100", 8): (0, 14), ("2xH100", 16): (-26, -10), ("2xH100", 24): (13, 12),
       ("2xH100", 32): (25, -6), ("2xH100", 48): (19, 7)}
for cfg, pts in d["configs"].items():
    c = COL[cfg]
    ok = [p for p in pts if p["ok"]]
    no = [p for p in pts if not p["ok"]]
    ax.plot([p["share"]*100 for p in pts], [p["price1k"] for p in pts],
            color=c, lw=2, zorder=3, alpha=.35)
    ax.plot([p["share"]*100 for p in ok], [p["price1k"] for p in ok],
            color=c, lw=2.2, zorder=4, label=f"{cfg}")
    ax.scatter([p["share"]*100 for p in ok], [p["price1k"] for p in ok],
               s=100, color=c, zorder=6, edgecolor=SURF, linewidth=2)
    ax.scatter([p["share"]*100 for p in no], [p["price1k"] for p in no],
               s=100, facecolor=SURF, edgecolor=c, linewidth=2, zorder=5)
    for p in pts:
        ax.annotate(f"N={p['n']}", (p["share"]*100, p["price1k"]),
                    textcoords="offset points", xytext=OFF[(cfg, p["n"])],
                    ha="center", fontsize=8.5,
                    color=INK2 if p["ok"] else MUTED)
    star = max(ok, key=lambda p: p["n"])
    ax.scatter([star["share"]*100], [star["price1k"]], s=250, facecolor="none",
               edgecolor=c, linewidth=1.4, zorder=5, alpha=.55)

ax.axhline(CH, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.annotate(f"{CHN} ${CH:.2f} — cheapest provider", (0.232, CH), ha="left",
            textcoords="offset points", xytext=(0, -15), fontsize=8.5, color=INK2)
ax.set_xscale("log")
ax.minorticks_off()
ax.set_xticks([0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0])
ax.set_xticklabels(["0.25", "0.35", "0.5", "0.75", "1.0", "1.5", "2.0"])
ax.set_xlim(0.22, 2.0)
ax.set_ylim(3.5, 21)
ax.set_xlabel("market share ONE node can serve  (%, log)", fontsize=10, color=INK2)
ax.set_ylabel("$ per 1,000 requests", fontsize=10, color=INK2)
ax.set_title("Price falls as concurrency rises — until the SLO stops it",
             fontsize=11.5, color=INK, loc="left", pad=27)
ax.text(0, 1.05, "filled = meets SLO   ·   hollow = violates it   ·   ring = N*",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
ax.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=INK2)

# ── B: what if the demand isn't there ───────────────────────────────────
s = np.logspace(np.log10(0.03), np.log10(12), 500) / 100
for cfg, pts in d["configs"].items():
    c, g = COL[cfg], pts[0]["g"]
    star = max((p for p in pts if p["ok"]), key=lambda p: p["n"])
    smax, pmax = star["share"], star["price1k"]
    y = np.where(s < smax, g*24*RATE*1000/(np.maximum(s, 1e-9)*M), pmax)
    bx.plot(s*100, y, color=c, lw=2.2, zorder=4, label=f"{cfg}, N*={star['n']}")
    bx.scatter([smax*100], [pmax], s=115, color=c, zorder=6,
               edgecolor=SURF, linewidth=2)
    tx, ha = (0.115, "left") if g == 1 else (2.6, "left")
    ty = 5.2 if g == 1 else 3.9
    bx.annotate(f"one node saturates:  {smax*100:.2f}% share at ${pmax:.2f}/1k",
                (smax*100, pmax), xytext=(tx, ty), ha=ha, fontsize=8.5, color=c,
                arrowprops=dict(arrowstyle="-", color=c, lw=1, alpha=.55,
                                shrinkA=2, shrinkB=6))
bx.axhline(CH, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
bx.annotate(f"{CHN} ${CH:.2f}", (0.035, CH), textcoords="offset points",
            xytext=(0, 7), fontsize=8.5, color=INK2)
bx.set_xscale("log"); bx.set_yscale("log"); bx.minorticks_off()
bx.set_xlim(0.03, 12); bx.set_ylim(3.2, 900)
bx.set_xticks([0.05, 0.1, 0.3, 1, 3, 10])
bx.set_xticklabels(["0.05", "0.1", "0.3", "1", "3", "10"])
bx.set_yticks([5, 10, 30, 100, 300])
bx.set_yticklabels(["5", "10", "30", "100", "300"])
bx.set_xlabel("market share actually captured  (%, log)", fontsize=10, color=INK2)
bx.set_ylabel("$ per 1,000 requests  (log)", fontsize=10, color=INK2)
bx.set_title("Below the kink, price is set by demand, not by the stack",
             fontsize=11.5, color=INK, loc="left", pad=27)
bx.text(0, 1.05, "left of the kink the node idles  ·  right of it, add nodes at flat price",
        transform=bx.transAxes, fontsize=8.5, color=INK2)
bx.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK2)

fig.text(0.006, 0.030,
         f"Qwen3.8-27B-FP8 · $3.00/GPU-hr · break-even · market {M:,.0f} req/day "
         f"(trailing 7d) at {d['IN']:,.0f} in / {d['OUT']:,.0f} out per request",
         fontsize=7.6, color=INK2)
fig.text(0.006, 0.008,
         f"SLO {d['slo']} · utilisation ceiling {U:.0%} = mean/peak of daily volume "
         f"· runs 1788287578 (1xH100), 1788276846 (2xH100)",
         fontsize=7.6, color=INK2)
fig.tight_layout(rect=(0, 0.052, 1, 1))
fig.savefig("docs/price-vs-share.png", dpi=170, facecolor=SURF)
print("wrote docs/price-vs-share.png")
