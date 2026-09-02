"""Magic-trace-style timeline rendering of a window to PNG."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .store import TraceStore


def _color(name: str):
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    return ((h >> 16 & 255) / 340 + 0.2, (h >> 8 & 255) / 340 + 0.2, (h & 255) / 340 + 0.2)


def timeline(store: TraceStore, t0: float, t1: float, out_png: str | Path,
             track_like: str = "%", max_labels: int = 40, marks: list[float] | None = None) -> dict:
    tracks = [t for t in store.conn.execute(
        "SELECT * FROM tracks WHERE (name LIKE ? OR ? = '%') ORDER BY kind DESC, id", (track_like, track_like))]
    plotted_tracks = []
    for t in tracks:
        spans = store.conn.execute(
            "SELECT s.ts, s.dur, n.name FROM spans s JOIN names n ON n.id=s.name_id "
            "WHERE s.track_id=? AND s.ts < ? AND s.ts + s.dur > ? ORDER BY s.ts LIMIT 5000",
            (t["id"], t1, t0)).fetchall()
        if spans:
            plotted_tracks.append((t, spans))
    fig, ax = plt.subplots(figsize=(16, max(2.4, 0.65 * len(plotted_tracks))))
    n_labels = 0
    for y, (_t, spans) in enumerate(plotted_tracks):
        bars = [(s["ts"], s["dur"]) for s in spans]
        cols = [_color(s["name"]) for s in spans]
        ax.broken_barh(bars, (y + 0.1, 0.8), facecolors=cols, edgecolor="none")
        for s in spans:
            if s["dur"] > (t1 - t0) * 0.02 and n_labels < max_labels:
                ax.text(s["ts"] + s["dur"] / 2, y + 0.5, s["name"][:24], ha="center", va="center",
                        fontsize=7, clip_on=True)
                n_labels += 1
    ax.set_yticks([y + 0.5 for y in range(len(plotted_tracks))])
    ax.set_yticklabels([(t["name"] or f"pid{t['pid']}/tid{t['tid']}")[:28] + (f" [{t['kind']}]" if t["kind"] else "")
                        for t, _ in plotted_tracks], fontsize=8)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time (us)")
    ax.set_title(f"trace timeline {t0:.0f}-{t1:.0f} us  ({(t1 - t0):.0f} us window)")
    ax.grid(axis="x", alpha=0.25)
    for mt in (marks or []):
        if t0 <= mt <= t1:
            ax.axvline(mt, color="red", lw=1.2, alpha=0.85, zorder=5)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"png": str(out_png), "tracks": len(plotted_tracks),
            "spans_drawn": sum(len(s) for _, s in plotted_tracks)}
