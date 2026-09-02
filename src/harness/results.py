"""What a run has found, best first.

Reads a fleet root the way a person would in the morning: every experiment
memory recorded, priced and ranked against the baseline it was judged on,
with the diff that produced the best ones pulled from the traces. This is the
data behind the TUI's results tab, `harness results`, and the context the
ask agent answers from.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
from dataclasses import dataclass, field

from . import traces as tr


@dataclass(frozen=True)
class Result:
    experiment_id: str
    agent_id: str
    verdict: str
    hypothesis: str
    summary: str
    bill_per_1k: float | None
    delta_pct: float | None
    rank: str                       # "9/12" on the OpenRouter board, or ""
    share_pct: float | None
    n_star: int | None
    quality: str                    # "gsm8k 69%" or ""
    stack_digest: str
    trace_ref: str
    ts: float
    metrics: dict = field(default_factory=dict)
    tier: str = "full"

    @property
    def title(self) -> str:
        return self.hypothesis[:70]


def leaderboard(root: str | pathlib.Path) -> list[Result]:
    """Every priced experiment under `root`, best delta first, then the
    unpriced ones (losses, invalid diffs) after."""
    db = pathlib.Path(root) / "memory.db"
    if not db.is_file():
        return []
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("SELECT * FROM experiments ORDER BY ts"):
        m = json.loads(r["metrics"] or "{}")
        base = json.loads(r["baseline_metrics"] or "{}")
        bill = m.get("bill_per_1k")
        # A screen is compared with stock at screen tier (see loop._delta).
        screen = base.get("screen") if m.get("tier") == "screen" else None
        b0 = (screen or {}).get("bill_per_1k") if isinstance(screen, dict) else base.get("bill_per_1k")
        delta = (round((bill - b0) / b0 * 100, 2)
                 if isinstance(bill, (int, float)) and isinstance(b0, (int, float)) and b0
                 else None)
        rank = (f"{m['rank_bill']}/{m['rank_of']}"
                if m.get("rank_bill") and m.get("rank_of") else "")
        q = ""
        for row in m.get("quality") or ():
            if isinstance(row, dict) and row.get("suite"):
                q = f"{row['suite']} {row.get('accuracy', 0):.0%}"
        share = m.get("share_per_node")
        out.append(Result(
            experiment_id=r["id"], agent_id=r["agent_id"] or "", verdict=r["verdict"] or "",
            hypothesis=r["hypothesis"] or "", summary=r["summary"] or "",
            bill_per_1k=bill if isinstance(bill, (int, float)) else None,
            delta_pct=delta, rank=rank,
            share_pct=(share * 100 if isinstance(share, (int, float)) else None),
            n_star=m.get("n_star"), quality=q, tier=m.get("tier") or "full",
            stack_digest=r["stack_digest"] or "",
            trace_ref=r["trace_ref"] or "", ts=r["ts"] or 0.0, metrics=m))
    c.close()
    priced = sorted((x for x in out if x.delta_pct is not None), key=lambda x: x.delta_pct)
    rest = sorted((x for x in out if x.delta_pct is None), key=lambda x: -x.ts)
    return priced + rest


def diff_for(root: str | pathlib.Path, res: Result, limit: int = 12000) -> str:
    """The unified diff behind a result, from its trace's eval_submit turn."""
    if not res.stack_digest:
        return ""
    for tf in tr.find(root):
        try:
            turns = tr.read(tf.path, kinds=("eval_submit",))
        except Exception:
            continue
        for t in turns:
            if t.get("name") == res.stack_digest:
                d = (t.get("data") or {}).get("diff") or ""
                return d[:limit]
    return ""


def summary_text(root: str | pathlib.Path, k: int = 8) -> str:
    rows = leaderboard(root)
    if not rows:
        return "no experiments recorded yet"
    lines = [f"{len(rows)} experiments; best first"]
    for r in rows[:k]:
        d = "-" if r.delta_pct is None else f"{r.delta_pct:+.1f}%"
        bill = "-" if r.bill_per_1k is None else f"${r.bill_per_1k:.2f}/1k"
        lines.append(f"  {r.verdict:8s} {d:>7} {bill:>10} {r.rank or '-':>6}  "
                     f"{r.agent_id}  {r.title}")
    return "\n".join(lines)
