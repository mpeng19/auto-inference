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
    tiers = _tiers_from_traces(root)
    out = []
    for r in c.execute("SELECT * FROM experiments ORDER BY ts"):
        m = json.loads(r["metrics"] or "{}")
        base = json.loads(r["baseline_metrics"] or "{}")
        bill = m.get("bill_per_1k")
        # Records written before the tier travelled with the metrics: the
        # trace's eval_result for the same stack says which tier it was.
        if not m.get("tier") and r["stack_digest"] in tiers:
            m["tier"] = tiers[r["stack_digest"]]
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


def _tiers_from_traces(root: str | pathlib.Path) -> dict[str, str]:
    """stack digest -> the tier of the *last* eval_result for it in the
    traces. A screen that was promoted has a later full result, which is the
    one memory recorded."""
    out: dict[str, str] = {}
    for tf in tr.find(root):
        try:
            for t in tr.read(tf.path, kinds=("eval_result",)):
                tier = (t.get("data") or {}).get("tier")
                if tier and t.get("name"):
                    out[t["name"]] = tier
        except Exception:
            continue
    return out


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
        lines.append(f"  {r.verdict:8s} {r.tier:6s} {d:>7} {bill:>10} {r.rank or '-':>6}  "
                     f"{r.agent_id}  {r.title}")
    return "\n".join(lines)


# ── what the run directory says about the fleet itself ──────────────────

def recent_calls(root: str | pathlib.Path, agent_id: str, k: int = 5) -> list[dict]:
    """The agent's last `k` model calls from `<root>/<agent>/calls/`, each
    summarised: phase, minutes, messages, output and cache-read tokens, the
    tools it leaned on. One reader for the TUI's detail pane and the ask
    context, so the two never disagree about what a call was."""
    d = pathlib.Path(root) / agent_id / "calls"
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.jsonl"))[-k:]:
        try:
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        except (OSError, ValueError):
            continue
        if not rows:
            continue
        msgs = [r for r in rows if r.get("type") == "assistant"]
        tools: dict[str, int] = {}
        for r in msgs:
            for name in r.get("tools") or ():
                tools[name] = tools.get(name, 0) + 1
        span = (rows[-1]["ts"] - rows[0]["ts"]) / 60 if len(rows) > 1 else 0.0
        out.append({"phase": f.stem.split("-")[0], "file": f.name, "min": span,
                    "msgs": len(msgs),
                    "out": sum(r.get("output", 0) for r in msgs),
                    "cache": sum(r.get("cache_read", 0) for r in msgs),
                    "tools": " ".join(f"{n}x{c}" for n, c in
                                      sorted(tools.items(), key=lambda kv: -kv[1])[:3])})
    return out


def load_summary(root: str | pathlib.Path) -> dict:
    """`<root>/summary.json` as the fleet last wrote it (see
    `Fleet.write_summary`), or `{}`. It is the only record of what each
    agent cost, how it spent its hours and how the fleet ended once the
    daemon and its session row are gone."""
    p = pathlib.Path(root) / "summary.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, ValueError):
        return {}


def snapshot_text(snap) -> str:
    """A `SessionView` -- or the dict `summary.json` holds -- as the lines an
    analyst reads: the fleet line, then one line per agent with its status,
    idea, attempts, Modal spend, tokens and time by phase. Dollars here are
    Modal only; Claude usage is on the subscription and is shown as tokens."""
    from dataclasses import asdict, is_dataclass

    d = asdict(snap) if is_dataclass(snap) else dict(snap or {})
    if not d:
        return "no snapshot"
    tok = d.get("tokens") or {}
    lines = [f"session {d.get('session_id')}  phase {d.get('phase')}  "
             f"agents {len(d.get('agents') or ())}/{d.get('target_agents')}  "
             f"Modal spend ${float(d.get('cost_usd') or 0):.2f} of ${float(d.get('budget_usd') or 0):.0f}  "
             f"tokens in {tok.get('input', 0):,} out {tok.get('output', 0):,} "
             f"cache {tok.get('cache_read', 0):,}",
             f"evals {d.get('evals_running', 0)} running, {d.get('evals_queued', 0)} queued, "
             f"{d.get('evals_completed', 0)} done, {d.get('evals_deduped', 0)} deduped"]
    if d.get("note"):
        lines.append(f"note: {d['note']}")
    for a in d.get("agents") or ():
        t = a.get("tokens") or {}
        ph = "  ".join(f"{k} {v / 60:.0f}m" for k, v in
                       sorted((a.get("phase_s") or {}).items(), key=lambda kv: -kv[1]))
        best = a.get("best_delta_pct")
        lines.append(
            f"{a.get('agent_id')}: {a.get('status')}  idea={a.get('idea_title') or '-'!r}  "
            f"attempt {a.get('attempt', 0)} of {a.get('attempts_total', 0)} total  "
            f"best {'-' if best is None else f'{best:+.1f}%'}  "
            f"spend ${float(a.get('cost_usd') or 0):.2f}  "
            f"tokens in {t.get('input', 0):,} out {t.get('output', 0):,} "
            f"cache {t.get('cache_read', 0):,}"
            + (f"  time: {ph}" if ph else "")
            + (f"  doing: {a['activity']}" if a.get("activity") else "")
            + (f"  note: {a['note']}" if a.get("note") else ""))
    return "\n".join(lines)
