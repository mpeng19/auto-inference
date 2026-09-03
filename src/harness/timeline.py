"""A run as a timeline a person can read.

The traces are JSONL for loaders; nobody reads them in the morning. This
turns them into what someone actually wants to know per agent: which ideas
it worked, when, how long each phase took (writing, waiting on a GPU,
studying), what came back, and what it cost. Rendered as text for the
terminal, Markdown for `<root>/timeline.md` (rewritten by the fleet after
every outcome, so it is always current), and a single-file HTML Gantt.

Works on traces written before phase stamping existed by taking each turn's
duration from the gap to the previous turn.
"""
from __future__ import annotations

import datetime as dt
import html
import pathlib
from dataclasses import dataclass, field

from . import traces as tr

# What a turn closes, from its kind/name when `data.phase` is absent.
_PHASE_BY_TURN = {
    ("thought", "recall"): "recall", ("thought", "propose"): "edit",
    ("eval_submit", None): "edit", ("thought", "study"): "study",
    ("eval_result", None): "wait", ("error", None): "error",
    ("tool_call", "check"): "check", ("prompt", None): "start",
}
PHASES = ("edit", "check", "study", "wait", "paper", "recall", "error", "other")
_COLOUR = {"edit": "#4c78a8", "study": "#72b7b2", "wait": "#f58518", "check": "#54a24b",
           "paper": "#eeca3b", "recall": "#b279a2", "error": "#e45756", "other": "#bab0ac",
           "start": "#bab0ac"}


@dataclass(frozen=True)
class Span:
    agent: str
    idea: str
    trace: str
    phase: str
    t0: float
    t1: float
    label: str = ""

    @property
    def seconds(self) -> float:
        return max(0.0, self.t1 - self.t0)


@dataclass
class IdeaRun:
    agent: str
    trace: str
    title: str
    started: float
    ended: float
    outcome: str
    cost_usd: float
    spans: list[Span] = field(default_factory=list)
    results: list[str] = field(default_factory=list)

    def by_phase(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.spans:
            out[s.phase] = out.get(s.phase, 0.0) + s.seconds
        return out


def _phase(turn: dict) -> str:
    d = turn.get("data") or {}
    p = d.get("phase")
    if p:
        return {"propose": "edit", "submit": "edit"}.get(p, p)
    k, n = turn.get("kind"), turn.get("name")
    return _PHASE_BY_TURN.get((k, n)) or _PHASE_BY_TURN.get((k, None)) or "other"


def _label(turn: dict) -> str:
    d = turn.get("data") or {}
    k = turn.get("kind")
    if k == "eval_submit":
        return f"submitted {d.get('tier', '')} run"
    if k == "eval_result":
        bill = d.get("bill_per_1k")
        if bill is None:
            return f"{d.get('tier', '')} failed: {str(d.get('reason') or d.get('error') or '')[:60]}"
        return f"{d.get('tier', '')} ${bill:.2f}/1k" + (
            f", quality {d['quality'][0].get('accuracy', 0):.0%}"
            if d.get("quality") else "")
    if k == "error":
        return str(turn.get("content") or "")[:80]
    if k == "thought" and turn.get("name") == "propose":
        den = d.get("denials")
        return "diff written" + (f" ({den} commands denied)" if den else "")
    if k == "tool_call":
        return str(turn.get("content") or "")[:40]
    return ""


def build(root: str | pathlib.Path, agent_id: str = "") -> list[IdeaRun]:
    runs: list[IdeaRun] = []
    for tf in tr.find(root, agent_id=agent_id):
        try:
            turns = tr.read(tf.path)
        except Exception:
            continue
        if not turns:
            continue
        title = next((str(t.get("content") or "")[:80] for t in turns
                      if t.get("kind") == "prompt"), tf.idea_id)
        run = IdeaRun(agent=tf.agent_id, trace=tf.trace_id, title=title,
                      started=tf.started_at or turns[0].get("ts", 0.0),
                      ended=tf.ended_at or turns[-1].get("ts", 0.0),
                      outcome=tf.outcome, cost_usd=tf.cost_usd)
        prev = run.started
        for t in turns:
            ts = float(t.get("ts") or prev)
            d = t.get("data") or {}
            el = d.get("elapsed_s")
            t0 = ts - float(el) if el is not None else prev
            phase = _phase(t)
            if phase != "start":
                run.spans.append(Span(run.agent, title, run.trace, phase, t0, ts, _label(t)))
            if t.get("kind") == "eval_result":
                run.results.append(_label(t))
            prev = ts
        runs.append(run)
    return sorted(runs, key=lambda r: (r.agent, r.started))


def _hm(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M") if ts else "--:--"


def _dur(s: float) -> str:
    return f"{s/60:.0f}m" if s >= 90 else f"{s:.0f}s"


def render_text(root: str | pathlib.Path, agent_id: str = "") -> str:
    runs = build(root, agent_id)
    if not runs:
        return "no traces yet"
    lines = []
    by_agent: dict[str, list[IdeaRun]] = {}
    for r in runs:
        by_agent.setdefault(r.agent, []).append(r)
    for agent, items in by_agent.items():
        lines.append(f"{agent}")
        for r in items:
            ph = r.by_phase()
            mix = "  ".join(f"{p} {_dur(ph[p])}" for p in PHASES if ph.get(p))
            lines.append(f"  {_hm(r.started)}-{_hm(r.ended)}  {r.title[:56]:<56}  "
                         f"{r.outcome or 'running':<11} ${r.cost_usd:.2f}")
            if mix:
                lines.append(f"           {mix}")
            for res in r.results:
                lines.append(f"           -> {res}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_markdown(root: str | pathlib.Path) -> str:
    runs = build(root)
    root = pathlib.Path(root)
    out = [f"# Timeline: {root.name}", "",
           f"_rewritten by the fleet after every outcome; {len(runs)} ideas_", ""]
    by_agent: dict[str, list[IdeaRun]] = {}
    for r in runs:
        by_agent.setdefault(r.agent, []).append(r)
    for agent, items in by_agent.items():
        out += [f"## {agent}", "", "| when | idea | outcome | cost | edit | study | wait | results |",
                "|---|---|---|---|---|---|---|---|"]
        for r in items:
            ph = r.by_phase()
            out.append(f"| {_hm(r.started)}-{_hm(r.ended)} | {r.title[:60]} | {r.outcome or 'running'} "
                       f"| ${r.cost_usd:.2f} | {_dur(ph.get('edit', 0))} | {_dur(ph.get('study', 0))} "
                       f"| {_dur(ph.get('wait', 0))} | {'; '.join(r.results) or '-'} |")
        out.append("")
    return "\n".join(out)


def write_markdown(root: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(root) / "timeline.md"
    p.write_text(render_markdown(root) + "\n")
    return p


def render_html(root: str | pathlib.Path) -> str:
    """One page, no dependencies: a row per idea, a bar per phase."""
    runs = build(root)
    if not runs:
        return "<p>no traces yet</p>"
    t0 = min(r.started for r in runs)
    t1 = max(max((s.t1 for s in r.spans), default=r.ended) for r in runs)
    span = max(60.0, t1 - t0)
    width, row_h, left = 1100, 22, 340
    rows = []
    y = 30
    agent_prev = None
    for r in runs:
        if r.agent != agent_prev:
            rows.append(f'<text x="8" y="{y+15}" font-weight="bold">{html.escape(r.agent)}</text>')
            y += row_h
            agent_prev = r.agent
        rows.append(f'<text x="24" y="{y+15}" font-size="12">{html.escape(r.title[:44])}</text>')
        for s in r.spans:
            x = left + (s.t0 - t0) / span * (width - left - 20)
            w = max(1.5, s.seconds / span * (width - left - 20))
            tip = f"{s.phase} {_dur(s.seconds)} {html.escape(s.label)}"
            rows.append(f'<rect x="{x:.1f}" y="{y+4}" width="{w:.1f}" height="{row_h-8}" '
                        f'fill="{_COLOUR.get(s.phase, "#bab0ac")}"><title>{tip}</title></rect>')
        rows.append(f'<text x="{width-14}" y="{y+15}" font-size="11" text-anchor="end">'
                    f'{html.escape(r.outcome or "running")} ${r.cost_usd:.2f}</text>')
        y += row_h
    legend = " ".join(f'<span style="display:inline-block;width:12px;height:12px;background:{_COLOUR[p]}"></span> {p}'
                      for p in ("edit", "check", "study", "wait", "recall", "error"))
    ticks = "".join(
        f'<text x="{left + i/6*(width-left-20):.0f}" y="20" font-size="11">{_hm(t0 + i/6*span)}</text>'
        for i in range(7))
    return (f"<!doctype html><meta charset='utf-8'><title>timeline {html.escape(pathlib.Path(root).name)}</title>"
            f"<style>body{{font:13px system-ui;margin:16px}} svg{{background:#fff}}</style>"
            f"<h2>{html.escape(pathlib.Path(root).name)}</h2><p>{legend}</p>"
            f'<svg width="{width}" height="{y+10}" xmlns="http://www.w3.org/2000/svg">{ticks}'
            + "".join(rows) + "</svg>")
