"""Ask a model about a run.

The morning question is rarely "what is the number" -- the status line
answers that -- but "why did a02's third attempt lose", "which of these
diffs is worth keeping", "did anyone touch the attention backend". That is a
reading job over memory, traces and the bank, and it is what this hands to
Claude: the fleet's state (live from the store while it runs, from
`summary.json` afterwards), every idea's outcome, the leaderboard, the diffs
behind its best results, the timeline, each agent's recent model calls, its
every GPU tool call (`workbench-N/`: what the script set out to do, whether
it ran, what it cost, the tail of what it printed), the profiles captured,
each attempt's report, the spend ledger, the design note and the manager's
tools, as a cached system context, then the conversation on top. A finished
run is answered from its directory alone: nothing in the context needs the
daemon or the session store to still exist.

The context is held under `CONTEXT_CHARS` by dropping the oldest workbench
runs first (an agent's last few are always kept), then trimming every
section in proportion.

Goes through the Anthropic API directly (not the Claude Code subscription),
because it is a question about data, not an edit to a repository, and the
API key is already in the environment. Model defaults to Claude Fable 5.1.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import time
from dataclasses import dataclass, field

from . import results as rs

DEFAULT_MODEL = "claude-fable-5-1"
CONTEXT_CHARS = 120_000
WORKBENCH_KEEP = 3           # per agent: never trimmed below this many runs

_SYSTEM = """You are the analyst for an auto-research run that edits SGLang to
lower the cost per output token of serving Qwen3.8-27B-FP8 on one H100 under a
latency SLO. The harness prices every candidate with a concurrency sweep
(N* = the highest concurrency that held the SLO; bill = $ per 1,000 market
requests at 20,583 input / 2,076 output tokens), checks accuracy with GSM8K
and a teacher-forced token-equivalence gate, replicates claimed wins, and
records verdicts in a memory with a 3% noise floor (win / loss / neutral).
Stock's baseline is the one each experiment's delta is against.

Answer from the run data below. Be specific: name agents, experiment ids,
files and numbers. Say plainly when the data does not support an answer.
Prefer short answers; use a list only for parallel items.

{context}"""


def build_context(root: str | pathlib.Path, k_diffs: int = 3, diff_chars: int = 6000,
                  view=None, calls_per_agent: int = 4, budget: int = CONTEXT_CHARS) -> str:
    """Everything worth reading about a run, as text. Stable between
    questions so it caches; the question is the only thing that changes.

    `view` is a live `SessionView` when the fleet is still running -- the
    store is fresher than the summary on disk by up to ten seconds. Without
    one, the fleet's state comes from `summary.json`, which the fleet
    rewrites while running and once more as it stops."""
    root = pathlib.Path(root)
    parts = [f"## Run root\n{root}"]
    cfg = root / "fleet.json"
    if cfg.is_file():
        with contextlib.suppress(ValueError, OSError):
            with_note = json.loads(cfg.read_text())
            keep = {k: with_note.get(k) for k in ("session_id", "agents", "model", "mode",
                                                  "baseline", "seeds", "bank", "manager",
                                                  "budget_usd", "note")}
            parts.append("## Fleet config\n" + json.dumps(keep, indent=1))
    summary = rs.load_summary(root)
    if view is not None:
        parts.append("## Fleet now (live)\n" + rs.snapshot_text(view, root))
    elif summary.get("snapshot"):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(summary.get("written_at", 0)))
        parts.append(f"## Fleet at last summary ({when})\n"
                     + rs.snapshot_text(summary["snapshot"], root))
    if summary.get("outcomes"):
        lines = []
        for o in summary["outcomes"]:
            best = o.get("best_delta") or {}
            d = best.get("bill_per_1k_pct")
            lines.append(f"- {o.get('agent_id')} {o.get('idea_id')} {o.get('title')!r}: "
                         f"stopped on {o.get('stop')} after {o.get('attempts')} attempts, "
                         f"${float(o.get('cost_usd') or 0):.2f}; best "
                         f"{o.get('best_experiment') or '-'}"
                         + (f" ({d:+.1f}%)" if isinstance(d, (int, float)) else "")
                         + (f"; {o['note']}" if o.get("note") else ""))
        parts.append("## Idea outcomes (in order finished)\n" + "\n".join(lines))
    rows = rs.leaderboard(root)
    parts.append("## Leaderboard (best first)\n" + rs.summary_text(root, k=40))
    for r in rows:
        if r.summary:
            parts.append(f"### {r.experiment_id} ({r.agent_id}, {r.verdict})\n"
                         f"hypothesis: {r.hypothesis}\nresult: {r.summary}\n"
                         + (f"quality: {r.quality}\n" if r.quality else "")
                         + (f"n*: {r.n_star}  rank: {r.rank}  share: {r.share_pct:.2f}%\n"
                            if r.share_pct is not None else ""))
    shown = 0
    for r in rows:
        if shown >= k_diffs or r.delta_pct is None:
            break
        d = rs.diff_for(root, r, limit=diff_chars)
        if d:
            parts.append(f"## Diff for {r.experiment_id} ({r.delta_pct:+.1f}%)\n```\n{d}\n```")
            shown += 1
    with contextlib.suppress(Exception):
        from .timeline import render_text
        tl = render_text(root)
        if tl and tl != "no traces yet":
            parts.append("## Timeline (from the traces)\n" + tl[:12000])
    agents = rs._agent_dirs(root)
    calls = []
    for d in agents:
        for c in rs.recent_calls(root, d.name, k=calls_per_agent):
            calls.append(f"{d.name} {c['phase']:<6} {c['min']:5.1f} min {c['msgs']:3} msgs "
                         f"out {c['out']:,} cache {c['cache']:,}"
                         + (f"  tools {c['tools']}" if c["tools"] else ""))
    if calls:
        parts.append("## Recent model calls (per agent, oldest first)\n" + "\n".join(calls))
    profiles = _profiles_text(root, rows)
    if profiles:
        parts.append("## GPU profiles (tracedb, one per stack measured)\n" + profiles)
    for d in agents:
        attempts = _attempts_text(d)
        if attempts:
            parts.append(f"## Attempts: {d.name} (from each report.txt)\n" + attempts)
        spend = _spend_text(d)
        if spend:
            parts.append(f"## Modal spend ledger: {d.name}\n" + spend)
        design = _design_text(d)
        if design:
            parts.append(f"## Design note: {d.name} (candidate/sglang/DESIGN.md, head)\n" + design)
    with contextlib.suppress(Exception):
        from .paper import find_papers
        papers = find_papers(root)
        if papers:
            parts.append("## Papers\n" + "\n".join(f"{k}: {v}" for k, v in sorted(papers.items())))
    tools = root / "tools" / "README.md"
    if tools.is_file():
        parts.append("## Shared tools (manager)\n" + tools.read_text()[:4000])
    runs = {d.name: [_workbench_text(w) for w in workbench_runs(d)] for d in agents}
    return _fit(parts, {k: v for k, v in runs.items() if v}, budget)


# ── the agent's own GPU tool calls ────────────────────────────────────────

def workbench_runs(agent_dir: str | pathlib.Path) -> list[dict]:
    """Every `<agent>/workbench-N/` in order, summarised: which tool it was
    (from the ledger's `where`, else the script's own markers), the
    script's first docstring line, whether it ran, exit code, cost, time,
    the last lines of stdout, the first of stderr when it failed, and the
    ncu table or equivalence agreement when the result carries one."""
    d = pathlib.Path(agent_dir)
    dirs = sorted((p for p in d.glob("workbench-*") if p.is_dir() and p.name[10:].isdigit()),
                  key=lambda p: int(p.name[10:]))
    named = {}
    for ln in rs._ledger_lines(d):
        if ln.get("where"):
            named[pathlib.PurePath(ln["where"]).name] = str(ln.get("tool") or "")
    out = []
    for w in dirs:
        script = _read(w / "script.py")
        res = {}
        with contextlib.suppress(OSError, ValueError):
            res = json.loads((w / "result.json").read_text()) if (w / "result.json").is_file() else {}
        stdout = res.get("stdout") if isinstance(res.get("stdout"), str) else _read(w / "stdout.txt")
        stderr = res.get("stderr") if isinstance(res.get("stderr"), str) else _read(w / "stderr.txt")
        ok = res.get("ok")
        run = {"dir": w.name, "agent": d.name, "tool": named.get(w.name) or _tool_of(script),
               "purpose": _script_head(script), "ok": ok,
               "exit": res.get("exit_code"), "cost_usd": res.get("cost_usd"),
               "elapsed_s": res.get("elapsed_s"),
               "stdout_tail": (stdout or "").strip().splitlines()[-8:],
               "stderr_head": ((stderr or "").strip().splitlines()[:4] if ok is False else []),
               "no_result": not res}
        eq = res.get("result") if isinstance(res.get("result"), dict) else res
        agree = {k: eq[k] for k in ("top1_agreement", "mean_abs_dlogprob", "max_abs_dlogprob",
                                    "decode_agreement", "decode_exact", "n") if k in eq}
        if agree:
            run["equivalence"] = agree
            if "regressed" in res:
                run["equivalence"]["regressed"] = res["regressed"]
        if isinstance(res.get("ncu"), (dict, list)):
            run["ncu"] = res["ncu"]
        out.append(run)
    return out


def _read(p: pathlib.Path, limit: int = 400_000) -> str:
    try:
        return p.read_text(errors="replace")[:limit] if p.is_file() else ""
    except OSError:
        return ""


def _tool_of(script: str) -> str:
    if "NCU_JSON" in script:
        return "ncu"
    if "PROMPTSET_DIGEST" in script and "MODE" in script:
        return "equivalence"
    return "gpu-run"


def _script_head(script: str) -> str:
    """The first line of the script's docstring, else its first two lines."""
    text = script.lstrip()
    for q in ('"""', "\'" * 3):
        if text.startswith(q):
            body = text[3:].split(q, 1)[0]
            for line in body.splitlines():
                if line.strip():
                    return line.strip()[:200]
            return ""
    return " / ".join(ln.strip() for ln in text.splitlines()[:2] if ln.strip())[:200]


def _workbench_text(w: dict) -> str:
    if w["no_result"]:
        status = "no result (killed, or still running)"
    else:
        status = (f"{'ok' if w['ok'] else 'FAILED'} exit {w['exit']}  "
                  f"${float(w['cost_usd'] or 0):.2f}  {float(w['elapsed_s'] or 0):.0f}s")
    lines = [f"### {w['agent']}/{w['dir']}  [{w['tool']}]  {status}",
             f"purpose: {w['purpose'] or '-'}"]
    if w.get("equivalence"):
        lines.append("equivalence: " + json.dumps(w["equivalence"]))
    if w.get("ncu"):
        lines.append("ncu: " + json.dumps(w["ncu"])[:800])
    if w["stdout_tail"]:
        lines.append("stdout (tail):\n  " + "\n  ".join(ln.strip()[:240] for ln in w["stdout_tail"]))
    if w["stderr_head"]:
        lines.append("stderr (head):\n  " + "\n  ".join(ln.strip()[:240] for ln in w["stderr_head"]))
    return "\n".join(lines)


def _attempts_text(agent_dir: pathlib.Path) -> str:
    """Per `runs/attempt-*/report.txt`: the stack, the N* line, the bill,
    the quality gates and any caveat -- the report's verdict lines."""
    keys = ("n* =", "whole bill", "quality", "caveat", "gate", "reject", "regress", "fail",
            "stack:")
    out = []
    for rep in sorted(agent_dir.glob("runs/*/report.txt")):
        text = _read(rep, 60_000)
        picked = [ln.strip() for ln in text.splitlines()
                  if any(k in ln.lower() for k in keys) and not ln.startswith("    ")
                  and "OpenRouter" not in ln]
        if not picked:
            picked = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
        out.append(f"{rep.parent.name}:\n  " + "\n  ".join(p[:200] for p in picked[:12]))
    return "\n".join(out)


def _spend_text(agent_dir: pathlib.Path) -> str:
    s = rs.spend_summary(agent_dir)
    if not s["by_tool"] and not s["on_disk"]:
        return ""
    lines = [f"ledger {tool}: {v['n']} calls, ${v['usd']:.2f}"
             for tool, v in sorted(s["by_tool"].items())]
    lines.append(f"workbench results on disk: ${s['on_disk']:.2f}")
    lines.append(f"not in the fleet's own cost figure: ${s['unreported']:.2f}")
    return "\n".join(lines)


def _design_text(agent_dir: pathlib.Path, lines: int = 60) -> str:
    p = agent_dir / "candidate" / "sglang" / "DESIGN.md"
    text = _read(p, 40_000)
    # indented so the note's own headings do not read as sections here
    return "\n".join("    " + ln for ln in text.splitlines()[:lines]) if text else ""


def _profiles_text(root: pathlib.Path, rows) -> str:
    """`<root>/profiles/*.sqlite`, each with the stack digest it was
    captured for (the file name, less an agent prefix) and the experiment
    that digest belongs to when the memory knows it."""
    d = root / "profiles"
    if not d.is_dir():
        return ""
    by_digest = {}
    for r in rows:
        by_digest.setdefault(r.stack_digest, r)
    out = []
    for f in sorted(d.glob("*.sqlite")):
        stem = f.stem
        digest = stem.split("-", 1)[1] if "-" in stem and stem.split("-", 1)[0].startswith("a") else stem
        r = by_digest.get(digest)
        who = (f"{r.experiment_id} ({r.agent_id}, {r.verdict})" if r else "no experiment recorded")
        out.append(f"{f.name}: stack {digest}  {who}")
    return "\n".join(out)


def _fit(parts: list[str], runs: dict[str, list[str]], budget: int) -> str:
    """Join everything under `budget` characters. Over it, drop the oldest
    workbench runs first (round-robin from the agent with the most, never
    below `WORKBENCH_KEEP` each); if that is not enough, take the excess
    from every section of any size, each in proportion to its length."""
    dropped = dict.fromkeys(runs, 0)
    sep = 2

    def total() -> int:
        return (sum(len(p) + sep for p in parts)
                + sum(len(t) + sep for v in runs.values() for t in v)
                + sum(80 for _ in runs))
    while total() > budget:
        big = max(runs, key=lambda k: len(runs[k]), default=None)
        if big is None or len(runs[big]) <= WORKBENCH_KEEP:
            break
        runs[big].pop(0)
        dropped[big] += 1
    sections = list(parts)
    for agent, texts in runs.items():
        head = (f"## GPU tool calls: {agent} ({len(texts)} shown, oldest first"
                + (f"; {dropped[agent]} older omitted" if dropped[agent] else "") + ")")
        sections.append(head + "\n\n" + "\n\n".join(texts))
    n = sum(len(x) + sep for x in sections)
    if n > budget:
        big = [i for i, x in enumerate(sections) if len(x) > 500]
        pool = sum(len(sections[i]) for i in big) or 1
        excess = n - budget
        for i in big:
            cut = int(excess * len(sections[i]) / pool) + 16
            sections[i] = sections[i][:max(0, len(sections[i]) - cut)] + "\n…[truncated]"
    return "\n\n".join(sections)


@dataclass
class Asker:
    """A conversation about one run. `client` is injectable for tests.

    `view_source`, when given, returns the fleet's current `SessionView` (or
    None once it has stopped) and the context is rebuilt from it before
    every question: a running fleet changes between questions, and an
    answer about "now" must not describe ten minutes ago. Without one the
    context is built once, from the run directory, and cached."""
    root: pathlib.Path
    model: str = DEFAULT_MODEL
    client: object | None = None
    history: list[dict] = field(default_factory=list)
    context: str = ""
    last_usage: dict = field(default_factory=dict)
    view_source: object | None = None

    def __post_init__(self):
        self.root = pathlib.Path(self.root)
        if not self.context:
            self.refresh()

    def refresh(self) -> str:
        view = self.view_source() if self.view_source is not None else None
        self.context = build_context(self.root, view=view)
        return self.context

    def _client(self):
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic()
        return self.client

    def ask(self, question: str) -> str:
        if self.view_source is not None:
            self.refresh()
        self.history.append({"role": "user", "content": question})
        system = [{"type": "text", "text": _SYSTEM.format(context=self.context),
                   "cache_control": {"type": "ephemeral"}}]
        client = self._client()
        # Streamed so a long answer cannot hit the request timeout; the
        # server-side fallback re-runs a refused request on another model
        # inside the same call.
        with client.beta.messages.stream(
                model=self.model, max_tokens=16000, system=system,
                messages=self.history,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if getattr(msg, "stop_reason", "") == "refusal" and not text:
            text = "(the model declined to answer this)"
        self.history.append({"role": "assistant", "content": text})
        u = getattr(msg, "usage", None)
        if u is not None:
            self.last_usage = {"input": getattr(u, "input_tokens", 0),
                               "output": getattr(u, "output_tokens", 0),
                               "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}
        return text
