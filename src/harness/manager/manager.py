"""Reference manager: batch the outcomes, ask one question, write the answer down.

Everything that decides anything is in the two prompts below. The code around
them only enforces the bars those prompts state -- an hours-saved figure before
a script is stashed, an existing fact to contradict before one is superseded --
and makes sure a malformed reply can never take an agent down with it, because
this runs on an agent's own thread as its idea finishes.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..contracts.agent import AgentOutcome

_REVIEW_PROMPT = """You manage a fleet of coding agents that edit SGLang to lower
cost per output token. Each agent works alone in its own copy of the package,
runs `harness tool` commands from its shell, and gets a GPU evaluation of its
diff. You do not write experiments. You decide whether a REUSABLE TOOL would
save the agents still to come real time -- a script every agent can run from
its workspace, checked into the run's shared tools directory.

Recent outcomes (newest last):
{outcomes}

Tools already in the shared directory:
{index}

The bar is explicit. Write a tool only if you can name at least one agent-hour
it saves across the rest of this run (an agent-hour is ~$4 of Claude Code and
GPU time). Good tools: a harness for micro-benchmarking a decode attention
kernel against the stock one on random inputs; a script that dumps the shapes
and dtypes flowing through a given layer; a checker for a mistake two agents
already made. Bad tools: anything an agent can do with one shell command, or
that only one hypothesis would ever use.

You also keep the SKILL BANK: general, falsifiable facts about serving this
model on this hardware that a future run should know -- not what one diff
scored, but what the evidence established ("FP8 greedy decoding is not
deterministic across batch compositions; a 2-point GSM8K gate rejects
stock"). Write a fact only when the outcomes above are evidence for it.
Facts already held:
{facts}

Reply with ONE JSON object and nothing else:
  {{"tool": null,
    "facts": [{{"claim": "<one falsifiable sentence>", "topic": "<short handle>",
               "evidence": "<numbers from the outcomes>", "confidence": <0-1>}}],
    "notes": "<one sentence why no tool this round>"}}
or with "tool" filled:
  {{"tool": {{"name": "<snake_case>", "purpose": "<one line, imperative>",
             "usage": "<one line: how an agent runs it from its workspace>",
             "hours_saved": <number>, "code": "<complete python source>"}},
   "facts": [], "notes": "<one sentence>"}}
`facts` may be empty. The tool code must be a single self-contained Python
file that runs with the package's dependencies (torch, triton, sglang) and
prints its findings."""

_JUDGE_PROMPT = """A new fact is being added to a bank of facts about LLM serving.
Which of the existing facts on the same topic does it CONTRADICT (not merely
refine or restate compatibly)? A contradiction is one where both cannot be
true. Reply with a JSON list of the ids of contradicted facts, e.g. ["fact_1"],
or [] if none.

New fact: {new}

Existing facts:
{existing}"""


@dataclass
class ToolStash:
    """`<root>/tools/`: scripts plus an index every agent's prompt carries."""
    root: pathlib.Path

    @property
    def dir(self) -> pathlib.Path:
        d = pathlib.Path(self.root) / "tools"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add(self, name: str, purpose: str, usage: str, code: str,
            hours_saved: float, by: str = "manager") -> pathlib.Path:
        name = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_") or "tool"
        path = self.dir / f"{name}.py"
        path.write_text(code if code.endswith("\n") else code + "\n")
        entry = {"name": name, "file": path.name, "purpose": purpose, "usage": usage,
                 "hours_saved": hours_saved, "by": by, "ts": time.time()}
        with (self.dir / "index.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._render()
        return path

    def entries(self) -> list[dict]:
        p = self.dir / "index.jsonl"
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def index(self) -> str:
        """What an agent is told. Empty string when there is nothing."""
        rows = self.entries()
        if not rows:
            return ""
        return "\n".join(f"  {self.dir / e['file']}: {e['purpose']}\n      {e['usage']}"
                         for e in rows)

    def _render(self) -> None:
        rows = self.entries()
        lines = ["# Shared tools for this run", "",
                 "Written by the manager when it judged a script would save the",
                 "agents still to come at least an agent-hour. Run from a workspace.", ""]
        for e in rows:
            lines += [f"## {e['name']}", "", e["purpose"], "", f"    {e['usage']}", "",
                      f"estimated saving: {e['hours_saved']} agent-hours", ""]
        (self.dir / "README.md").write_text("\n".join(lines))


@dataclass
class Manager:
    """Reviews outcomes in batches; stashes a tool when the bar is met."""
    root: pathlib.Path
    ask: Callable[[str], str]
    every: int = 3
    min_hours_saved: float = 1.0
    skills: object | None = None          # SkillBankService; the manager writes it
    session_id: str = ""
    stash: ToolStash = field(init=False)
    _pending: list[AgentOutcome] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    reviews: int = field(default=0, init=False)
    log: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.root = pathlib.Path(self.root)
        self.stash = ToolStash(self.root)

    def tools_index(self) -> str:
        return self.stash.index()

    def on_outcome(self, out: AgentOutcome) -> None:
        """Called by the fleet as ideas finish. Reviews every `every`th."""
        with self._lock:
            self._pending.append(out)
            if len(self._pending) < self.every:
                return
            batch, self._pending = self._pending, []
        # Off the fleet's lock, and never allowed to take an agent down.
        with contextlib.suppress(Exception):
            self.review(batch)

    def review(self, batch: list[AgentOutcome]) -> dict | None:
        held = ""
        if self.skills is not None:
            with contextlib.suppress(Exception):
                held = "\n".join(f"  - [{f.topic}] {f.claim} ({f.id})"
                                  for f in self.skills.list()[-30:])
        prompt = _REVIEW_PROMPT.format(outcomes=_describe(batch),
                                       index=self.stash.index() or "  (none yet)",
                                       facts=held or "  (none yet)")
        text = self.ask(prompt)
        self.reviews += 1
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            self.log.append("review: no JSON in reply")
            return None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            self.log.append("review: bad JSON")
            return None
        self._record_facts(d.get("facts") or [])
        tool = d.get("tool")
        note = str(d.get("notes", ""))
        if not tool:
            self.log.append(f"review: no tool ({note})")
            return d
        hours = float(tool.get("hours_saved") or 0)
        if hours < self.min_hours_saved or not tool.get("code"):
            self.log.append(f"review: tool {tool.get('name')} below the bar ({hours}h)")
            return d
        path = self.stash.add(tool["name"], tool.get("purpose", ""), tool.get("usage", ""),
                              tool["code"], hours)
        self.log.append(f"review: stashed {path.name} ({hours}h: {note})")
        return d

    # ── the skill bank ──────────────────────────────────────────────────
    def _record_facts(self, facts: list) -> None:
        if self.skills is None or not facts:
            return
        from ..contracts.skills import Fact

        for f in facts:
            if not isinstance(f, dict) or not f.get("claim"):
                continue
            fact = Fact(claim=str(f["claim"])[:400], topic=str(f.get("topic") or "general")[:40],
                        evidence=str(f.get("evidence") or "")[:600],
                        source=self.session_id or str(self.root.name),
                        confidence=float(f.get("confidence") or 0.6))
            try:
                fid, losers = self.skills.add(fact, judge=self.judge)
                self.log.append(f"fact {fid} [{fact.topic}]"
                                + (f" supersedes {', '.join(losers)}" if losers else ""))
            except Exception as e:
                self.log.append(f"fact rejected: {type(e).__name__}: {e}")

    def judge(self, new, existing) -> tuple[str, ...]:
        """The contradiction judge: the model, asked one narrow question."""
        if not existing:
            return ()
        text = self.ask(_JUDGE_PROMPT.format(
            new=f"{new.claim}  (evidence: {new.evidence})",
            existing="\n".join(f"  {f.id}: {f.claim}" for f in existing)))
        m = re.search(r"\[.*?\]", text, re.S)
        if not m:
            return ()
        try:
            ids = json.loads(m.group(0))
        except json.JSONDecodeError:
            return ()
        known = {f.id for f in existing}
        return tuple(str(i) for i in ids if str(i) in known)


def _describe(batch: list[AgentOutcome]) -> str:
    rows = []
    for o in batch:
        best = o.best.metrics.get("bill_per_1k") if o.best else None
        delta = o.best.delta.get("bill_per_1k_pct") if o.best else None
        fails = [a.failure for a in o.attempts if not a.ok and a.failure]
        rows.append(f"- {o.agent_id} | {o.idea.title} | stop={o.stop} | attempts={len(o.attempts)}"
                    + (f" | best ${best:.2f}/1k ({delta:+.1f}%)" if best is not None and delta is not None else "")
                    + (f" | failures: {', '.join(fails[:4])}" if fails else "")
                    + (f"\n    {o.note[:300]}" if o.note else ""))
    return "\n".join(rows) or "  (none)"
