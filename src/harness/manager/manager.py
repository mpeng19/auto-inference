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

Reply with ONE JSON object and nothing else:
  {{"tool": null, "notes": "<one sentence why nothing this round>"}}
or
  {{"tool": {{"name": "<snake_case>", "purpose": "<one line, imperative>",
             "usage": "<one line: how an agent runs it from its workspace>",
             "hours_saved": <number>, "code": "<complete python source>"}},
   "notes": "<one sentence>"}}
The code must be a single self-contained Python file that runs with the
package's dependencies (torch, triton, sglang) and prints its findings."""


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
        prompt = _REVIEW_PROMPT.format(outcomes=_describe(batch),
                                       index=self.stash.index() or "  (none yet)")
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
