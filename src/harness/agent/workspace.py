"""One agent's working directory, and the API it uses to propose a diff.

This is the whole interface between an agent and the thing it is trying to
improve. An agent reads stock SGLang, writes candidate files, and hands the
result to the simulator:

    ws = Workspace("agents/a01")
    src = ws.read("srt/managers/schedule_policy.py")
    ws.replace("srt/managers/schedule_policy.py", old_snippet, new_snippet)
    ok, why = ws.check()                     # parses? actually changed?
    result = await Simulator(root_dir=ws.run_dir(), stack=ws.stack()).eval()

Four decisions worth stating.

**Whole files, not patch application.** The agent edits text and we compute the
diff; we never ask it to produce a unified diff that has to apply cleanly.
Patch application is a second failure mode with its own error messages, and
`InferenceStack` carries files by value anyway, so the patch would only be
un-applied at the far end.

**`upstream_sha` is filled in automatically.** Every candidate records the hash
of the stock file it was derived from, so if the pinned SGLang moves, the stack
refuses to apply rather than silently reverting upstream changes. Doing this by
hand is how a manifest goes stale.

**Cheap checks before expensive ones.** `check()` parses the candidate and
confirms it actually differs. A syntax error costs nothing here and six GPU
minutes if it reaches the runner, which is roughly a dollar per typo.

**No sandbox, deliberately.** Agents write text files and call an API; the
modified SGLang is only ever executed inside a fresh Modal container. The
isolation that matters is already paid for, and the resource agents genuinely
contend over is GPU concurrency, which the orchestrator gates.
"""
from __future__ import annotations

import ast
import difflib
import pathlib
import shutil
from dataclasses import dataclass

from .stock import SGLANG_VERSION, StockSource, stock


@dataclass
class Workspace:
    """A per-agent directory: candidates, traces, and one dir per evaluation."""
    root: str | pathlib.Path
    agent_id: str = ""
    sglang_version: str = SGLANG_VERSION
    source: StockSource | None = None

    def __post_init__(self):
        self.root = pathlib.Path(self.root)
        self.source = self.source or stock(self.sglang_version)
        for d in (self.candidates, self.traces, self.runs):
            d.mkdir(parents=True, exist_ok=True)

    # ── layout ──────────────────────────────────────────────────────────
    @property
    def candidates(self) -> pathlib.Path:
        return self.root / "candidate" / "sglang"

    @property
    def traces(self) -> pathlib.Path:
        return self.root / "traces"

    @property
    def runs(self) -> pathlib.Path:
        return self.root / "runs"

    def run_dir(self, attempt: int = 0) -> pathlib.Path:
        d = self.runs / f"attempt-{attempt:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── reading stock ───────────────────────────────────────────────────
    def ls(self, prefix: str = "srt") -> tuple[str, ...]:
        return self.source.ls(prefix)

    def read(self, rel: str) -> str:
        """The candidate if one exists, else the stock file. What is 'current'."""
        c = self.candidates / rel
        return c.read_text() if c.is_file() else self.source.read(rel)

    def stock_text(self, rel: str) -> str:
        return self.source.read(rel)

    # ── writing candidates ──────────────────────────────────────────────
    def edit(self, rel: str, text: str) -> None:
        """Replace a whole file."""
        out = self.candidates / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

    def replace(self, rel: str, old: str, new: str, count: int = 1) -> None:
        """Targeted edit. Refuses unless `old` appears exactly `count` times.

        The ambiguity is the point: an edit that matches three places and was
        meant for one produces a diff that looks plausible and behaves
        differently, which is the most expensive kind of wrong here.
        """
        cur = self.read(rel)
        n = cur.count(old)
        if n != count:
            raise ValueError(
                f"{rel}: pattern occurs {n} times, expected {count}. "
                "Give more surrounding context so the edit is unambiguous.")
        self.edit(rel, cur.replace(old, new))

    def reset(self, rel: str | None = None) -> None:
        if rel is None:
            shutil.rmtree(self.candidates, ignore_errors=True)
            self.candidates.mkdir(parents=True, exist_ok=True)
        else:
            (self.candidates / rel).unlink(missing_ok=True)

    # ── inspecting the proposal ─────────────────────────────────────────
    def touched(self) -> tuple[str, ...]:
        return tuple(sorted(str(p.relative_to(self.candidates))
                            for p in self.candidates.rglob("*.py")))

    def diff(self, rel: str | None = None, context: int = 3) -> str:
        """Unified diff against stock. For the agent to review and for the log."""
        rels = (rel,) if rel else self.touched()
        out = []
        for r in rels:
            a = self.stock_text(r).splitlines(keepends=True)
            b = self.read(r).splitlines(keepends=True)
            out.extend(difflib.unified_diff(
                a, b, fromfile=f"stock/{r}", tofile=f"candidate/{r}", n=context))
        return "".join(out)

    def check(self) -> tuple[bool, str]:
        """Everything worth knowing before renting a GPU."""
        touched = self.touched()
        if not touched:
            return False, "no files changed; the stack would be identical to stock"
        for r in touched:
            text = self.read(r)
            try:
                ast.parse(text)
            except SyntaxError as e:
                return False, f"{r}: syntax error at line {e.lineno}: {e.msg}"
            if text == self.stock_text(r):
                return False, f"{r}: written but byte-identical to stock"
        return True, ""

    # ── handing it to the simulator ─────────────────────────────────────
    def stack(self, label: str = ""):
        """Build the `simulator.InferenceStack` this workspace describes."""
        from simulator import InferenceStack

        ok, why = self.check()
        if not ok:
            raise ValueError(f"workspace is not a valid stack: {why}")
        rels = self.touched()
        return InferenceStack(
            files={r: self.read(r) for r in rels},
            # Recorded from the same source the agent read, so drift is caught
            # at apply time instead of producing a quietly wrong experiment.
            upstream_sha={r: self.source.sha(r) for r in rels},
            label=label or f"{self.agent_id or self.root.name}: {', '.join(rels)}")
