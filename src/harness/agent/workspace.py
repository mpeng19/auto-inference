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
import json
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

    def run_dir(self, attempt: int = 0, suffix: str = "") -> pathlib.Path:
        """Where one evaluation writes. Created here, because the simulator
        refuses to invent its own output directory; `suffix` names a second
        measurement of the same attempt (`-rep1`)."""
        d = self.runs / f"attempt-{attempt:03d}{suffix}"
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
    def materialise(self, *rels: str) -> tuple[str, ...]:
        """Write stock copies into the workspace so an editor can open them.

        Needed because the agent is a Claude Code process working in this
        directory, not a function handed a string: it wants real files to read
        and edit in place. Copies are only made where none exists, so calling
        this twice never discards an agent's work.
        """
        out = []
        for rel in rels:
            dst = self.candidates / rel
            if not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(self.source.read(rel))
            out.append(rel)
        return tuple(out)

    def touched(self) -> tuple[str, ...]:
        """Files that actually differ from stock.

        Not "files present": an agent editing in place starts from a stock copy
        of everything it might touch, and most of those stay identical. A stack
        containing an unmodified file is a no-op wearing a diff's clothes --
        exactly what the deleted `overlays/` directory turned out to be.
        """
        out = []
        for f in sorted(self.candidates.rglob("*.py")):
            rel = str(f.relative_to(self.candidates))
            try:
                if f.read_text() != self.source.read(rel):
                    out.append(rel)
            except (OSError, FileNotFoundError):
                out.append(rel)          # not in stock at all: genuinely new
        return tuple(out)

    def diff(self, rel: str | None = None, context: int = 3) -> str:
        """Unified diff against stock, plus the launch overrides if any.
        For the agent to review and for the log."""
        rels = (rel,) if rel else self.touched()
        out = []
        if rel is None:
            serving, env, _ = self.serving()
            if serving or env:
                out.append("--- launch: stock\n+++ launch: candidate\n")
                out.extend(f"+{k}={v}\n" for k, v in sorted(serving.items()))
                out.extend(f"+env {k}={v}\n" for k, v in sorted(env.items()))
        for r in rels:
            a = self.stock_text(r).splitlines(keepends=True)
            b = self.read(r).splitlines(keepends=True)
            out.extend(difflib.unified_diff(
                a, b, fromfile=f"stock/{r}", tofile=f"candidate/{r}", n=context))
        return "".join(out)

    SERVING_FILE = "serving.json"

    def serving(self) -> tuple[dict, dict, str]:
        """The candidate's launch overrides: (serving, env, error).

        `serving.json` in the candidate directory, shaped
        `{"serving": {...ServingConfig fields...}, "env": {...}}` -- or the
        flat form `{...ServingConfig fields...}`. Validated here against the
        real ServingConfig so a typo is caught before a GPU is rented.
        """
        p = self.candidates / self.SERVING_FILE
        if not p.is_file():
            return {}, {}, ""
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            return {}, {}, f"{self.SERVING_FILE}: not JSON ({e.msg} at line {e.lineno})"
        if not isinstance(d, dict):
            return {}, {}, f"{self.SERVING_FILE}: expected an object"
        if "serving" in d or "env" in d:
            serving, env = dict(d.get("serving") or {}), dict(d.get("env") or {})
        else:
            serving, env = dict(d), {}
        env = {str(k): str(v) for k, v in env.items()}
        try:
            from simulator.config import ServingConfig
            ServingConfig().with_overrides(serving)
        except ValueError as e:
            return {}, {}, f"{self.SERVING_FILE}: {e}"
        return serving, env, ""

    def check(self) -> tuple[bool, str]:
        """Everything worth knowing before renting a GPU.

        Also parses every *present* file, not just the changed ones: an agent
        editing in place can leave a syntax error in a file it then reverted,
        and finding that out six GPU-minutes in costs about a dollar a typo.
        """
        for f in sorted(self.candidates.rglob("*.py")):
            rel = str(f.relative_to(self.candidates))
            try:
                ast.parse(f.read_text())
            except SyntaxError as e:
                return False, f"{rel}: syntax error at line {e.lineno}: {e.msg}"
        serving, env, why = self.serving()
        if why:
            return False, why
        if not self.touched() and not serving and not env:
            return False, ("no files changed and no serving.json; the stack would be "
                           "identical to stock")
        return True, ""

    # ── handing it to the simulator ─────────────────────────────────────
    def stack(self, label: str = ""):
        """Build the `simulator.InferenceStack` this workspace describes."""
        from simulator import InferenceStack

        ok, why = self.check()
        if not ok:
            raise ValueError(f"workspace is not a valid stack: {why}")
        rels = self.touched()
        serving, env, _ = self.serving()
        what = list(rels) + [f"{k}={v}" for k, v in sorted(serving.items())]
        return InferenceStack(
            files={r: self.read(r) for r in rels},
            # Recorded from the same source the agent read, so drift is caught
            # at apply time instead of producing a quietly wrong experiment.
            upstream_sha={r: self.source.sha(r) for r in rels},
            serving=serving, env=env,
            label=label or f"{self.agent_id or self.root.name}: {', '.join(what)}")
