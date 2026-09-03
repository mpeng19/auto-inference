"""One agent's working directory, and how its edits become a stack.

This is the whole interface between an agent and the thing it is trying to
improve. `candidate/sglang/` holds a copy of the stock files the agent was
pointed at; the agent (a Claude Code process with that directory as its cwd)
edits them in place, and the loop reads the result back:

    ws = Workspace("agents/<session>/a01")
    ws.materialise("srt/layers/attention/triton_backend.py")   # stock copy to edit
    ...                                                        # the agent edits
    ok, why = ws.check()                     # parses? actually changed?
    stack = ws.stack()                       # simulator.InferenceStack, priced by the evaluator

`edit` and `replace` write a file programmatically, for scripted proposers
and tests. Four decisions worth stating.

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

**A base, when the fleet compounds.** With `base=` (or a `base.json` in the
agent directory, which is how `harness tool` finds it from the agent's shell)
the workspace's "stock" is that saved stack: `read` shows the base's files,
`touched` means "differs from the base", and `stack()` returns the *full*
stack -- base files plus the agent's edits, base serving and env with the
agent's `serving.json` layered on top -- so the evaluator, the quality cache
and the equivalence reference all see one digest that already includes the
base.
"""
from __future__ import annotations

import ast
import difflib
import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .stock import SGLANG_VERSION, BaseSource, StockSource, stock

if TYPE_CHECKING:
    from simulator.stack import InferenceStack


@dataclass
class Workspace:
    """A per-agent directory: candidates, traces, and one dir per evaluation."""
    root: str | pathlib.Path
    agent_id: str = ""
    sglang_version: str = SGLANG_VERSION
    source: StockSource | None = None
    # The stack this workspace's edits are relative to (see the module
    # docstring). None means stock; when None and `<root>/base.json` exists,
    # that file is the base, so tools run from the agent's shell agree with
    # the daemon that wrote it.
    base: InferenceStack | None = None

    BASE_FILE = "base.json"

    def __post_init__(self):
        self.root = self.locate(self.root)
        if isinstance(self.source, BaseSource):
            self.base = self.source.base
        elif self.base is None and (self.root / self.BASE_FILE).is_file():
            from simulator import InferenceStack
            self.base = InferenceStack.load(self.root / self.BASE_FILE)
        self.source = self.source or stock(self.sglang_version)
        if self.base is not None and not isinstance(self.source, BaseSource):
            self.source = BaseSource(self.base, self.source)
        for d in (self.candidates, self.traces, self.runs):
            d.mkdir(parents=True, exist_ok=True)
        if self.base is not None:
            self._persist_base()

    def set_base(self, base: InferenceStack | None) -> None:
        """Make `base` the stack this workspace builds on (None: stock), and
        record it in `<root>/base.json` so a later `Workspace(root)` -- the
        agent's own `harness tool` calls -- sees the same base."""
        under = self.source.stock if isinstance(self.source, BaseSource) else self.source
        self.base = base
        self.source = BaseSource(base, under) if base is not None else under
        if base is None:
            (self.root / self.BASE_FILE).unlink(missing_ok=True)
        else:
            self._persist_base()

    def _persist_base(self) -> None:
        p = self.root / self.BASE_FILE
        text = json.dumps(self.base.as_dict(), sort_keys=True)
        if not p.is_file() or p.read_text() != text:
            p.write_text(text)

    @property
    def base_name(self) -> str:
        """"stock", or the base's digest: what the diff is against."""
        return f"base {self.base.digest}" if self.base is not None else "stock"

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

    def reset(self) -> None:
        """Back to stock (or the base): the loop calls this before every attempt."""
        shutil.rmtree(self.candidates, ignore_errors=True)
        self.candidates.mkdir(parents=True, exist_ok=True)

    # ── inspecting the proposal ─────────────────────────────────────────
    RESERVED = ("candidate", "runs", "traces", "sglang")

    @staticmethod
    def locate(root) -> pathlib.Path:
        """The agent directory, given it or its `candidate/` subdirectory.

        Agents run their tools from inside `candidate/` with `--workspace .`;
        treating that as an agent root created a nested workspace there and
        re-materialised the targets over the agent's edits. An agent root is
        the directory that *contains* `candidate/`.
        """
        p = pathlib.Path(root)
        if p.name == "sglang" and p.parent.name == "candidate":
            return p.parent.parent          # candidate/sglang -> the agent dir
        if p.name == "candidate" and (p / "sglang").is_dir():
            return p.parent                 # candidate -> the agent dir
        return p

    def materialise(self, *rels: str) -> tuple[str, ...]:
        """Write stock copies into the workspace so an editor can open them.

        Needed because the agent is a Claude Code process working in this
        directory, not a function handed a string: it wants real files to read
        and edit in place. Copies are only made where none exists, so calling
        this twice never discards an agent's work.

        A target that does not exist in this SGLang version is skipped and
        listed in `missing_targets`, not raised: idea records are written
        against papers and books, not against 0.5.18's file layout, and one
        stale path must not throw away the whole idea.
        """
        out = []
        missing = []
        for rel in rels:
            dst = self.candidates / rel
            if dst.is_file():
                out.append(rel)
                continue
            try:
                text = self.source.read(rel)
            except (OSError, FileNotFoundError):
                missing.append(rel)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
            out.append(rel)
        self.missing_targets = tuple(missing)
        return tuple(out)

    def touched(self) -> tuple[str, ...]:
        """Files that actually differ from stock -- or from the base, if the
        workspace has one.

        Not "files present": an agent editing in place starts from a stock copy
        of everything it might touch, and most of those stay identical. A stack
        containing an unmodified file is a no-op wearing a diff's clothes.
        """
        out = []
        for f in sorted(self.candidates.rglob("*.py")):
            rel = str(f.relative_to(self.candidates))
            if rel.split("/", 1)[0] in self.RESERVED:
                continue                 # a misplaced tree; check() reports it
            try:
                if f.read_text() != self.source.read(rel):
                    out.append(rel)
            except (OSError, FileNotFoundError):
                out.append(rel)          # not in stock at all: genuinely new
        return tuple(out)

    def _in_stock(self, rel: str) -> bool:
        try:
            self.source.read(rel)
            return True
        except (OSError, FileNotFoundError):
            return False

    def _upstream_sha(self, rel: str) -> str | None:
        """The hash of the upstream file `rel` was derived from, or None for a
        file that is new. With a base this is the *stock* hash underneath,
        not the base's text: drift is measured against the installed
        package, which is what `apply` compares with."""
        try:
            if isinstance(self.source, BaseSource):
                return self.source.stock_sha(rel)
            return self.source.sha(rel)
        except (OSError, FileNotFoundError):
            return None

    def install_skills(self, skills: dict[str, str]) -> tuple[pathlib.Path, ...]:
        """Write `.claude/skills/<name>/SKILL.md` into the candidate directory,
        which is the agent's cwd, so Claude Code discovers them as project
        skills. Not Python, so never part of the diff. Rewritten every call:
        the serving-facts skill changes as the bank does."""
        out = []
        for name, text in skills.items():
            if not text:
                continue
            d = self.candidates / ".claude" / "skills" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(text)
            out.append(d / "SKILL.md")
        return tuple(out)

    def misplaced(self) -> tuple[str, ...]:
        """Top-level directories in the candidate that cannot be part of the
        package: a nested `sglang/` (the agent treated this as a repo root),
        or a nested workspace. Either means files the container would never
        load, so it is an error, not a warning."""
        return tuple(sorted(d.name for d in self.candidates.iterdir()
                            if d.is_dir() and d.name in self.RESERVED))

    def diff(self, rel: str | None = None, context: int = 3) -> str:
        """Unified diff against stock, plus the launch overrides if any.
        For the agent to review and for the log."""
        rels = (rel,) if rel else self.touched()
        out = []
        against = "base" if self.base is not None else "stock"
        if rel is None:
            serving, env, _ = self.serving()
            if serving or env:
                out.append(f"--- launch: {against}\n+++ launch: candidate\n")
                out.extend(f"+{k}={v}\n" for k, v in sorted(serving.items()))
                out.extend(f"+env {k}={v}\n" for k, v in sorted(env.items()))
        for r in rels:
            a = self.stock_text(r).splitlines(keepends=True)
            b = self.read(r).splitlines(keepends=True)
            out.extend(difflib.unified_diff(
                a, b, fromfile=f"{against}/{r}", tofile=f"candidate/{r}", n=context))
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
            cfg = ServingConfig()
            if self.base is not None:
                cfg = cfg.with_overrides(self.base.serving)   # the launch line it lands on
            cfg.with_overrides(serving)
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
        bad = self.misplaced()
        if bad:
            return False, (f"{', '.join(bad)}/ inside the candidate: this directory IS the "
                           "sglang package root (srt/ is here); move files up a level")
        serving, env, why = self.serving()
        if why:
            return False, why
        if not self.touched() and not serving and not env:
            return False, ("no files changed and no serving.json; the stack would be "
                           f"identical to {'the base' if self.base is not None else 'stock'}")
        return True, ""

    # ── handing it to the simulator ─────────────────────────────────────
    def stack(self, label: str = ""):
        """Build the `simulator.InferenceStack` this workspace describes.

        With a base, the **full** stack: `InferenceStack.compose(base,
        edits)`. The agent's edits alone would be a diff against files the
        runner does not have, and a digest that collides with the same edit
        made on stock.
        """
        from simulator import InferenceStack

        ok, why = self.check()
        if not ok:
            raise ValueError(f"workspace is not a valid stack: {why}")
        rels = self.touched()
        serving, env, _ = self.serving()
        what = list(rels) + [f"{k}={v}" for k, v in sorted(serving.items())]
        shas = {r: self._upstream_sha(r) for r in rels}
        overlay = InferenceStack(
            files={r: self.read(r) for r in rels},
            # Recorded from the same source the agent read, so drift is caught
            # at apply time instead of producing a quietly wrong experiment.
            upstream_sha={r: h for r, h in shas.items() if h},
            serving=serving, env=env,
            label=label or f"{self.agent_id or self.root.name}: {', '.join(what)}")
        if self.base is None:
            return overlay
        return InferenceStack.compose(self.base, overlay)
