"""A Proposer backed by Claude Code, run as a subprocess in the workspace.

Deliberately *not* a hand-rolled agent loop. Claude Code already reads files,
edits them, runs commands and keeps its own context; rebuilding that badly is
weeks of work for something worse. What this harness adds is the part Claude
Code does not have -- a fleet, a shared memory of every experiment, a priced
evaluation of the diff -- and the cheapest way to combine them is to hand
Claude Code a workspace and a brief and let it work.

    claude -p "<task>" --model sonnet --output-format json
           --permission-mode acceptEdits --allowedTools "Bash(harness tool:*)" ...
           (cwd = the agent's candidate tree)

Four choices worth stating.

**The subscription, not the API.** `claude` billed through a Max/Team plan is
the cheap way to run several agents for hours; the same traffic over the API
is not. That is why this shells out rather than calling the SDK with a key --
and why it **strips `ANTHROPIC_API_KEY` from the subprocess environment**.
Claude Code prefers an API key over the subscription when both are present, so
inheriting the parent environment silently bills the wrong account: the only
sign is a "claude.ai connectors are disabled because ANTHROPIC_API_KEY ...
takes precedence" line before each call. Set `use_api_key=True` to opt in
deliberately.

**The tools have to be allowed, or the agent guesses.** `acceptEdits`
auto-approves edits and a handful of filesystem commands; every other Bash
command still asks, and in headless mode there is nobody to ask, so it is
refused. With only the permission mode set, agents never ran `harness tool
preflight` or `recall` once and reasoned from memory instead, reporting the
refusal as a choice. `allowed_tools` is passed as `--allowedTools`, which is
*additive* on top of the permission mode: it grants the harness tools without
turning the run into `--dangerously-skip-permissions`.

**The job is a kernel-scale build, not a tweak.** The idea comes from the bank
with a mechanism, targets, an expected gain and its risks written down, and
the prompt asks for the change at that scale: a design note first, a
multi-file diff, and a correctness and micro-benchmark run on a real GPU
before a sweep is spent. Asking for "the smallest edit that tests the
hypothesis" produced one-line constant changes inside the measurement noise.

**Edits happen in the workspace, and we read the diff back.** We do not ask for
a patch in the response. Claude Code is good at editing files and bad at
emitting context-perfect unified diffs, and `Workspace.touched()` already knows
what changed -- so the file tree is the interface, and the response is only
used for the rationale.

Token usage comes back in the JSON envelope, which is what makes per-agent
cost visible on the dashboard without any estimation. `CallStats` carries the
rest of that envelope -- duration, API duration, turn count -- so a trace can
say *why* an agent took two hours.

**Skills.** Before every edit and before the paper step, the harness's own
skills (`harness/skills/docs/*/SKILL.md`: `tracedb`, `writeup`) and the
rendered skill bank (`serving-facts`) are written into `.claude/skills/` of
the directory the call runs in -- the candidate tree for an edit, the paper
directory for the write-up -- which is where Claude Code discovers project
skills. `HARNESS_EXTRA_SKILLS`, a colon-separated list of skill directories
(each holding a `SKILL.md`), adds skills from outside the repo the same way:
a LaTeX document skill for the paper step, say. Each is symlinked (copied
when a symlink cannot be made) under `.claude/skills/<directory name>/`.
Nothing here names a particular path; the variable is the only way in.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

from ..contracts.agent import Attempt, Idea
from ..contracts.memory import Brief
from ..contracts.session import TokenUse
from ..profile import MCP_SERVER_NAME
from .workspace import Workspace

# Files an agent starts from when its idea names none. Not a hard limit -- it
# may edit anything under `srt/` -- but a menu, because "here are 1602
# modules" is not a useful opening position, and these are where a
# kernel-scale change lands. Every path is checked against the pinned wheel
# (`test_build_targets_exist_in_the_pinned_wheel`): a menu of paths that do
# not exist costs the agent its first ten minutes discovering that.
DEFAULT_TARGETS = (
    "srt/layers/attention/triton_backend.py",
    "srt/layers/attention/flashinfer_backend.py",
    "srt/layers/attention/flashattention_backend.py",
    "srt/layers/attention/base_attn_backend.py",
    "srt/layers/attention/merge_state.py",
    "srt/layers/radix_attention.py",
    "srt/mem_cache/memory_pool.py",
    "srt/mem_cache/radix_cache.py",
    "srt/mem_cache/allocator/__init__.py",
    "srt/managers/schedule_batch.py",
    "srt/speculative/eagle_utils.py",
    "srt/speculative/spec_info.py",
)

# An edit writes a kernel, runs it on an H100 and scores equivalence; two
# hours is the size of that job.
DEFAULT_EDIT_TIMEOUT_S = 2 * 3600

# Rules for `--allowedTools`. Two forms of Bash prefix match are documented and
# equivalent -- `Bash(cmd:*)` and `Bash(cmd *)` -- and both are listed because
# the harness tools are reached both directly and through `uv run`, and a rule
# that does not match is silently a refusal. Read/Grep/Glob need no rule (they
# are allowed by default) and are listed so this tuple reads as the agent's
# whole non-edit surface. `--allowedTools` adds permissions, it does not
# restrict: Edit and Write still come from `--permission-mode acceptEdits`.
DEFAULT_ALLOWED_TOOLS = (
    "Bash(harness tool:*)",
    "Bash(uv run harness tool:*)",
    "Bash(harness tool *)",
    "Bash(uv run harness tool *)",
    # The GPU workbench, listed explicitly as well as under the `harness tool`
    # prefixes above, so the rules survive someone narrowing that prefix.
    "Bash(harness tool gpu-run:*)",
    "Bash(harness tool equivalence:*)",
    "Bash(harness tool ncu:*)",
    "Bash(python -c:*)",
    "Bash(python3 -c:*)",
    "Bash(ruff:*)",
    # Reading and running inside its own copy of the package. Without these
    # an agent was refused eight commands in one call and gave up before the
    # workbench; a sandboxed copy of sglang is not worth guarding command by
    # command.
    "Bash(python:*)", "Bash(python3:*)", "Bash(harness:*)", "Bash(uv run harness:*)",
    "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)",
    "Bash(grep:*)", "Bash(rg:*)", "Bash(find:*)", "Bash(sed -n:*)", "Bash(diff:*)",
    "Bash(git diff:*)", "Bash(git status:*)", "Bash(mkdir:*)", "Bash(pwd)", "Bash(tree:*)",
    "Read",
    "Grep",
    "Glob",
)


class ClaudeCodeUnavailable(RuntimeError):
    pass


@dataclass
class CallStats:
    """What one `claude -p` call cost in time, and how it ended.

    `wall_s` comes from `time.time()` and not `time.monotonic()` on purpose.
    On 2026-09-02 a closed lid froze three agents for five hours; every call
    looked instantaneous afterwards, because the monotonic clock stops while
    the host sleeps -- which is *also* why `subprocess.run(timeout=...)` never
    fired and the calls were never killed. A wall clock that counts the sleep
    is the difference between "the model is slow" and "the machine was
    asleep", and that distinction was unavailable for a whole night.

    `duration_ms` / `duration_api_ms` / `num_turns` / `is_error` / `denials`
    come from the claude JSON envelope, so wall time and the model's own
    accounting can be compared: a large `wall_s` with a small `duration_ms` is
    the host, not the model.
    """
    phase: str = ""
    model: str = ""
    started_at: float = 0.0
    wall_s: float = 0.0
    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    is_error: bool = False
    # How many tool calls the permission system refused. An agent that
    # "decided not to run preflight" was in fact told it could not, and the
    # write-up reports the refusal as if it were a choice; this is the only
    # place the difference shows.
    denials: int = 0
    returncode: int = 0
    cancelled: bool = False
    timed_out: bool = False
    transient: bool = False       # failed for the API's reasons (529/429/5xx): worth a retry
    # From the stream: how many model responses the call took and what they
    # cost, in aggregate. The per-message rows are in the calls log.
    n_messages: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    log_path: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClaudeCodeProposer:
    """Runs Claude Code in the agent's workspace and reads back the diff."""

    # `claude --model`. Prefer opus or sonnet: a reasoning-heavy frontier
    # model spends its budget thinking about a task whose difficulty lives in
    # the codebase, not in the prompt.
    model: str = "sonnet"
    # The study and paper calls; an edit gets `DEFAULT_EDIT_TIMEOUT_S`
    # unless `edit_timeout_s` says otherwise.
    timeout_s: float = 900.0
    edit_timeout_s: float = 0.0
    targets: tuple[str, ...] = DEFAULT_TARGETS
    binary: str = "claude"
    permission_mode: str = "acceptEdits"
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    extra_args: tuple[str, ...] = ()
    # Path to a `--mcp-config` JSON. When a fleet has captured a GPU profile,
    # this points the agent at `tracedb`'s tools so it can ask where the time
    # actually went instead of reasoning about it.
    mcp_config: str = ""
    # What `mcp_config` falls back to when an attempt has no profile yet.
    mcp_config_default: str = ""
    # Off by default: an inherited API key silently bills the wrong account.
    use_api_key: bool = False
    # Set by the agent loop so token use lands on the right dashboard row.
    on_tokens: object | None = None
    # The manager's shared tools for this run, as an index the agent reads
    # with the brief. A callable, because the index grows while the run is
    # on and the prompt must carry what exists *now*.
    session_tools: object | None = None
    # The skill bank, rendered: what earlier runs established. A callable
    # for the same reason as `session_tools`.
    session_skills: object | None = None
    # Where per-call event logs go (`<agent>/calls/<phase>-<ts>.jsonl`);
    # set by the loop's workspace before a call. Empty: no log.
    calls_dir: str = ""
    # Every call this proposer has made, newest last. The loop stamps
    # `last_call` onto the turn it appends, so a trace can be read for where
    # the hours went without keeping the proposer alive.
    last_call: CallStats = field(default_factory=CallStats)
    calls: list[CallStats] = field(default_factory=list)

    # How often the wait loop looks at the clock and the cancel flag. Small
    # enough that a cancelled study dies while the result is still landing,
    # large enough to be free.
    POLL_S = 0.2
    # SIGTERM, then SIGKILL. Claude Code flushes its transcript on TERM; a
    # process still alive after this was not going to flush anything.
    KILL_GRACE_S = 5.0

    # ── invocation ───────────────────────────────────────────────────────
    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    AUTH_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                 "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
                 "CLAUDE_CODE_USE_VERTEX")

    def _env(self) -> dict:
        """The subprocess environment, with API auth removed by default.

        Claude Code prefers an API key over the subscription when both are
        present, so inheriting the parent environment bills the wrong account
        without saying so -- it only prints a warning about connectors being
        disabled, which is easy to read past.
        """
        env = {**os.environ, "CLAUDE_CODE_NONINTERACTIVE": "1"}
        # `harness` lives in this interpreter's environment, and the agent's
        # cwd is its candidate directory, where `uv run` finds no project.
        # Put the console scripts first on PATH so `harness tool ...` resolves.
        bindir = os.path.dirname(sys.executable)
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        # Claude Code kills a shell command at 10 minutes by default and
        # `harness tool gpu-run` takes 5-15 (model load alone is 3-5). On
        # build-4 that was 21 tool calls cut at exactly 600 s: the Modal call
        # ran on and billed, the agent never saw the result and ran it again.
        env.setdefault("BASH_DEFAULT_TIMEOUT_MS", str(30 * 60 * 1000))
        env.setdefault("BASH_MAX_TIMEOUT_MS", str(60 * 60 * 1000))
        if not self.use_api_key:
            for k in self.AUTH_VARS:
                env.pop(k, None)
        return env

    def _tools(self) -> tuple[str, ...]:
        """The `--allowedTools` rules for this call.

        The trace tools are MCP tools, and MCP tools need permission too: with
        a `--mcp-config` and no rule they are offered to the agent and then
        refused, which is the same failure as the shell tools had.
        """
        tools = tuple(self.allowed_tools)
        if self.mcp_config and not any(t.startswith("mcp__") for t in tools):
            tools += (f"mcp__{MCP_SERVER_NAME}", f"mcp__{MCP_SERVER_NAME}_stock")
        return tools

    @staticmethod
    def skill_docs() -> dict[str, str]:
        """The skills shipped with the harness (`harness/skills/docs/*/SKILL.md`)."""
        base = pathlib.Path(__file__).resolve().parent.parent / "skills" / "docs"
        out = {}
        if base.is_dir():
            for d in sorted(base.iterdir()):
                f = d / "SKILL.md"
                if f.is_file():
                    out[d.name] = f.read_text()
        return out

    EXTRA_SKILLS_VAR = "HARNESS_EXTRA_SKILLS"

    @classmethod
    def extra_skill_dirs(cls) -> tuple[pathlib.Path, ...]:
        """Skill directories named by `HARNESS_EXTRA_SKILLS` (colon-separated)
        that exist and hold a `SKILL.md`. Anything else in the variable is
        ignored rather than fatal: a missing skill is a poorer paper, not a
        dead agent."""
        raw = os.environ.get(cls.EXTRA_SKILLS_VAR, "")
        out = []
        for part in raw.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            d = pathlib.Path(part).expanduser()
            if d.is_dir() and (d / "SKILL.md").is_file():
                out.append(d)
        return tuple(out)

    def _install_skills(self, ws: Workspace, into: pathlib.Path | None = None) -> None:
        """Write the skills where this call's cwd will find them: the
        candidate tree by default, `into` for a call that runs elsewhere
        (the paper step runs in the paper directory)."""
        skills = dict(self.skill_docs())
        if self.session_skills is not None:
            with contextlib.suppress(Exception):
                text = str(self.session_skills() or "")
                if text:
                    skills["serving-facts"] = text
        target = pathlib.Path(into) if into is not None else ws.candidates
        with contextlib.suppress(Exception):
            if into is None:
                ws.install_skills(skills)
            else:
                _write_skills(target, skills)
        for d in self.extra_skill_dirs():
            with contextlib.suppress(Exception):
                _link_skill(target / ".claude" / "skills" / d.name, d)

    def _mcp_for(self, ws: Workspace, history: tuple[Attempt, ...]) -> str:
        """An MCP config for this attempt: the agent's latest profile, and
        stock's if the fleet has one. Empty when there is nothing to query,
        so the agent is not offered tools that answer with an empty table."""
        from ..profile import write_mcp_config

        dbs: dict[str, str] = {}
        for a in reversed(history):
            db = (a.metrics or {}).get("profile_db")
            if db and pathlib.Path(db).is_file():
                dbs[MCP_SERVER_NAME] = db
                break
        stock = ws.root.parent / "profiles" / "stock.sqlite"
        if stock.is_file():
            dbs[f"{MCP_SERVER_NAME}_stock"] = str(stock)
        if not dbs:
            return ""
        return str(write_mcp_config(ws.root, dbs))

    def _cmd(self, prompt: str, model: str) -> list[str]:
        """The argv. `--allowedTools` is variadic, so it goes last.

        One argv token per rule rather than one comma-joined string: the flag
        takes a comma- *or* space-separated list and the rules themselves
        contain spaces (`Bash(harness tool *)`), so joining them is the one
        way to get a rule silently split in half.
        """
        return [self.binary, "-p", prompt,
                "--model", model or self.model,
                # stream-json: one event per inner request and response, each
                # with its own usage, so tokens are counted as they are spent
                # rather than when a two-hour call finally returns.
                "--output-format", "stream-json", "--verbose",
                "--permission-mode", self.permission_mode,
                *(("--mcp-config", self.mcp_config) if self.mcp_config else ()),
                *self.extra_args,
                *(("--allowedTools", *self._tools()) if self._tools() else ())]

    def _edit_timeout(self) -> float:
        return self.edit_timeout_s or DEFAULT_EDIT_TIMEOUT_S

    # Waits between retries of a call the API refused for its own reasons.
    # build-4, 09:30 on 2026-09-03: opus returned 529 Overloaded for an hour,
    # every edit failed after ~4 min, and each failure closed an idea as an
    # error and claimed the next -- the bank was being churned at one record
    # per four minutes for nothing. Three retries over ~11 min ride out the
    # usual outage; a longer one still ends in the error it always was.
    TRANSIENT_BACKOFF_S: tuple[float, ...] = (60.0, 180.0, 420.0)

    def _run(self, prompt: str, cwd, model: str = "", *,
             cancel: threading.Event | None = None,
             timeout_s: float | None = None,
             phase: str = "") -> tuple[str, TokenUse]:
        """`_run_once`, retried while the failure is the API's (see
        `TRANSIENT_BACKOFF_S`). Every attempt is a call in `self.calls`, so
        the retries are visible in the call stats. Cancellation during a
        wait re-raises the last error rather than starting another call."""
        for wait in (*self.TRANSIENT_BACKOFF_S, None):
            try:
                return self._run_once(prompt, cwd, model, cancel=cancel,
                                      timeout_s=timeout_s, phase=phase)
            except RuntimeError:
                st = self.last_call
                if wait is None or st is None or not st.transient:
                    raise
                if cancel is not None:
                    if cancel.wait(wait):
                        raise
                else:
                    time.sleep(wait)
        raise AssertionError("unreachable")

    def _run_once(self, prompt: str, cwd, model: str = "", *,
                  cancel: threading.Event | None = None,
                  timeout_s: float | None = None,
                  phase: str = "") -> tuple[str, TokenUse]:
        """One `claude -p` call: cancellable, timed, and never orphaning a child.

        `Popen` plus a polling wait rather than `subprocess.run(timeout=...)`
        for two reasons. The timeout there is measured on a **monotonic clock
        that stops while the host sleeps**, so the five-hour lid-close on
        2026-09-02 did not trip a single 15-minute timeout -- the calls simply
        looked fast. And `run` has no way to be interrupted from another
        thread, which is what a study needs when its evaluation lands early.

        The child gets its own session (`start_new_session=True`) so that
        killing it kills the tools it spawned: `claude` runs shell commands as
        children, and a SIGTERM to the leader alone leaves a `harness tool
        gpu-run` holding an H100.

        Cancellation is not an error. What the model wrote before it was cut
        off is usually still worth keeping, so this returns it with
        `last_call.cancelled` set rather than raising -- the caller decides.
        Tokens for a cancelled call are lost: the usage only arrives in the
        final JSON envelope, which is exactly what never got written.
        """
        if not self.available():
            raise ClaudeCodeUnavailable(
                f"{self.binary!r} is not on PATH. The fleet runs agents as "
                "Claude Code processes; install it or pass a different Proposer.")
        limit = self.timeout_s if timeout_s is None else timeout_s
        started = time.time()
        acct = _CallAccounting(self, phase, started)
        proc = subprocess.Popen(
            self._cmd(prompt, model), cwd=str(cwd), env=self._env(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True)
        out, err, cancelled, timed_out = self._wait(proc, limit, cancel, on_line=acct.line)

        text, use, meta = _parse(out)
        if not use.total and acct.reported.total:
            use = acct.reported            # cancelled: no envelope, but the stream counted
        acct.close()
        stats = CallStats(
            phase=phase, model=model or self.model, started_at=started,
            wall_s=round(time.time() - started, 3),
            duration_ms=int(meta.get("duration_ms", 0)),
            duration_api_ms=int(meta.get("duration_api_ms", 0)),
            num_turns=int(meta.get("num_turns", 0)),
            is_error=bool(meta.get("is_error", False)),
            denials=int(meta.get("denials", 0)),
            returncode=proc.returncode if proc.returncode is not None else -1,
            cancelled=cancelled, timed_out=timed_out,
            transient=bool(proc.returncode) and not cancelled and not timed_out
            and _transient(meta, (out or "") + (err or "")),
            n_messages=acct.n_messages, input_tokens=use.input, output_tokens=use.output,
            cache_read=use.cache_read, cache_write=use.cache_write,
            log_path=acct.log_path)
        self.last_call = stats
        self.calls.append(stats)

        if timed_out:
            raise TimeoutError(
                f"claude ({phase or 'call'}) killed after {stats.wall_s:.0f}s "
                f"wall against a {limit:.0f}s limit")
        if not cancelled and proc.returncode:
            raise RuntimeError(
                f"claude exited {proc.returncode}: {(err or out)[-800:]}")
        if self.on_tokens is not None:
            # The stream already reported each message as it landed; report
            # only what the envelope adds, so the total matches the envelope
            # and nothing is counted twice.
            rest = use - acct.reported
            if rest.total:
                with contextlib.suppress(Exception):
                    self.on_tokens(rest)
        return text, use

    def _wait(self, proc, limit: float,
              cancel: threading.Event | None, on_line=None) -> tuple[str, str, bool, bool]:
        """Poll until the child exits, the clock runs out, or `cancel` is set.

        The pipes are drained by threads because a JSON envelope from a long
        session is bigger than a pipe buffer, and a child blocked writing
        stdout can neither finish nor be timed out -- it just hangs, which is
        the classic `Popen` + `poll()` deadlock.
        """
        chunks: dict[str, list[str]] = {"out": [], "err": []}

        def pump(stream, key):
            try:
                for line in iter(stream.readline, ""):
                    chunks[key].append(line)
                    if key == "out" and on_line is not None:
                        with contextlib.suppress(Exception):
                            on_line(line)
            except (ValueError, OSError):       # closed under us by the kill
                pass
            finally:
                with contextlib.suppress(Exception):
                    stream.close()

        pumps = [threading.Thread(target=pump, args=(proc.stdout, "out"),
                                  daemon=True),
                 threading.Thread(target=pump, args=(proc.stderr, "err"),
                                  daemon=True)]
        for t in pumps:
            t.start()

        deadline = time.time() + limit if limit else float("inf")
        cancelled = timed_out = False
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            if time.time() >= deadline:
                timed_out = True
                break
            time.sleep(self.POLL_S)
        if cancelled or timed_out:
            self._terminate(proc)
        else:
            with contextlib.suppress(Exception):
                proc.wait(timeout=self.KILL_GRACE_S)
        for t in pumps:
            t.join(timeout=self.KILL_GRACE_S)
        return "".join(chunks["out"]), "".join(chunks["err"]), cancelled, timed_out

    def _terminate(self, proc) -> None:
        """SIGTERM the whole process group, then SIGKILL whatever survives."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                return
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), sig)
            deadline = time.time() + self.KILL_GRACE_S
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.05)

    # ── the Proposer surface ─────────────────────────────────────────────
    # No `seed`: ideas come from the bank (the daemon refuses to run without
    # one), so this proposer never invents its own.
    def edit(self, ws: Workspace, idea: Idea, brief: Brief, attempt: int,
             history: tuple[Attempt, ...], cancel: threading.Event | None = None) -> str:
        # Give it real files to open. Without this the first thing it does is
        # discover the directory is empty.
        files = idea.targets or self.targets
        self.calls_dir = str(ws.root / "calls")
        self._install_skills(ws)
        self.mcp_config = self._mcp_for(ws, history) or self.mcp_config_default
        present = ws.materialise(*files)
        missing = getattr(ws, "missing_targets", ())
        if not present:
            files = self.targets
            ws.materialise(*files)
        else:
            files = tuple(present)
        base = getattr(ws, "base", None)
        base_note = ""
        if base is not None:
            base_note = (f"\n**You are editing a base stack, not stock.** This tree is "
                         f"{ws.base_name}: it already carries earlier wins (its files, "
                         "serving.json and env are the starting point, and the diff and "
                         "the price are measured against it). Build on it; do not undo "
                         "what it does.\n")
        prompt = _EDIT_PROMPT.format(
            hypothesis=idea.hypothesis, title=idea.title, attempt=attempt,
            design=_indent(idea.design) or "    (the idea bank recorded none; "
                                           "work it out and say so in DESIGN.md)",
            base_note=base_note,
            brief=self._brief_text(brief, "(nothing on record yet)"),
            history=_history(history),
            files="\n".join(f"  - {t}" for t in files)
            + ("\n  (not in this SGLang version, find the equivalent code: "
               + ", ".join(missing) + ")" if missing else ""))
        text, _ = self._run(prompt, cwd=str(ws.candidates), phase="edit",
                            timeout_s=self._edit_timeout(), cancel=cancel)
        return text.strip()[:4000]

    def _brief_text(self, brief: Brief, empty: str) -> str:
        """The memory brief, plus the manager's tool index when there is one."""
        text = brief.text or empty
        index = ""
        if self.session_tools is not None:
            with contextlib.suppress(Exception):
                index = str(self.session_tools() or "")
        if index:
            text += ("\n\nShared tools for this run, written by the manager because "
                     "agents kept re-deriving them (run from your workspace):\n" + index)
        return text

    def paper(self, ws: Workspace, idea: Idea, attempts: tuple[Attempt, ...],
              baseline: float | None, diff: str,
              cancel: threading.Event | None = None) -> str:
        """The write-up as a PDF, at the end of an idea that reached a full
        sweep. The template carries the numbers; the model writes the prose."""
        from ..paper import (
            PaperInputs,
            compile_tex,
            figures_for,
            paper_dir,
            prompt_for,
            render_template,
        )
        from ..results import evidence_for

        priced = [a for a in attempts if a.ok and a.metrics.get("bill_per_1k") is not None]
        rows = []
        for a in attempts:
            q = a.metrics.get("quality") or []
            gates = ", ".join(f"{x.get('suite')} {x.get('accuracy', 0):.0%}" for x in q
                              if isinstance(x, dict)) or (a.failure or "")
            rows.append({"n": a.n, "tier": a.tier, "bill": a.metrics.get("bill_per_1k"),
                         "delta": a.delta.get("bill_per_1k_pct"),
                         "n_star": a.metrics.get("n_star"), "gates": gates})
        best_ns = sorted({a.n for a in priced if a.tier == "full"}) or sorted({a.n for a in priced})
        # The evidence for the stack the paper is about: the best full-tier
        # attempt's. Replicated means two priced full runs of that digest
        # (the loop appends the first run before keeping the worse).
        full = [a for a in priced if a.tier == "full"]
        best = min(full or priced, key=lambda a: a.metrics.get("bill_per_1k") or 0.0, default=None)
        ev: dict = {}
        if best is not None:
            n_full = sum(1 for a in full if a.stack_digest == best.stack_digest)
            d_pct = best.delta.get("bill_per_1k_pct")
            verdict = "win" if d_pct is not None and d_pct <= -3.0 else "neutral"
            with contextlib.suppress(Exception):
                ev = evidence_for(ws.root.parent, ws.root.name, best.stack_digest,
                                  baseline=baseline, replicated=n_full >= 2,
                                  verdict=verdict, metrics=best.metrics)
        inp = PaperInputs(title=idea.title, author=f"{ws.agent_id or ws.root.name} (auto-inference)",
                          attempts=rows, baseline=baseline,
                          figures=figures_for(ws.root, best_ns[-2:]),
                          evidence=ev, run_root=str(ws.root))
        d = paper_dir(ws.root, idea.id)
        tex = render_template(inp, d)
        # The write-up runs in the paper directory, so the skills go there:
        # `writeup` (what a paper is here), `tracedb` (how to cite the
        # profile) and whatever HARNESS_EXTRA_SKILLS names.
        self._install_skills(ws, into=d)
        design = ""
        for cand in (ws.candidates / "DESIGN.md", ws.root / "DESIGN.md"):
            if cand.is_file():
                design = cand.read_text()[:6000]
                break
        prompt = prompt_for(inp, idea.hypothesis, design, diff)
        _text, _ = self._run(prompt, cwd=str(d), phase="paper", timeout_s=1200.0,
                             cancel=cancel)
        pdf = compile_tex(tex)
        return str(pdf or tex)

    def study(self, ws: Workspace, idea: Idea, brief: Brief,
              history: tuple[Attempt, ...],
              cancel: threading.Event | None = None) -> str:
        """Runs while a GPU sweep is in flight, so it must not edit anything.

        `cancel` is set by the loop the moment the result lands. Studying past
        that point is time spent answering a question that has been answered,
        so the call is killed and whatever it had written is kept.
        """
        prompt = _STUDY_PROMPT.format(
            hypothesis=idea.hypothesis, brief=self._brief_text(brief, "(nothing yet)"),
            history=_history(history))
        try:
            text, _ = self._run(prompt, cwd=str(ws.candidates), phase="study",
                                cancel=cancel)
        except Exception as e:
            return f"study skipped: {e}"
        # A cancelled call keeps whatever it wrote; the loop flags the turn,
        # so the marker is not added twice.
        return text.strip()[:2000]


# ── plumbing ─────────────────────────────────────────────────────────────

def _write_skills(base: pathlib.Path, skills: dict[str, str]) -> tuple[pathlib.Path, ...]:
    """`Workspace.install_skills` for a directory that is not the candidate
    tree: `.claude/skills/<name>/SKILL.md` under `base`."""
    out = []
    for name, text in skills.items():
        if not text:
            continue
        d = base / ".claude" / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(text)
        out.append(d / "SKILL.md")
    return tuple(out)


def _link_skill(dst: pathlib.Path, src: pathlib.Path) -> None:
    """Put an external skill directory at `dst`: a symlink, so the skill
    is always current and a 30 MB skill costs nothing per call; a copy when
    the filesystem refuses. Whatever was at `dst` before is replaced."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


_TRANSIENT_STATUS = {429, 500, 502, 503, 529}
_TRANSIENT_RE = re.compile(r"Overloaded|overloaded_error|rate.?limit|API Error: 5\d\d", re.I)


def _transient(meta: dict, text: str) -> bool:
    """Did the call fail for the API's reasons rather than the prompt's?"""
    status = meta.get("api_error_status")
    if status in _TRANSIENT_STATUS:
        return True
    if meta.get("terminal_reason") == "api_error":
        return True
    return bool(_TRANSIENT_RE.search(text[-4000:]))


def _parse(stdout: str) -> tuple[str, TokenUse, dict]:
    """Pull the result text, token usage and timings out of the JSON envelope.

    A cancelled call has no envelope -- it was killed before the last line was
    written -- so the raw stdout is returned as the text and the timings are
    zero. Partial thinking is still worth keeping in the trace.
    """
    events = _events(stdout)
    if not events:
        return stdout, TokenUse(), {}
    d = next((x for x in reversed(events) if x.get("type") == "result"), None)
    if d is None:
        # Cancelled or killed before the envelope: keep what the model said.
        text = "\n".join(_assistant_text(e) for e in events if e.get("type") == "assistant")
        return text, TokenUse(), {}
    u = d.get("usage") or {}
    meta = {k: d[k] for k in ("duration_ms", "duration_api_ms", "num_turns",
                              "is_error", "api_error_status", "terminal_reason") if k in d}
    meta["denials"] = len(d.get("permission_denials") or ())
    return str(d.get("result", "")), TokenUse(
        input=int(u.get("input_tokens", 0)),
        output=int(u.get("output_tokens", 0)),
        cache_read=int(u.get("cache_read_input_tokens", 0)),
        cache_write=int(u.get("cache_creation_input_tokens", 0))), meta


def _indent(text: str, pad: str = "    ") -> str:
    """Line up a free-text block under its heading. A prompt whose sections do
    not line up reads as one paragraph, and the model treats it as one."""
    body = (text or "").strip()
    return "\n".join(pad + ln for ln in body.splitlines()) if body else ""


def _history(history: tuple[Attempt, ...]) -> str:
    if not history:
        return "  (this is the first attempt)"
    return "\n".join(
        f"  - attempt {a.n} ({a.tier}): "
        + ("failed: " + a.failure if not a.ok
           else f"bill ${a.metrics.get('bill_per_1k', '?')}/1k, "
                f"{a.delta.get('bill_per_1k_pct', 0):+.1f}% vs baseline")
        for a in history[-5:])


def _events(stdout: str) -> list[dict]:
    """stream-json is one JSON object per line; the old envelope is one
    object. Both come back as a list of events."""
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def _assistant_text(event: dict) -> str:
    content = (event.get("message") or {}).get("content") or []
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


def _usage(u: dict | None) -> TokenUse:
    u = u or {}
    return TokenUse(input=int(u.get("input_tokens", 0)), output=int(u.get("output_tokens", 0)),
                    cache_read=int(u.get("cache_read_input_tokens", 0)),
                    cache_write=int(u.get("cache_creation_input_tokens", 0)))


class _CallAccounting:
    """Counts a call's stream as it happens: every assistant message's usage
    goes to `on_tokens` immediately and to the call log as a row, so the
    dashboard moves during a two-hour edit and the log answers "what did the
    forty-first turn cost, and which tool did it call"."""

    def __init__(self, proposer: ClaudeCodeProposer, phase: str, started: float):
        self.p = proposer
        self.reported = TokenUse()
        self.n_messages = 0
        self.prev_ts = started
        self.log_path = ""
        self._fh = None
        if proposer.calls_dir:
            try:
                d = pathlib.Path(proposer.calls_dir)
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{phase or 'call'}-{int(started)}.jsonl"
                self._fh = path.open("a")
                self.log_path = str(path)
            except OSError:
                self._fh = None

    def line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            return
        d = json.loads(line)
        kind = d.get("type")
        now = time.time()
        row = {"ts": round(now, 3), "since_prev_s": round(now - self.prev_ts, 3), "type": kind}
        if kind == "assistant":
            msg = d.get("message") or {}
            use = _usage(msg.get("usage"))
            content = msg.get("content") or []
            tools = [b.get("name") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            row.update({"input": use.input, "output": use.output, "cache_read": use.cache_read,
                        "cache_write": use.cache_write, "tools": tools,
                        "text_chars": len(_assistant_text(d))})
            self.reported = self.reported + use
            self.n_messages += 1
            if self.p.on_tokens is not None and use.total:
                with contextlib.suppress(Exception):
                    self.p.on_tokens(use)
        elif kind == "user":
            content = (d.get("message") or {}).get("content") or []
            row["tool_result_chars"] = sum(
                len(str(b.get("content", ""))) for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result")
        elif kind == "result":
            row.update({"duration_ms": d.get("duration_ms"), "num_turns": d.get("num_turns"),
                        "is_error": d.get("is_error"), "total": _usage(d.get("usage")).__dict__})
        else:
            return
        self.prev_ts = now
        if self._fh is not None:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()


_EDIT_PROMPT = """You are building a kernel-scale change to SGLang. This
directory is a copy of SGLang's `sglang/` package; edit the files here
directly. This is not a tuning task -- constant tweaks land inside the
measurement noise and are not worth a sweep.

The idea, as the idea bank recorded it:
  title:      {title}
  mechanism:  {hypothesis}
  expected gain, risks, and how it is meant to work:
{design}
  targets it expects to touch:
{files}

This is attempt {attempt}.
{base_note}
What the fleet already knows:
{brief}

Your previous attempts on this idea:
{history}

**Write DESIGN.md first**, in this directory, before you edit anything. It is
not part of the diff (only `.py` files are) and nothing reads it but you and
the next agent on this idea. Keep it under a page and cover three things:
  1. the kernel or algorithm change, concretely enough to implement -- what
     runs instead of what, and why that is fewer bytes moved or fewer
     launches, not just "faster";
  2. every file you intend to touch, and what changes in each;
  3. how you will know it is CORRECT before a sweep is spent -- which stock
     path you compare against, on what inputs, and what tolerance.

Then build it. Multi-file diffs are expected here. You may add new Triton
kernels or a CUDA extension; you may NOT add a pip dependency -- if it is not
already in SGLang's requirements it does not exist on the serving container.

Your workbench, in order. Run all three before you finish:
  harness tool preflight --workspace .
      parses everything and checks for undefined names. Free. A NameError
      costs six GPU-minutes to discover on a GPU.
  harness tool gpu-run <script.py>
      runs your script on an H100 inside this exact stack. ~$1, 5-15 minutes
      (model load alone is 3-5). Your shell tool allows 30 minutes per
      command here; keep one gpu-run per command and do not run it in the
      background -- a killed command still bills the GPU and loses the result.
  harness tool ncu <script.py> --kernel <regex>
      hardware counters per kernel from Nsight Compute: DRAM and SM
      throughput as % of peak, occupancy, L2 hit rate. tracedb says which
      kernel and how long; this says why. Profile a decode step, not a sweep.
      Write a script that does BOTH: a micro-benchmark of the kernel you
      changed (old path vs new, same shapes, timed properly with warmup), and
      a correctness check against the stock path on random inputs -- shapes
      and dtypes that match serving, max absolute and relative error printed,
      not just a boolean. This is the step that catches a kernel that is fast
      and wrong.
  harness tool equivalence
      teacher-force-scores your candidate against stock on a fixed prompt set
      and reports top-1 agreement and logprob drift. ~$1, ~6 minutes. Run it
      last, once the micro-benchmark says the kernel is worth keeping.

These commands are allowed to run without asking. If one is refused, say so in
your reply rather than guessing at what it would have printed.

**On numerics.** Every evaluation scores GSM8K on an idle server, and a stack
that answers worse is rejected however good the price looks. Changing numerics
is allowed *when that is the hypothesis* -- KV compression and lower-precision
attention are on the table -- but then equivalence and the accuracy gate are
the experiment, so report their numbers and expect to be judged on them.

**This directory is the package root.** `srt/` is right here; new files go
under it (e.g. `srt/layers/attention/my_kernel.py`) or beside it. Do not
create a `sglang/` directory here, and run tools as `harness tool <name>
--workspace .` from this directory.

**The launch line is yours too.** Write `serving.json` in this directory to
change how the server is started for your evaluation:
  {{"serving": {{"chunked_prefill_size": 16384, "mem_fraction_static": 0.90,
               "max_running_requests": 64, "schedule_policy": "lpm",
               "extra_args": ["--flag", "value"]}},
   "env": {{"SGLANG_SOME_VAR": "1"}}}}
Any ServingConfig field except model, gpu, n_gpu and enable_metrics. Stock's
launch line is the baseline, so a win must beat stock's deployment, not just
its code. `harness tool preflight` validates the file.

**`env` reaches the sweep's server, not the workbench.** `gpu-run` and
`equivalence` run in a container with its own environment, so a change gated on
an environment variable is measured *inert* there: on 2026-09-02 an env-gated
numerics change scored top-1 agreement exactly 1.0000 with |dlogprob| exactly
0.0000, which is the tell that the candidate ran stock, and cost a workbench
run to learn. Export the variable inside your script, or make the change
default-on with an env kill switch.

When done, reply with 4-8 sentences: the mechanism you implemented, and the
numbers off your workbench -- micro-benchmark speedup, the correctness error
you measured against stock, and what equivalence reported."""

_STUDY_PROMPT = """A GPU evaluation of your current diff is running; it
will take 25-60 minutes. Do NOT edit any files -- the diff under test is
frozen, and the sweep is already paid for.

Your hypothesis: {hypothesis}
What the fleet knows:
{brief}
Your attempts so far:
{history}

Spend this time writing the NEXT VERIFICATION STEP, not a new plan. In under
250 words: which specific claim in DESIGN.md is still unverified, and the
exact `harness tool gpu-run` script that would settle it -- what it measures,
against which stock path, on what shapes, and what result would falsify the
mechanism. Name the number that would make you abandon this idea.

You may be interrupted the moment the result lands. Write it as you go rather
than saving it for the end."""
