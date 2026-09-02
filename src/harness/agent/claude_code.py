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

Five choices worth stating.

**The subscription, not the API.** `claude` billed through a Max/Team plan is
the cheap way to run ten agents for hours; the same traffic over the API is
not. That is why this shells out rather than calling the SDK with a key -- and
why it **strips `ANTHROPIC_API_KEY` from the subprocess environment**. Claude
Code prefers an API key over the subscription when both are present, so simply
inheriting the parent environment silently bills the wrong account. On this
machine that key is set for unrelated reasons, and the first real fleet run
printed "claude.ai connectors are disabled because ANTHROPIC_API_KEY ... takes
precedence" before every call. Set `use_api_key=True` to opt in deliberately.

**The tools have to be allowed, or the agent guesses.** `acceptEdits`
auto-approves edits and a handful of filesystem commands; every other Bash
command still asks, and in headless mode there is nobody to ask, so it is
refused. Night-3 write-ups were full of "This command needs your approval to
run (it's a harness tool...)" -- the agents never ran `harness tool preflight`
or `recall` once and reasoned from memory instead. `allowed_tools` is passed
as `--allowedTools`, which is *additive* on top of the permission mode: it
grants the harness tools without turning the run into `--dangerously-skip-
permissions`.

**Model per phase, not per fleet.** Seeding an idea and reviewing a diff are
short; writing the diff is the long, expensive part. `model` picks the default
and `seed_model` can be cheaper. Prefer `opus` or `sonnet` here -- a reasoning-
heavy frontier model spends its budget thinking about a task whose difficulty
lives in the codebase, not in the prompt.

**Two modes, because "smallest edit" and "build a kernel" are different jobs.**
`mode="tune"` asks for the smallest edit that tests a hypothesis, and on
scheduler constants that produced fifteen one-line changes all inside the
measurement noise. `mode="build"` takes an idea that already has a mechanism,
targets, an expected gain and its risks written down, and asks for the
kernel-scale change: a design note first, a multi-file diff, and a correctness
and micro-benchmark run on a real GPU before a sweep is spent.

**Edits happen in the workspace, and we read the diff back.** We do not ask for
a patch in the response. Claude Code is good at editing files and bad at
emitting context-perfect unified diffs, and `Workspace.touched()` already knows
what changed -- so the file tree is the interface, and the response is only
used for the rationale.

Token usage comes back in the JSON envelope, which is what makes per-agent
cost visible on the dashboard without any estimation. `CallStats` carries the
rest of that envelope -- duration, API duration, turn count -- so a trace can
say *why* an agent took two hours.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..contracts.agent import Attempt, Idea
from ..contracts.memory import Brief
from ..contracts.session import TokenUse
from ..profile import MCP_SERVER_NAME
from .workspace import Workspace

# Files an agent is allowed to start from. Not a hard limit -- it may edit
# anything under `srt/` -- but a menu, because "here are 1602 modules" is not a
# useful opening position and the ones that matter for serving cost are few.
DEFAULT_TARGETS = (
    "srt/managers/schedule_policy.py",
    "srt/managers/schedule_batch.py",
    "srt/managers/scheduler.py",
    "srt/mem_cache/radix_cache.py",
    "srt/mem_cache/memory_pool.py",
)

# Where a kernel-scale change actually lands. Five scheduler files can only be
# tuned; these can be rewritten. Every path is checked against the pinned
# wheel (`test_build_targets_exist_in_stock`) because a menu of paths that do
# not exist is worse than no menu -- the agent spends its first ten minutes
# discovering that.
DEFAULT_BUILD_TARGETS = (
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

# A build turn writes a kernel, runs it on an H100 and scores equivalence.
# Fifteen minutes is a tuning budget; this is the other kind of work.
DEFAULT_BUILD_TIMEOUT_S = 2 * 3600

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
    # Built in parallel with this file: `gpu-run` puts a script on an H100
    # inside the agent's own stack (~$1, 3-8 min) and `equivalence` teacher-
    # force-scores the candidate against stock (~$1, ~6 min). Listed
    # explicitly as well as under the `harness tool` prefixes above, so the
    # rules survive someone narrowing that prefix.
    "Bash(harness tool gpu-run:*)",
    "Bash(harness tool equivalence:*)",
    "Bash(python -c:*)",
    "Bash(python3 -c:*)",
    "Bash(ruff:*)",
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
    # How many tool calls the permission system refused. Non-zero is the
    # signature of the night-3 bug -- an agent that "decided not to run
    # preflight" was in fact told it could not -- and it is invisible in the
    # write-up, which reports the refusal as if it were a choice.
    denials: int = 0
    returncode: int = 0
    cancelled: bool = False
    timed_out: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClaudeCodeProposer:
    """Runs Claude Code in the agent's workspace and reads back the diff."""

    model: str = "sonnet"
    seed_model: str = ""            # defaults to `model`
    timeout_s: float = 900.0
    # 0 means "derive from the mode": a tuning edit is a 15-minute job and a
    # build is a two-hour one, and one default cannot be both.
    edit_timeout_s: float = 0.0
    mode: Literal["tune", "build"] = "tune"
    targets: tuple[str, ...] = DEFAULT_TARGETS
    binary: str = "claude"
    permission_mode: str = "acceptEdits"
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    extra_args: tuple[str, ...] = ()
    # Path to a `--mcp-config` JSON. When a fleet has captured a GPU profile,
    # this points the agent at `tracedb`'s tools so it can ask where the time
    # actually went instead of reasoning about it.
    mcp_config: str = ""
    # Off by default: an inherited API key silently bills the wrong account.
    use_api_key: bool = False
    # Set by the agent loop so token use lands on the right dashboard row.
    on_tokens: object | None = None
    last_usage: TokenUse = field(default_factory=TokenUse)
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

    def __post_init__(self):
        # Build mode starts from the kernels, not the scheduler -- but only
        # when the caller left the menu alone. An explicit list always wins.
        if self.mode == "build" and self.targets == DEFAULT_TARGETS:
            self.targets = DEFAULT_BUILD_TARGETS

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
            tools += (f"mcp__{MCP_SERVER_NAME}",)
        return tools

    def _cmd(self, prompt: str, model: str) -> list[str]:
        """The argv. `--allowedTools` is variadic, so it goes last.

        One argv token per rule rather than one comma-joined string: the flag
        takes a comma- *or* space-separated list and the rules themselves
        contain spaces (`Bash(harness tool *)`), so joining them is the one
        way to get a rule silently split in half.
        """
        return [self.binary, "-p", prompt,
                "--model", model or self.model,
                "--output-format", "json",
                "--permission-mode", self.permission_mode,
                *(("--mcp-config", self.mcp_config) if self.mcp_config else ()),
                *self.extra_args,
                *(("--allowedTools", *self._tools()) if self._tools() else ())]

    def _edit_timeout(self) -> float:
        return self.edit_timeout_s or (
            DEFAULT_BUILD_TIMEOUT_S if self.mode == "build" else self.timeout_s)

    def _run(self, prompt: str, cwd, model: str = "", *,
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
        proc = subprocess.Popen(
            self._cmd(prompt, model), cwd=str(cwd), env=self._env(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True)
        out, err, cancelled, timed_out = self._wait(proc, limit, cancel)

        text, use, meta = _parse(out)
        stats = CallStats(
            phase=phase, model=model or self.model, started_at=started,
            wall_s=round(time.time() - started, 3),
            duration_ms=int(meta.get("duration_ms", 0)),
            duration_api_ms=int(meta.get("duration_api_ms", 0)),
            num_turns=int(meta.get("num_turns", 0)),
            is_error=bool(meta.get("is_error", False)),
            denials=int(meta.get("denials", 0)),
            returncode=proc.returncode if proc.returncode is not None else -1,
            cancelled=cancelled, timed_out=timed_out)
        self.last_call = stats
        self.calls.append(stats)

        if timed_out:
            raise TimeoutError(
                f"claude ({phase or 'call'}) killed after {stats.wall_s:.0f}s "
                f"wall against a {limit:.0f}s limit")
        if not cancelled and proc.returncode:
            raise RuntimeError(
                f"claude exited {proc.returncode}: {(err or out)[-800:]}")
        self.last_usage = use
        if self.on_tokens is not None:
            # Accounting must never take an experiment down.
            with contextlib.suppress(Exception):
                self.on_tokens(use)
        return text, use

    def _wait(self, proc, limit: float,
              cancel: threading.Event | None) -> tuple[str, str, bool, bool]:
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
    def seed(self, live_ideas: tuple[Idea, ...], brief: Brief) -> Idea:
        taken = "\n".join(f"  - {i.title}: {i.hypothesis}" for i in live_ideas)
        prompt = _SEED_PROMPT.format(
            brief=brief.text or "(nothing on record yet)",
            taken=taken or "  (none)",
            targets="\n".join(f"  - {t}" for t in self.targets))
        text, _ = self._run(prompt, cwd=os.getcwd(), phase="seed",
                            model=self.seed_model or self.model)
        return _parse_idea(text, self.targets)

    def edit(self, ws: Workspace, idea: Idea, brief: Brief, attempt: int,
             history: tuple[Attempt, ...]) -> str:
        # Give it real files to open. Without this the first thing it does is
        # discover the directory is empty.
        files = idea.targets or self.targets
        ws.materialise(*files)
        template = _BUILD_PROMPT if self.mode == "build" else _EDIT_PROMPT
        prompt = template.format(
            hypothesis=idea.hypothesis, title=idea.title, attempt=attempt,
            design=_indent(idea.design) or "    (the idea bank recorded none; "
                                           "work it out and say so in DESIGN.md)",
            brief=brief.text or "(nothing on record yet)",
            history=_history(history),
            files="\n".join(f"  - {t}" for t in files))
        text, _ = self._run(prompt, cwd=str(ws.candidates), phase="edit",
                            timeout_s=self._edit_timeout())
        return text.strip()[:4000]

    def study(self, ws: Workspace, idea: Idea, brief: Brief,
              history: tuple[Attempt, ...],
              cancel: threading.Event | None = None) -> str:
        """Runs while a GPU sweep is in flight, so it must not edit anything.

        `cancel` is set by the loop the moment the result lands. Studying past
        that point is time spent answering a question that has been answered,
        so the call is killed and whatever it had written is kept.
        """
        template = _BUILD_STUDY_PROMPT if self.mode == "build" else _STUDY_PROMPT
        prompt = template.format(
            hypothesis=idea.hypothesis, brief=brief.text or "(nothing yet)",
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

def _parse(stdout: str) -> tuple[str, TokenUse, dict]:
    """Pull the result text, token usage and timings out of the JSON envelope.

    A cancelled call has no envelope -- it was killed before the last line was
    written -- so the raw stdout is returned as the text and the timings are
    zero. Partial thinking is still worth keeping in the trace.
    """
    try:
        d = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, TokenUse(), {}
    if isinstance(d, list):                 # stream-json, if anyone sets it
        d = next((x for x in reversed(d) if isinstance(x, dict)
                  and x.get("type") == "result"), {})
    u = d.get("usage") or {}
    meta = {k: d[k] for k in ("duration_ms", "duration_api_ms", "num_turns",
                              "is_error") if k in d}
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


def _parse_idea(text: str, targets: tuple[str, ...]) -> Idea:
    """Take the first fenced JSON object; fall back to the raw text as a title."""
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return Idea(title=str(d.get("title", ""))[:80],
                        hypothesis=str(d.get("hypothesis", ""))[:400],
                        design=str(d.get("design", ""))[:4000],
                        targets=tuple(d.get("targets") or targets))
        except json.JSONDecodeError:
            pass
    first = text.strip().splitlines()[0] if text.strip() else "unnamed idea"
    return Idea(title=first[:80], hypothesis=text.strip()[:400] or first,
                targets=targets)


_SEED_PROMPT = """You are one agent in a fleet improving SGLang's serving cost.

The fleet is measured on ONE number: dollars per 1,000 marketplace requests,
at 20,583 input / 2,076 output tokens each, subject to latency SLOs. Lower is
better. Output tokens are ~95% of that bill, so decode throughput per GPU is
what matters; prefill is nearly free by comparison.

What the fleet already knows:
{brief}

Ideas other agents are working on RIGHT NOW (do not duplicate these):
{taken}

Files most likely to matter:
{targets}

Propose ONE idea nobody is working on. Reply with a single JSON object and
nothing else:
{{"title": "3-5 words", "hypothesis": "X will lower cost per output token because Y",
  "targets": ["srt/..."]}}"""

_EDIT_PROMPT = """You are improving SGLang's serving cost. This directory is a
copy of SGLang's `sglang/` package; edit the files here directly.

Your idea: {title}
Hypothesis: {hypothesis}
This is attempt {attempt}.

What the fleet already knows (read this before proposing something tried):
{brief}

Your previous attempts on this idea:
{history}

Files you started with:
{files}

Tools available in your shell (use them; they are far cheaper than a sweep):
  harness tool recall "<what you are about to try>"
      what the fleet has already tried, including what failed and why
  harness tool roofline --context 20583 --batch 12
      predicted decode step time and $/M for a batch, from first principles
      and from what this stack actually measures. The gap is the headroom.
  harness tool preflight --workspace .
      parses your edit and checks for undefined names. Run this before you
      finish; a NameError costs six GPU-minutes to discover otherwise.

These are allowed to run without asking. If a command is refused, say so in
your reply instead of guessing at what it would have printed.

**Accuracy is checked, not assumed.** Every evaluation scores GSM8K on an
idle server before load. A change that makes the model faster but answers
worse is rejected outright, however good the price looks -- so do not touch
sampling, numerics, KV precision or eviction in ways that could change what
the model says, unless testing exactly that is the hypothesis.

Make the smallest edit that tests the hypothesis. Constraints:
  - Python must parse; a syntax error wastes a GPU sweep.
  - Do not add imports of packages SGLang does not already depend on.
  - Do not change public function signatures other modules call.
  - Prefer one file. A diff spanning five files cannot be attributed.

When done, reply with 2-4 sentences: what you changed and the mechanism by
which it should lower cost per output token."""

_BUILD_PROMPT = """You are building a kernel-scale change to SGLang. This
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
      runs your script on an H100 inside this exact stack. ~$1, 3-8 minutes.
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

When done, reply with 4-8 sentences: the mechanism you implemented, and the
numbers off your workbench -- micro-benchmark speedup, the correctness error
you measured against stock, and what equivalence reported."""

_STUDY_PROMPT = """A GPU evaluation of your current diff is running; it will
take 25-60 minutes. Do NOT edit any files -- the diff under test is frozen.

Your hypothesis: {hypothesis}
What the fleet knows:
{brief}
Your attempts so far:
{history}

Spend this time reading the code around your change and answer, in under 200
words: if this attempt shows no improvement, what is the single most likely
reason, and what would you change next?

You may be interrupted the moment the result lands. Write the answer as you
go rather than saving it for the end."""

_BUILD_STUDY_PROMPT = """A GPU evaluation of your current diff is running; it
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
