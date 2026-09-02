"""A Proposer backed by Claude Code, run as a subprocess in the workspace.

Deliberately *not* a hand-rolled agent loop. Claude Code already reads files,
edits them, runs commands and keeps its own context; rebuilding that badly is
weeks of work for something worse. What this harness adds is the part Claude
Code does not have -- a fleet, a shared memory of every experiment, a priced
evaluation of the diff -- and the cheapest way to combine them is to hand
Claude Code a workspace and a brief and let it work.

    claude -p "<task>" --model sonnet --output-format json
           --permission-mode acceptEdits          (cwd = the agent's candidate tree)

Three choices worth stating.

**The subscription, not the API.** `claude` billed through a Max/Team plan is
the cheap way to run ten agents for hours; the same traffic over the API is
not. That is why this shells out rather than calling the SDK with a key -- and
why it **strips `ANTHROPIC_API_KEY` from the subprocess environment**. Claude
Code prefers an API key over the subscription when both are present, so simply
inheriting the parent environment silently bills the wrong account. On this
machine that key is set for unrelated reasons, and the first real fleet run
printed "claude.ai connectors are disabled because ANTHROPIC_API_KEY ... takes
precedence" before every call. Set `use_api_key=True` to opt in deliberately.

**Model per phase, not per fleet.** Seeding an idea and reviewing a diff are
short; writing the diff is the long, expensive part. `model` picks the default
and `seed_model` can be cheaper. Prefer `opus` or `sonnet` here -- a reasoning-
heavy frontier model spends its budget thinking about a task whose difficulty
lives in the codebase, not in the prompt.

**Edits happen in the workspace, and we read the diff back.** We do not ask for
a patch in the response. Claude Code is good at editing files and bad at
emitting context-perfect unified diffs, and `Workspace.touched()` already knows
what changed -- so the file tree is the interface, and the response is only
used for the rationale.

Token usage comes back in the JSON envelope, which is what makes per-agent
cost visible on the dashboard without any estimation.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field

from ..contracts.agent import Attempt, Idea
from ..contracts.memory import Brief
from ..contracts.session import TokenUse
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


class ClaudeCodeUnavailable(RuntimeError):
    pass


@dataclass
class ClaudeCodeProposer:
    """Runs Claude Code in the agent's workspace and reads back the diff."""

    model: str = "sonnet"
    seed_model: str = ""            # defaults to `model`
    timeout_s: float = 900.0
    targets: tuple[str, ...] = DEFAULT_TARGETS
    binary: str = "claude"
    permission_mode: str = "acceptEdits"
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

    def _run(self, prompt: str, cwd, model: str = "") -> tuple[str, TokenUse]:
        if not self.available():
            raise ClaudeCodeUnavailable(
                f"{self.binary!r} is not on PATH. The fleet runs agents as "
                "Claude Code processes; install it or pass a different Proposer.")
        cmd = [self.binary, "-p", prompt,
               "--model", model or self.model,
               "--output-format", "json",
               "--permission-mode", self.permission_mode,
               *(("--mcp-config", self.mcp_config) if self.mcp_config else ()),
               *self.extra_args]
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=self.timeout_s, env=self._env())
        if r.returncode != 0:
            raise RuntimeError(
                f"claude exited {r.returncode}: {(r.stderr or r.stdout)[-800:]}")
        text, use = _parse(r.stdout)
        self.last_usage = use
        if self.on_tokens is not None:
            # Accounting must never take an experiment down.
            with contextlib.suppress(Exception):
                self.on_tokens(use)
        return text, use

    # ── the Proposer surface ─────────────────────────────────────────────
    def seed(self, live_ideas: tuple[Idea, ...], brief: Brief) -> Idea:
        taken = "\n".join(f"  - {i.title}: {i.hypothesis}" for i in live_ideas)
        prompt = _SEED_PROMPT.format(
            brief=brief.text or "(nothing on record yet)",
            taken=taken or "  (none)",
            targets="\n".join(f"  - {t}" for t in self.targets))
        text, _ = self._run(prompt, cwd=os.getcwd(),
                            model=self.seed_model or self.model)
        return _parse_idea(text, self.targets)

    def edit(self, ws: Workspace, idea: Idea, brief: Brief, attempt: int,
             history: tuple[Attempt, ...]) -> str:
        # Give it real files to open. Without this the first thing it does is
        # discover the directory is empty.
        ws.materialise(*(idea.targets or self.targets))
        prompt = _EDIT_PROMPT.format(
            hypothesis=idea.hypothesis, title=idea.title, attempt=attempt,
            brief=brief.text or "(nothing on record yet)",
            history=_history(history),
            files="\n".join(f"  - {t}" for t in (idea.targets or self.targets)))
        text, _ = self._run(prompt, cwd=str(ws.candidates))
        return text.strip()[:4000]

    def study(self, ws: Workspace, idea: Idea, brief: Brief,
              history: tuple[Attempt, ...]) -> str:
        """Runs while a GPU sweep is in flight, so it must not edit anything."""
        prompt = _STUDY_PROMPT.format(
            hypothesis=idea.hypothesis, brief=brief.text or "(nothing yet)",
            history=_history(history))
        try:
            text, _ = self._run(prompt, cwd=str(ws.candidates))
        except Exception as e:
            return f"study skipped: {e}"
        return text.strip()[:2000]


# ── plumbing ─────────────────────────────────────────────────────────────

def _parse(stdout: str) -> tuple[str, TokenUse]:
    """Pull the result text and token usage out of the JSON envelope."""
    try:
        d = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, TokenUse()
    if isinstance(d, list):                 # stream-json, if anyone sets it
        d = next((x for x in reversed(d) if isinstance(x, dict)
                  and x.get("type") == "result"), {})
    u = d.get("usage") or {}
    return str(d.get("result", "")), TokenUse(
        input=int(u.get("input_tokens", 0)),
        output=int(u.get("output_tokens", 0)),
        cache_read=int(u.get("cache_read_input_tokens", 0)),
        cache_write=int(u.get("cache_creation_input_tokens", 0)))


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

Make the smallest edit that tests the hypothesis. Constraints:
  - Python must parse; a syntax error wastes a GPU sweep.
  - Do not add imports of packages SGLang does not already depend on.
  - Do not change public function signatures other modules call.
  - Prefer one file. A diff spanning five files cannot be attributed.

When done, reply with 2-4 sentences: what you changed and the mechanism by
which it should lower cost per output token."""

_STUDY_PROMPT = """A GPU evaluation of your current diff is running; it will
take 25-60 minutes. Do NOT edit any files -- the diff under test is frozen.

Your hypothesis: {hypothesis}
What the fleet knows:
{brief}
Your attempts so far:
{history}

Spend this time reading the code around your change and answer, in under 200
words: if this attempt shows no improvement, what is the single most likely
reason, and what would you change next?"""
