"""Ask a model about a run.

The morning question is rarely "what is the number" -- the status line
answers that -- but "why did a02's third attempt lose", "which of these
diffs is worth keeping", "did anyone touch the attention backend". That is a
reading job over memory, traces and the bank, and it is what this hands to
Claude: the run's leaderboard, the diffs behind its best results, the
manager's tools, and the bank claims, as a cached system context, then the
conversation on top.

Goes through the Anthropic API directly (not the Claude Code subscription),
because it is a question about data, not an edit to a repository, and the
API key is already in the environment. Model defaults to Claude Fable 5.1.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from . import results as rs

DEFAULT_MODEL = "claude-fable-5-1"

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


def build_context(root: str | pathlib.Path, k_diffs: int = 3, diff_chars: int = 6000) -> str:
    """Everything worth reading about a run, as text. Stable between
    questions so it caches; the question is the only thing that changes."""
    root = pathlib.Path(root)
    parts = [f"## Run root\n{root}"]
    cfg = root / "fleet.json"
    if cfg.is_file():
        with_note = json.loads(cfg.read_text())
        keep = {k: with_note.get(k) for k in ("session_id", "agents", "model", "mode",
                                              "baseline", "seeds", "bank", "manager",
                                              "budget_usd", "note")}
        parts.append("## Fleet config\n" + json.dumps(keep, indent=1))
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
    tools = root / "tools" / "README.md"
    if tools.is_file():
        parts.append("## Shared tools (manager)\n" + tools.read_text()[:4000])
    return "\n\n".join(parts)


@dataclass
class Asker:
    """A conversation about one run. `client` is injectable for tests."""
    root: pathlib.Path
    model: str = DEFAULT_MODEL
    client: object | None = None
    history: list[dict] = field(default_factory=list)
    context: str = ""
    last_usage: dict = field(default_factory=dict)

    def __post_init__(self):
        self.root = pathlib.Path(self.root)
        if not self.context:
            self.context = build_context(self.root)

    def _client(self):
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic()
        return self.client

    def ask(self, question: str) -> str:
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
