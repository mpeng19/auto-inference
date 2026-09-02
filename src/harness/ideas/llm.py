"""The one model call the producers share.

Runs through the same Claude Code binary the agents use, so it bills the
same subscription and strips the same API auth from the environment. Kept
behind a plain `str -> str` so every producer stays testable with a lambda.
"""
from __future__ import annotations

import pathlib


def ask_with(model: str = "opus", cwd: str | pathlib.Path | None = None):
    """A `prompt -> text` callable backed by `claude -p`."""
    from ..agent.claude_code import ClaudeCodeProposer

    proposer = ClaudeCodeProposer(model=model)
    where = str(cwd or pathlib.Path.cwd())

    def ask(prompt: str) -> str:
        text, _ = proposer._run(prompt, cwd=where)
        return text

    return ask
