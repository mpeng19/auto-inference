"""The manager: one process-wide reviewer that keeps a run diverse and
turns what agents keep re-deriving into tools the next agent can call.

Two jobs, deliberately small:

  assignment   which bank record an agent gets. Already a property of the
               bank's claim (least similar to what is live or tried); the
               manager only decides *whether* to draw from the bank or let
               an agent self-seed, and records what happened.
  stashing     after every few outcomes, read what the agents did and decide
               whether a reusable tool -- a script under `<root>/tools/` --
               would save real time for the agents still to come. The prompt
               makes the bar explicit: a tool is written only when the
               manager can name the hours it saves; a tool nobody calls is
               worse than none, because every agent reads the index.

The manager is a model call with a small state file, not a fourth agent
thread holding a GPU slot. That keeps its cost to a few calls a night.
"""
from .manager import Manager, ToolStash

__all__ = ["Manager", "ToolStash"]
