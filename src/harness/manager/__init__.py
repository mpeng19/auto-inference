"""The manager: one reviewer per run, reading outcomes the agents cannot see.

Two jobs, deliberately small:

  stashing     after every few outcomes, read what the agents did and decide
               whether a reusable tool -- a script under `<root>/tools/` --
               would save real time for the agents still to come. The prompt
               makes the bar explicit: a tool is written only when the
               manager can name the hours it saves; a tool nobody calls is
               worse than none, because every agent reads the index.
  facts        the same review writes the skill bank: general, falsifiable
               claims about serving this model on this hardware, each with its
               evidence. A claim that contradicts a held one supersedes it,
               and the model is the judge of what "contradicts" means.

Which bank record an agent gets is *not* decided here. That is the bank's own
claim rule -- the available record least like what is live or tried -- in
`harness.ideas`.

The manager is a model call with a small state file, not a fourth agent
thread holding a GPU slot. That keeps its cost to a few calls a night.
"""
from .manager import Manager, ToolStash

__all__ = ["Manager", "ToolStash"]
