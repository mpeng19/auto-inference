"""The fleet and the evaluation queue.

`Fleet` implements `harness.contracts.OrchestrationService` and, for its
agents, `AgentControl`: it claims one bank record per agent, runs each
agent's loop on a thread, keeps ideas diverse, enforces the fleet budget,
and publishes to the `SessionStore` so a TUI or `harness status` can watch
and command it. `EvalBroker` implements `EvalService`: the submit/collect
queue in front of the GPUs, with content dedup, screen-tier reservation and
priority ordering.

    broker = EvalBroker(runner, capacity=2)          # runner(EvalRequest) -> (ok, metrics, failure)
    fleet = Fleet(None, broker, store=..., session_id=..., root=...)
    fleet.bank = ...; fleet.make_agent = ...; fleet.start(FleetSpec(...)); fleet.stop()

Writes `<root>/timeline.md`, `<root>/fleet.json`-adjacent state through the
session store, and the agents' own directories through `make_agent`.
`harness.daemon` is the only caller that assembles all of it.
"""
from .broker import EvalBroker
from .fleet import Fleet

__all__ = ["EvalBroker", "Fleet"]
