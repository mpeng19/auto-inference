"""Reference AgentService: iterate on one idea until it pays or stops paying.

The LLM is deliberately **not** in this file. An agent implementation differs
from another only in how it turns "here is the code and what the fleet knows"
into an edit, so that step is a `Proposer` plug point and everything else --
budgeting, memory reads and writes, divergence, retry policy -- is the same
whatever proposes. That also means the whole loop is testable without a model.

The three guards the contract promises:

**Retries mean something.** An infrastructure failure (the GPU died, the server
never started) is retried unchanged. A rejected hypothesis is not: re-running
it produces the same number and a second bill. `Attempt.failure` carries which
kind it was and only one of them is retried.

**Divergence is measured every attempt.** An agent iterating on "improve prefix
caching" will, given enough attempts, be rewriting the scheduler. Sometimes
that is right; usually it is a lost run. Drift past the threshold stops the
loop while there is still budget to spend on a fresh idea.

**No progress stops the loop.** Attempts cost ~25 GPU-minutes. Spending the
last three to establish a dead idea more precisely is the most expensive thing
this system can do by accident.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from ..contracts.agent import AgentBudget, AgentOutcome, Attempt, Idea, Stop
from ..contracts.context import TraceMeta, Turn
from ..contracts.memory import Brief, Experiment, Recall, Relation
from .workspace import Workspace


class Proposer(Protocol):
    """Turns knowledge into an edit. The only place a model belongs."""

    def seed(self, live_ideas: tuple[Idea, ...], brief: Brief) -> Idea: ...

    def edit(self, ws: Workspace, idea: Idea, brief: Brief, attempt: int,
             history: tuple[Attempt, ...]) -> str:
        """Mutate the workspace. Returns the rationale for what it changed."""
        ...


class Evaluator(Protocol):
    """Runs one measurement. Wraps the simulator; swappable for a fake."""

    def evaluate(self, stack, run_dir) -> tuple[bool, dict, str]:
        """-> (ok, metrics, failure_kind). `failure_kind` is '' when ok."""
        ...


@dataclass
class IterativeAgent:
    """One agent loop. Owns a workspace, reads memory, writes back what it learns."""
    agent_id: str
    workspace: Workspace
    memory: object            # contracts.memory.MemoryService
    context: object           # contracts.context.ContextService
    proposer: Proposer
    evaluator: Evaluator
    fleet: object | None = None       # for eval-slot gating; None runs ungated
    baseline: dict = field(default_factory=dict)

    # ── seeding ──────────────────────────────────────────────────────────
    def propose(self, *, seed: Idea | None, live_ideas: tuple[Idea, ...]) -> Idea:
        brief = self.memory.recall(Recall(
            intent=seed.hypothesis if seed else "choose a new direction to try",
            agent_id=self.agent_id, k=10))
        if seed is not None:
            return seed
        return self.proposer.seed(live_ideas, brief)

    # ── the loop ─────────────────────────────────────────────────────────
    def run(self, idea: Idea, budget: AgentBudget) -> AgentOutcome:
        started, spent = time.time(), 0.0
        attempts: list[Attempt] = []
        best: Attempt | None = None
        since_progress = 0
        stop: Stop = "exhausted"

        trace = self.context.open(TraceMeta(
            agent_id=self.agent_id, idea_id=idea.id, model="proposer"))
        self.context.append(trace, Turn(kind="prompt", content=idea.hypothesis))

        n = 0
        while n < budget.max_attempts:
            if spent >= budget.max_usd:
                stop = "budget"
                break
            if time.time() - started > budget.max_wall_s:
                stop = "budget"
                break

            # What does the fleet already know about what I am about to do?
            brief = self.memory.recall(Recall(
                intent=idea.hypothesis, agent_id=self.agent_id, idea_id=idea.id))
            self.context.append(trace, Turn(
                kind="thought", name="recall", content=brief.text,
                tokens_in=brief.est_tokens))

            self.workspace.reset()
            try:
                rationale = self.proposer.edit(
                    self.workspace, idea, brief, n, tuple(attempts))
            except Exception as e:
                self.context.append(trace, Turn(kind="error", content=str(e)))
                stop = "error"
                break

            ok, why = self.workspace.check()
            if not ok:
                # Costs nothing to reject here; six GPU-minutes if it ships.
                attempts.append(Attempt(idea_id=idea.id, agent_id=self.agent_id,
                                        n=n, ok=False, failure="invalid_diff"))
                self.context.append(trace, Turn(kind="error", name="check", content=why))
                n += 1
                continue

            stack = self.workspace.stack(label=f"{idea.title} #{n}")
            self.context.append(trace, Turn(
                kind="eval_submit", name=stack.digest,
                content=rationale, data={"diff": self.workspace.diff()[:20000]}))

            gated = self.fleet is not None
            if gated and not self.fleet.acquire_eval_slot(self.agent_id):
                stop = "budget"
                break
            try:
                good, metrics, failure = self.evaluator.evaluate(
                    stack, self.workspace.run_dir(n))
            finally:
                if gated:
                    self.fleet.release_eval_slot(self.agent_id)

            att = Attempt(
                idea_id=idea.id, agent_id=self.agent_id, n=n,
                stack_digest=stack.digest, trace_ref=trace, ok=good,
                failure="" if good else (failure or "hypothesis"),
                metrics=metrics, delta=self._delta(metrics),
                cost_usd=float(metrics.get("cost_usd", 0.0)))
            spent += att.cost_usd
            self.context.append(trace, Turn(
                kind="eval_result", name=stack.digest, data=metrics))

            # Infrastructure failures are retried unchanged; a rejected
            # hypothesis is not, because it would cost the same and say the same.
            if not good and att.failure == "infra":
                attempts.append(att)
                continue

            exp = self._record(idea, att, rationale, trace, attempts)
            att = Attempt(**{**att.__dict__, "experiment_id": exp})
            attempts.append(att)

            drift = self.divergence(idea, att)
            if drift > budget.divergence_threshold:
                stop = "diverged"
                break

            improved = att.ok and self._improved(att, best)
            if improved:
                best, since_progress = att, 0
            else:
                since_progress += 1
            if since_progress >= budget.patience:
                stop = "no_progress"
                break
            if improved and att.delta.get("bill_per_1k_pct", 0.0) <= -5.0:
                stop = "won"
                break
            n += 1

        self.context.close(trace, outcome=stop, cost_usd=spent)
        return AgentOutcome(agent_id=self.agent_id, idea=idea, stop=stop,
                            attempts=tuple(attempts), best=best, cost_usd=spent)

    # ── guards and bookkeeping ───────────────────────────────────────────
    def divergence(self, idea: Idea, attempt: Attempt) -> float:
        """Fraction of touched files the idea never claimed it would touch.

        Crude on purpose. It is meant to catch "started on the radix cache,
        now rewriting the scheduler", not to judge research taste.
        """
        touched = set(self.workspace.touched())
        if not touched:
            return 0.0
        if not idea.targets:
            return 0.0
        claimed = set(idea.targets)
        return len(touched - claimed) / len(touched)

    def _delta(self, metrics: dict) -> dict:
        out = {}
        for k, base in self.baseline.items():
            if k in metrics and isinstance(base, (int, float)) and base:
                out[f"{k}_pct"] = round((metrics[k] - base) / base * 100, 2)
        return out

    @staticmethod
    def _improved(att: Attempt, best: Attempt | None) -> bool:
        """Lower whole-bill is better; nothing else counts as progress."""
        cur = att.metrics.get("bill_per_1k")
        if cur is None:
            return False
        if best is None:
            return True
        prev = best.metrics.get("bill_per_1k")
        return prev is None or cur < prev

    def _record(self, idea: Idea, att: Attempt, rationale: str, trace: str,
                history: list[Attempt]) -> str:
        verdict = ("win" if att.ok and att.delta.get("bill_per_1k_pct", 0) < 0
                   else "loss" if att.ok else "invalid")
        exp = Experiment(
            agent_id=self.agent_id, idea_id=idea.id, timeline=idea.timeline,
            hypothesis=idea.hypothesis, rationale=rationale,
            stack_digest=att.stack_digest, verdict=verdict,
            metrics=att.metrics, baseline_metrics=self.baseline,
            summary=self._summarise(att), tags=idea.targets, trace_ref=trace)
        self.memory.record(exp)
        if history:
            prev = history[-1]
            if prev.experiment_id:
                self.memory.relate(Relation(
                    src=prev.experiment_id, dst=exp.id, kind="derived_from",
                    note=f"attempt {att.n} of {idea.title}"))
        return exp.id

    @staticmethod
    def _summarise(att: Attempt) -> str:
        if not att.ok:
            return f"evaluation failed ({att.failure})"
        d = att.delta.get("bill_per_1k_pct")
        bill = att.metrics.get("bill_per_1k")
        if d is None:
            return f"ran; bill ${bill}/1k, no baseline to compare"
        return (f"bill ${bill}/1k, {d:+.1f}% vs baseline; "
                f"N*={att.metrics.get('n_star')}")
