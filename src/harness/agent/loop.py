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

import contextlib
import time
from dataclasses import dataclass, field, replace
from typing import Protocol

from ..contracts.agent import AgentBudget, AgentOutcome, Attempt, Idea, Stop
from ..contracts.context import TraceMeta, Turn
from ..contracts.evaluation import EvalRequest
from ..contracts.memory import Brief, Experiment, Recall, Relation
from .workspace import Workspace


class Proposer(Protocol):
    """Turns knowledge into an edit. The only place a model belongs."""

    def seed(self, live_ideas: tuple[Idea, ...], brief: Brief) -> Idea: ...

    def edit(self, ws: Workspace, idea: Idea, brief: Brief, attempt: int,
             history: tuple[Attempt, ...]) -> str:
        """Mutate the workspace. Returns the rationale for what it changed."""
        ...

    def study(self, ws: Workspace, idea: Idea, brief: Brief,
              history: tuple[Attempt, ...]) -> str:
        """Optional. Useful non-GPU work to do while an evaluation runs.

        This is where an agent reads other agents' traces, refines its
        hypothesis, or drafts the next candidate. It exists so that "waiting
        for a GPU" and "doing nothing" stop being the same state -- the whole
        point of the evaluation queue. Implementations may omit it.
        """
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
    evals: object             # contracts.evaluation.EvalService
    control: object | None = None     # contracts.agent.AgentControl
    baseline: dict = field(default_factory=dict)
    priority: int = 0

    # ── the control seam ─────────────────────────────────────────────────
    def _report(self, **fields) -> None:
        if self.control is not None:
            # Telemetry must never take an experiment down.
            with contextlib.suppress(Exception):
                self.control.report(self.agent_id, **fields)

    def _may_continue(self) -> bool:
        """Checked where stopping is cheap: between attempts, not mid-sweep.

        Cooperative on purpose. An operator killing an agent should not
        abandon an evaluation that has already been paid for.
        """
        if self.control is None:
            return True
        if not self.control.wait_if_paused(self.agent_id):
            return False
        return not self.control.should_stop(self.agent_id)

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
        """Iterate on one idea. Never blocks on a GPU with work still to do.

        The shape that matters: submit, then *study*, then collect. An
        evaluation is 25-60 minutes; an agent that spends those minutes in a
        semaphore is capacity being rented and not used.
        """
        started, spent, idle = time.time(), 0.0, 0.0
        infra_retries = 0
        attempts: list[Attempt] = []
        best: Attempt | None = None
        since_progress = 0
        stop: Stop = "exhausted"

        trace = self.context.open(TraceMeta(
            agent_id=self.agent_id, idea_id=idea.id, model="proposer"))
        self.context.append(trace, Turn(kind="prompt", content=idea.hypothesis))

        n = 0
        while n < budget.max_attempts:
            if spent >= budget.max_usd or time.time() - started > budget.max_wall_s:
                stop = "budget"
                break
            if not self._may_continue():
                stop = "budget"
                self._report(status="stopping", activity="stopped by operator")
                break

            self._report(status="thinking", attempt=n,
                         activity=f"attempt {n}: reading what the fleet knows")
            brief = self.memory.recall(Recall(
                intent=idea.hypothesis, agent_id=self.agent_id, idea_id=idea.id))
            self.context.append(trace, Turn(
                kind="thought", name="recall", content=brief.text,
                tokens_in=brief.est_tokens))

            self.workspace.reset()
            self._report(activity=f"attempt {n}: writing a diff")
            try:
                rationale = self.proposer.edit(
                    self.workspace, idea, brief, n, tuple(attempts))
            except Exception as e:
                self.context.append(trace, Turn(kind="error", content=str(e)))
                stop = "error"
                break

            ok, why = self.workspace.check()
            if not ok:
                # Free to reject here; ~6 GPU-minutes if it reaches the runner.
                # Keep what the model said: an agent that returns no diff
                # twice in a row is either stuck or being refused, and the
                # trace is the only place that distinction can be made.
                attempts.append(Attempt(idea_id=idea.id, agent_id=self.agent_id,
                                        n=n, ok=False, failure="invalid_diff"))
                self.context.append(trace, Turn(kind="thought", name="propose",
                                                content=str(rationale)[:4000]))
                self.context.append(trace, Turn(kind="error", name="check", content=why))
                n += 1
                continue

            stack = self.workspace.stack(label=f"{idea.title} #{n}")

            # Screen first, confirm what survives. Most candidates die cheaply.
            tier = "screen" if budget.screen_first else "full"
            att, waited = self._measure(stack, idea, trace, rationale, n, tier,
                                        brief, tuple(attempts))
            idle += waited
            spent += att.cost_usd

            if (tier == "screen" and att.ok
                    and att.delta.get("bill_per_1k_pct", 0.0) <= budget.screen_promise_pct):
                full, waited = self._measure(stack, idea, trace, rationale, n,
                                             "full", brief, tuple(attempts))
                idle += waited
                spent += full.cost_usd
                attempts.append(att)          # keep the screen in the record
                att = full

            # Infrastructure failures are retried unchanged; a rejected
            # hypothesis is not -- it costs the same and says the same.
            if not att.ok and att.failure == "infra":
                attempts.append(att)
                infra_retries += 1
                if infra_retries > budget.max_infra_retries:
                    stop = "error"
                    break
                continue

            exp = self._record(idea, att, rationale, trace, attempts)
            att = replace(att, experiment_id=exp)
            attempts.append(att)

            if self.divergence(idea, att) > budget.divergence_threshold:
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
                            attempts=tuple(attempts), best=best, cost_usd=spent,
                            idle_s=round(idle, 2))

    def _measure(self, stack, idea: Idea, trace: str, rationale: str, n: int,
                 tier: str, brief: Brief, history: tuple[Attempt, ...]
                 ) -> tuple[Attempt, float]:
        """Submit, work while it runs, collect. Returns (attempt, seconds idle).

        `idle` counts only the time left over after the agent ran out of useful
        things to do. Keeping it near zero is the entire justification for the
        evaluation queue existing.
        """
        req = EvalRequest(stack=stack, agent_id=self.agent_id, idea_id=idea.id,
                          attempt=n, tier=tier, priority=self.priority,
                          run_dir=str(self.workspace.run_dir(n)),
                          label=f"{idea.title} #{n} ({tier})")
        ticket = self.evals.submit(req)         # returns immediately, always
        self._report(status="queued", eval_ticket=ticket.id, attempt=n,
                     activity=f"attempt {n}: submitted a {tier} run")
        self.context.append(trace, Turn(
            kind="eval_submit", name=stack.digest, content=rationale,
            data={"tier": tier, "ticket": ticket.id, "queued": self.evals.stats().queued,
                  "diff": self.workspace.diff()[:20000]}))

        # Useful non-GPU work while the sweep runs.
        self._report(status="evaluating",
                     activity=f"attempt {n}: {tier} running; studying meanwhile")
        study = getattr(self.proposer, "study", None)
        if study is not None:
            try:
                note = study(self.workspace, idea, brief, history)
                if note:
                    self.context.append(trace, Turn(kind="thought", name="study",
                                                    content=str(note)))
            except Exception as e:              # studying must never kill a run
                self.context.append(trace, Turn(kind="error", name="study",
                                                content=str(e)))

        t0 = time.time()
        rec = self.evals.collect(ticket.id)
        idle = time.time() - t0
        self._report(status="thinking", queued_s=rec.queued_s, eval_ticket="",
                     cost_delta=rec.cost_usd,
                     activity=f"attempt {n}: {tier} done"
                              + (f" ({rec.failure})" if not rec.ok else ""))
        self.context.append(trace, Turn(kind="eval_result", name=stack.digest,
                                        data={"tier": tier, **rec.metrics}))
        att = Attempt(
            idea_id=idea.id, agent_id=self.agent_id, n=n, tier=tier,
            stack_digest=stack.digest, trace_ref=trace, ok=rec.ok,
            failure="" if rec.ok else (rec.failure or "hypothesis"),
            metrics=rec.metrics, delta=self._delta(rec.metrics, tier),
            cost_usd=rec.cost_usd, queued_s=rec.queued_s)
        return att, idle

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

    def _delta(self, metrics: dict, tier: str = "full") -> dict:
        """Percent change against stock *measured the same way*.

        A screen is a short sweep over a couple of levels, and its price
        carries warm-up that a full sweep amortises: on 2026-09-02 stock
        priced $14.96/1k at full tier and ~$17/1k at screen tier. Judged
        against the full baseline, no screen could ever clear the promotion
        threshold, so every idea died in the cheap tier and nothing was
        confirmed. `baseline["screen"]` holds stock at screen tier; without
        it the full baseline is used and the caller should know that is a
        comparison the screen cannot win.
        """
        base_map = self.baseline
        if tier == "screen" and isinstance(self.baseline.get("screen"), dict):
            base_map = self.baseline["screen"]
        out = {}
        for k, base in base_map.items():
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

    # Run-to-run noise of the measurement itself: two stock full sweeps on
    # 2026-09-02 differed by 0.8%, three stock-equivalent screens by 1.2%.
    # A verdict inside that band is not a result, and memory must not tell
    # the next agent it was.
    NOISE_PCT = 3.0

    @staticmethod
    def _verdict(att: Attempt) -> str:
        noise = IterativeAgent.NOISE_PCT
        if not att.ok:
            return "invalid" if att.failure == "invalid_diff" else "loss"
        d = att.delta.get("bill_per_1k_pct")
        if d is None:
            return "neutral"
        if d <= -noise:
            return "win"
        if d >= noise:
            return "loss"
        return "neutral"

    def _record(self, idea: Idea, att: Attempt, rationale: str, trace: str,
                history: list[Attempt]) -> str:
        verdict = self._verdict(att)
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
