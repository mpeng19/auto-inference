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

And two things a night of real runs added:

**A study ends when its result lands.** The point of studying during a sweep is
that waiting is free; a study that outlives its evaluation is the opposite,
and it holds the attempt open. It runs in a thread with a cancel flag now, and
the partial thought is kept.

**Every turn is timed.** A trace that records what happened but not how long
cannot tell a slow model from a sleeping laptop -- which is exactly the
question a five-hour gap on 2026-09-02 raised and nothing could answer.
"""
from __future__ import annotations

import contextlib
import inspect
import threading
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

        May accept a keyword-only `cancel: threading.Event`, which the loop
        sets when the result lands. Taking it is how a study stops being work
        the moment it stops being useful; a proposer that does not take one
        simply runs to completion and the loop waits for it.
        """
        ...


class Evaluator(Protocol):
    """Runs one measurement. Wraps the simulator; swappable for a fake."""

    def evaluate(self, stack, run_dir) -> tuple[bool, dict, str]:
        """-> (ok, metrics, failure_kind). `failure_kind` is '' when ok."""
        ...


# Trace phases, folded into the four an operator reasons about.
_PHASE_BUCKET = {"propose": "edit", "submit": "edit", "check": "edit", "edit": "edit",
                 "study": "study", "wait": "wait", "recall": "recall", "start": "other",
                 "done": "other", "paper": "paper"}


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

    def _append(self, trace: str, turn: Turn, *, since: float,
                phase: str = "", call: bool = False) -> None:
        """Append a turn, stamped with how long the phase it closes took.

        Every turn the loop writes carries `elapsed_s` and a `phase`, and a
        turn that closes a model call carries that call's own accounting too.
        The reason is a night that produced three attempts and no explanation:
        a closed lid had frozen the fleet for five hours, and nothing in the
        trace could tell that apart from a model thinking slowly. Wall seconds
        beside the model's own `duration_ms` answers it in one line -- a big
        gap between them is the host, not the model.

        Inline stamping was the alternative and it is how a phase quietly goes
        untimed: there are nine append sites and they are added to.
        """
        data = {"phase": phase or turn.name or turn.kind,
                "elapsed_s": round(time.time() - since, 3), **turn.data}
        # The fleet keeps a running total per phase for the dashboard.
        self._report(phase_delta=(_PHASE_BUCKET.get(data["phase"], "other"), data["elapsed_s"]))
        if call:
            stats = getattr(self.proposer, "last_call", None)
            with contextlib.suppress(Exception):
                # `phase` is dropped: the call's own label duplicates the one
                # above it, and losing the loop's would mislabel the turn.
                data.update({k: v for k, v in stats.as_dict().items()
                             if k != "phase"})
        self.context.append(trace, replace(turn, data=data))

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
        self._append(trace, Turn(kind="prompt", content=idea.hypothesis),
                     since=started, phase="start")

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
            t_phase = time.time()
            brief = self.memory.recall(Recall(
                intent=idea.hypothesis, agent_id=self.agent_id, idea_id=idea.id))
            self._append(trace, Turn(
                kind="thought", name="recall", content=brief.text,
                tokens_in=brief.est_tokens), since=t_phase, phase="recall")

            self.workspace.reset()
            # A build-mode edit can run for hours and tokens land only when
            # it returns; say when it started so a long call is not read as
            # a stall.
            self._report(activity=f"attempt {n}: writing a diff (since {time.strftime('%H:%M')})")
            t_phase = time.time()
            try:
                rationale = self.proposer.edit(
                    self.workspace, idea, brief, n, tuple(attempts))
            except Exception as e:
                self._append(trace, Turn(kind="error", name="propose",
                                         content=str(e)),
                             since=t_phase, phase="propose", call=True)
                stop = "error"
                break
            # Kept whether or not the diff survives the check: an agent that
            # returns no diff twice in a row is either stuck or being refused,
            # and the trace is the only place that distinction can be made.
            self._append(trace, Turn(kind="thought", name="propose",
                                     content=str(rationale)[:4000]),
                         since=t_phase, phase="propose", call=True)

            t_phase = time.time()
            ok, why = self.workspace.check()
            self._append(trace, Turn(kind="tool_call" if ok else "error",
                                     name="check", content=why or "ok",
                                     data={"ok": ok}),
                         since=t_phase, phase="check")
            if not ok:
                # Free to reject here; ~6 GPU-minutes if it reaches the runner.
                attempts.append(Attempt(idea_id=idea.id, agent_id=self.agent_id,
                                        n=n, ok=False, failure="invalid_diff"))
                n += 1
                continue

            stack = self.workspace.stack(label=f"{idea.title} #{n}")

            # Screen first, confirm what survives. Most candidates die cheaply.
            tier = "screen" if budget.screen_first else "full"
            att, waited = self._measure(stack, idea, trace, rationale, n, tier,
                                        brief, tuple(attempts), budget=budget)
            idle += waited
            spent += att.cost_usd

            if (tier == "screen" and att.ok
                    and att.delta.get("bill_per_1k_pct", 0.0) <= budget.screen_promise_pct):
                full, waited = self._measure(stack, idea, trace, rationale, n,
                                             "full", brief, tuple(attempts),
                                             budget=budget)
                idle += waited
                spent += full.cost_usd
                attempts.append(att)          # keep the screen in the record
                att = full

            # A win is measured twice. See AgentBudget.replicate_wins.
            if (att.tier == "full" and att.ok and budget.replicate_wins
                    and att.delta.get("bill_per_1k_pct", 0.0) <= -self.NOISE_PCT):
                again, waited = self._measure(stack, idea, trace, rationale, n,
                                              "full", brief, tuple(attempts),
                                              replicate=1, budget=budget)
                idle += waited
                spent += again.cost_usd
                attempts.append(att)
                att = self._worse(att, again)

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

        if any(a.ok and a.tier == "full" for a in attempts):
            # The write-up. Only after a full sweep: a paper about a screen
            # would be a paper about warm-up. Never allowed to fail the idea.
            since = time.time()
            try:
                write = getattr(self.proposer, "paper", None)
                if write is not None:
                    diff = ""
                    with contextlib.suppress(Exception):
                        diff = self.workspace.diff()
                    path = write(self.workspace, idea, tuple(attempts),
                                 self.baseline.get("bill_per_1k"), diff)
                    self._append(trace, Turn(kind="tool_call", name="paper", content=str(path)),
                                 since=since, phase="paper", call=True)
            except Exception as e:
                self._append(trace, Turn(kind="error", name="paper", content=str(e)),
                             since=since, phase="paper")
        self.context.close(trace, outcome=stop, cost_usd=spent)
        return AgentOutcome(agent_id=self.agent_id, idea=idea, stop=stop,
                            attempts=tuple(attempts), best=best, cost_usd=spent,
                            idle_s=round(idle, 2))

    def _measure(self, stack, idea: Idea, trace: str, rationale: str, n: int,
                 tier: str, brief: Brief, history: tuple[Attempt, ...],
                 replicate: int = 0,
                 budget: AgentBudget | None = None) -> tuple[Attempt, float]:
        """Submit, work while it runs, collect. Returns (attempt, seconds idle).

        `idle` counts only the time left over after the agent ran out of useful
        things to do. Keeping it near zero is the entire justification for the
        evaluation queue existing.
        """
        budget = budget or AgentBudget()
        rep = f"-rep{replicate}" if replicate else ""
        t_phase = time.time()
        req = EvalRequest(stack=stack, agent_id=self.agent_id, idea_id=idea.id,
                          attempt=n, tier=tier, priority=self.priority,
                          replicate=replicate,
                          run_dir=str(self.workspace.run_dir(n, rep)),
                          label=f"{idea.title} #{n} ({tier}{rep})")
        ticket = self.evals.submit(req)         # returns immediately, always
        self._report(status="queued", eval_ticket=ticket.id, attempt=n,
                     activity=f"attempt {n}: submitted a {tier} run")
        self._append(trace, Turn(
            kind="eval_submit", name=stack.digest, content=rationale,
            data={"tier": tier, "ticket": ticket.id,
                  "queued": self.evals.stats().queued,
                  "diff": self.workspace.diff()[:400_000]}),
            since=t_phase, phase="submit")

        # Useful non-GPU work while the sweep runs.
        self._report(status="evaluating",
                     activity=f"attempt {n}: {tier} running; studying meanwhile")
        t_phase = time.time()
        rec, idle = self._study_until_result(trace, ticket, idea, brief,
                                             history, budget)
        self._report(status="thinking", queued_s=rec.queued_s, eval_ticket="",
                     cost_delta=rec.cost_usd,
                     activity=f"attempt {n}: {tier} done"
                              + (f" ({rec.failure})" if not rec.ok else ""))
        if rec.ok and rec.metrics.get("bill_per_1k") is not None:
            m = rec.metrics
            self._report(last_bill_per_1k=m.get("bill_per_1k"),
                         last_rank=(f"{m['rank_bill']}/{m['rank_of']}"
                                    if m.get("rank_bill") and m.get("rank_of") else ""),
                         last_share_pct=((m.get("share_per_node") or 0.0) * 100.0
                                         if m.get("share_per_node") is not None else None))
        self._append(trace, Turn(kind="eval_result", name=stack.digest,
                                 data={"tier": tier, **rec.metrics}),
                     since=t_phase, phase="wait")
        att = Attempt(
            idea_id=idea.id, agent_id=self.agent_id, n=n, tier=tier,
            stack_digest=stack.digest, trace_ref=trace, ok=rec.ok,
            failure="" if rec.ok else (rec.failure or "hypothesis"),
            metrics=rec.metrics, delta=self._delta(rec.metrics, tier),
            cost_usd=rec.cost_usd, queued_s=rec.queued_s)
        return att, idle

    # How long the wait loop blocks on the broker before looking at the study
    # again. It waits on the broker's own condition variable, so this is not a
    # spin -- it is how often a finished study gets noticed.
    COLLECT_POLL_S = 1.0
    # A cancelled study is killed, not asked nicely; this is how long the loop
    # waits for the thread to notice before giving up on its note.
    STUDY_JOIN_S = 30.0
    _DONE = ("done", "failed", "cancelled")

    def _study_until_result(self, trace: str, ticket, idea: Idea, brief: Brief,
                            history: tuple[Attempt, ...],
                            budget: AgentBudget):
        """Study while the sweep runs, and stop the moment the result lands.

        The study used to run to completion and only then did the agent block
        on `collect`, so a study that outlived its evaluation left the agent
        answering a question the GPU had already answered -- and holding the
        attempt open while it did. Now the study runs in a thread with a
        cancel flag, this loop watches both, and whichever finishes first ends
        the other. A partial thought is still worth keeping, so the note is
        appended either way, flagged when it was cut off.

        The ticket is polled rather than collected outright because `collect`
        with a timeout reports its own timeout as a failed record -- fine to
        wait on, not something to believe. `poll` is the verdict; `collect` is
        just the wait.

        Returns (record, idle_s), where idle is the stretch after the study
        stopped with the result still not in. That is the number the queue
        exists to keep near zero, so it must not count time spent studying.
        """
        study = getattr(self.proposer, "study", None)
        box: dict = {}
        cancel = threading.Event()
        started = time.time()
        th = None
        if study is not None:
            def _work():
                try:
                    box["note"] = self._call_study(study, idea, brief, history,
                                                   cancel)
                except Exception as e:          # studying must never kill a run
                    box["error"] = str(e)
                finally:
                    box["ended"] = time.time()

            th = threading.Thread(target=_work, daemon=True,
                                  name=f"study-{self.agent_id}")
            th.start()

        while True:
            rec = self.evals.poll(ticket.id)
            if rec.status in self._DONE:
                break
            if th is not None and not th.is_alive():
                rec = self.evals.collect(ticket.id)   # nothing left but to wait
                break
            if (th is not None and not cancel.is_set()
                    and time.time() - started > budget.study_timeout_s):
                cancel.set()
            self.evals.collect(ticket.id, timeout_s=self.COLLECT_POLL_S)
        arrived = time.time()

        if th is None:
            return rec, arrived - started

        # Read before we set it: a flag already set here means the study ran
        # past `study_timeout_s`, which is a different fact about the agent
        # than a study the result overtook.
        capped = cancel.is_set()
        cut_short = th.is_alive()
        if cut_short:
            cancel.set()
            th.join(timeout=self.STUDY_JOIN_S)
        note = str(box.get("note") or "")
        why = "study budget spent" if capped else "result arrived"
        if cut_short:
            note = (note + f"\n\n(cut short: {why})").strip()
        if note:
            self._append(trace, Turn(kind="thought", name="study", content=note,
                                     data={"cut_short": cut_short,
                                           "cut_short_why": why if cut_short else ""}),
                         since=started, phase="study", call=True)
        elif box.get("error"):
            self._append(trace, Turn(kind="error", name="study",
                                     content=str(box["error"])),
                         since=started, phase="study", call=True)
        # `ended` missing means the thread is still running, i.e. the agent is
        # busy and owes nothing to idle.
        return rec, max(0.0, arrived - box.get("ended", arrived))

    def _call_study(self, study, idea: Idea, brief: Brief,
                    history: tuple[Attempt, ...], cancel: threading.Event) -> str:
        """Hand the cancel flag only to a proposer that takes one.

        Checked by signature rather than by catching TypeError: a TypeError
        raised *inside* a study would otherwise look like a signature mismatch
        and run the whole call a second time.
        """
        try:
            accepts = "cancel" in inspect.signature(study).parameters
        except (TypeError, ValueError):
            accepts = False
        if accepts:
            return study(self.workspace, idea, brief, history, cancel=cancel)
        return study(self.workspace, idea, brief, history)

    # ── guards and bookkeeping ───────────────────────────────────────────
    def divergence(self, idea: Idea, attempt: Attempt) -> float:
        """Fraction of touched files the idea never claimed it would touch.

        Crude on purpose. It is meant to catch "started on the radix cache,
        now rewriting the scheduler", not to judge research taste.

        `attempt` goes unread: the workspace still holds the diff that attempt
        measured, and reading it there means a diff is judged even when the
        evaluation that would have carried it never produced metrics.
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
    def _worse(a: Attempt, b: Attempt) -> Attempt:
        """Of two measurements of the same code, the one to believe. A failed
        replicate is the verdict; otherwise the higher bill."""
        if not b.ok:
            return b
        if not a.ok:
            return a
        return b if b.metrics.get("bill_per_1k", 0) > a.metrics.get("bill_per_1k", 0) else a

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
            # The tier travels with the metrics: a screen's price is judged
            # against stock at screen tier, and a reader of memory needs to
            # know which baseline a number was scored on.
            metrics={**att.metrics, "tier": att.tier}, baseline_metrics=self.baseline,
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
