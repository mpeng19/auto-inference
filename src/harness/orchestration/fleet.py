"""Reference OrchestrationService: N agent loops, live control, one snapshot.

Three jobs no single agent can do for itself.

**Hold the budget.** Whether the fleet can afford to keep going at all. The
*scheduling* of GPU time lives in `EvalService`, deliberately: an earlier
version handed out slots with a semaphore and seven of ten agents sat blocked
in it, which is capacity being rented and not used.

**Keep seeds diverse.** No agent can tell whether its idea duplicates another's;
only something holding all ten can.

**Be controllable while running.** A fleet runs detached for hours and an
operator watching it needs to pause one agent, kill another, and add two more
without restarting anything. Control is *cooperative* -- agents check at points
where stopping is cheap -- and it arrives through the session store rather than
through a method call, because the operator is in a different process.

Threads, not processes: the work is waiting on a network call, and one process
makes state inspection trivial. If an agent implementation ever executes
untrusted code locally that changes, but the modified SGLang only ever runs
inside a fresh Modal container, so today it does not.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from ..contracts.agent import AgentOutcome, AgentService, Idea
from ..contracts.orchestration import AgentState, FleetSpec, FleetState
from ..contracts.session import AgentView, Command, SessionView, TokenUse


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()
            if len(w) > 3}


class _Slot:
    """One agent's runtime state: its control flags and its published view."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.stop = threading.Event()
        self.resume = threading.Event()
        self.resume.set()                       # not paused
        self.view = AgentView(agent_id=agent_id)

    @property
    def paused(self) -> bool:
        return not self.resume.is_set()


class Fleet:
    """Reference implementation of `contracts.orchestration.OrchestrationService`."""

    def __init__(self, make_agent, evals, *, store=None, session_id: str = "",
                 root: str = "", similarity_threshold: float = 0.6,
                 tick_s: float = 1.0, max_reseeds: int = 4):
        self.make_agent = make_agent
        self._evals = evals
        self.store = store
        self.session_id = session_id or f"sess-{int(time.time())}"
        self.root = root
        self.similarity_threshold = similarity_threshold
        self.tick_s = tick_s
        self.max_reseeds = max_reseeds

        self._lock = threading.RLock()
        self._state = FleetState()
        self._spec: FleetSpec | None = None
        self._slots: dict[str, _Slot] = {}
        self._live_ideas: list[Idea] = []
        self._pool: ThreadPoolExecutor | None = None
        self._ctl: threading.Thread | None = None
        self._stop = threading.Event()
        self._next_index = 0
        self._target = 0
        self._cost = 0.0
        self._completed: list[AgentOutcome] = []

    @property
    def evals(self):
        """The evaluation queue. Agents submit here and keep working."""
        return self._evals

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self, spec: FleetSpec) -> str:
        with self._lock:
            if self._state.running:
                raise RuntimeError("fleet already running")
            self._spec = spec
            self._stop.clear()
            self._target = spec.fleet_budget.max_agents
            self._state = FleetState(running=True, started_at=time.time())
        # Generously sized: threads are cheap and mostly parked on a queue, and
        # `scale` must be able to add agents without a restart.
        self._pool = ThreadPoolExecutor(max_workers=max(64, self._target * 4),
                                        thread_name_prefix="agent")
        for _ in range(self._target):
            self._spawn()
        if self.store is not None:
            self.store.create(self._snapshot())
        self._ctl = threading.Thread(target=self._control_loop, name="fleet-ctl",
                                     daemon=True)
        self._ctl.start()
        return self.session_id

    def state(self) -> FleetState:
        with self._lock:
            return replace(
                self._state,
                agents=tuple(AgentState(agent_id=s.agent_id,
                                        status=s.view.status,
                                        attempts=s.view.attempts_total,
                                        cost_usd=s.view.cost_usd,
                                        note=s.view.note)
                             for s in self._slots.values()),
                cost_usd=self._cost, completed=tuple(self._completed))

    def stop(self, reason: str = "") -> FleetState:
        self._stop.set()
        with self._lock:
            for s in self._slots.values():
                s.stop.set()
                s.resume.set()                  # unblock anyone paused
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._ctl is not None:
            self._ctl.join(timeout=5)
        with self._lock:
            self._state = replace(self._state, running=False)
        if self.store is not None:
            self.store.publish(replace(self._snapshot(), phase="stopped",
                                       note=reason))
        return self.state()

    # ── control surface ──────────────────────────────────────────────────
    def pause_agent(self, agent_id: str) -> bool:
        with self._lock:
            s = self._slots.get(agent_id)
            if s is None or s.stop.is_set():
                return False
            s.resume.clear()
            s.view = replace(s.view, status="paused")
            return True

    def resume_agent(self, agent_id: str) -> bool:
        with self._lock:
            s = self._slots.get(agent_id)
            if s is None:
                return False
            s.resume.set()
            s.view = replace(s.view, status="thinking")
            return True

    def kill_agent(self, agent_id: str) -> bool:
        """Wind the agent up. Cooperative: a paid evaluation still finishes."""
        with self._lock:
            s = self._slots.get(agent_id)
            if s is None:
                return False
            s.stop.set()
            s.resume.set()
            s.view = replace(s.view, status="stopping")
            return True

    def scale(self, target: int) -> int:
        """Change how many agents run. Adds immediately, removes gracefully."""
        with self._lock:
            target = max(0, target)
            self._target = target
            live = [s for s in self._slots.values() if not s.stop.is_set()]
            if target > len(live):
                for _ in range(target - len(live)):
                    self._spawn()
            elif target < len(live):
                # Newest first: an agent several attempts into an idea has more
                # sunk cost and more chance of being about to pay off.
                for s in sorted(live, key=lambda x: x.agent_id, reverse=True)[
                        :len(live) - target]:
                    s.stop.set()
                    s.resume.set()
                    s.view = replace(s.view, status="stopping")
            return target

    def _spawn(self) -> str:
        agent_id = f"a{self._next_index:02d}"
        self._next_index += 1
        slot = _Slot(agent_id)
        self._slots[agent_id] = slot
        self._pool.submit(self._run_agent, agent_id)
        return agent_id

    # ── the AgentControl seam ────────────────────────────────────────────
    def should_stop(self, agent_id: str) -> bool:
        s = self._slots.get(agent_id)
        return self._stop.is_set() or (s is not None and s.stop.is_set())

    def wait_if_paused(self, agent_id: str, timeout_s: float = 3600) -> bool:
        s = self._slots.get(agent_id)
        if s is None:
            return False
        if not s.resume.wait(timeout=timeout_s):
            return False
        return not self.should_stop(agent_id)

    def report(self, agent_id: str, **fields) -> None:
        """Publish what an agent is doing. Never blocks, never raises."""
        with self._lock:
            s = self._slots.get(agent_id)
            if s is None:
                return
            tok = fields.pop("tokens", None)
            # An operator's state outranks the agent's own. Pause is
            # cooperative -- the agent keeps working until its next checkpoint
            # -- so without this the row flips straight back to "evaluating"
            # and `pause` looks like it did nothing. The activity still
            # updates, so the operator can see what it is finishing.
            if s.view.status in ("paused", "stopping") and \
                    fields.get("status") not in ("done", "failed"):
                fields.pop("status", None)
            s.view = replace(s.view, updated_at=time.time(),
                             **{k: v for k, v in fields.items()
                                if hasattr(s.view, k)})
            if tok is not None:
                s.view = replace(s.view, tokens=s.view.tokens + tok)
                if self.store is not None:
                    self.store.add_tokens(self.session_id, agent_id, tok)

    # ── budget and seeding ───────────────────────────────────────────────
    def _within_budget(self) -> bool:
        b = self._spec.fleet_budget
        if self._cost >= b.max_usd_total:
            return False
        started = self._state.started_at
        return not (started and time.time() - started > b.max_wall_s)

    def claim_idea(self, agent_id: str, proposed: Idea) -> Idea | None:
        with self._lock:
            for other in self._live_ideas:
                if self._too_similar(proposed, other):
                    return None
            self._live_ideas.append(proposed)
            return proposed

    def live_ideas(self) -> tuple[Idea, ...]:
        with self._lock:
            return tuple(self._live_ideas)

    def _too_similar(self, a: Idea, b: Idea) -> bool:
        ta, tb = _tokens(a.hypothesis + " " + a.title), _tokens(b.hypothesis + " " + b.title)
        if not ta or not tb:
            return False
        return len(ta & tb) / len(ta | tb) >= self.similarity_threshold

    # ── the published snapshot ───────────────────────────────────────────
    def _snapshot(self) -> SessionView:
        with self._lock:
            q = self._evals.stats() if self._evals is not None else None
            phase = ("stopping" if self._stop.is_set()
                     else "running" if self._state.running else "stopped")
            if self._slots and all(s.paused for s in self._slots.values()):
                phase = "paused"
            total = TokenUse()
            for s in self._slots.values():
                total = total + s.view.tokens
            return SessionView(
                session_id=self.session_id, phase=phase,
                started_at=self._state.started_at,
                agents=tuple(s.view for s in self._slots.values()),
                target_agents=self._target, cost_usd=round(self._cost, 4),
                tokens=total,
                evals_queued=getattr(q, "queued", 0),
                evals_running=getattr(q, "running", 0),
                evals_capacity=getattr(q, "capacity", 0),
                evals_completed=getattr(q, "completed", 0),
                evals_deduped=getattr(q, "deduped", 0),
                gpu_utilisation=getattr(q, "utilisation", 0.0),
                budget_usd=self._spec.fleet_budget.max_usd_total if self._spec else 0.0,
                pid=os.getpid(), root=self.root)

    def _control_loop(self) -> None:
        """Apply operator commands, publish the snapshot. Once per tick."""
        while not self._stop.is_set():
            if self.store is not None:
                # A watcher's database must never be able to kill a fleet
                # that is mid-experiment and holding rented GPUs.
                with contextlib.suppress(Exception):
                    for cmd in self.store.take_commands(self.session_id):
                        self.store.acknowledge(cmd.id, self._apply(cmd))
                    self.store.publish(self._snapshot())
            if not self._within_budget():
                break
            time.sleep(self.tick_s)

    def _apply(self, cmd: Command) -> str:
        k, a = cmd.kind, cmd.agent_id
        if k == "pause":
            return "paused" if self.pause_agent(a) else f"no such agent {a}"
        if k == "resume":
            return "resumed" if self.resume_agent(a) else f"no such agent {a}"
        if k == "kill":
            return "stopping" if self.kill_agent(a) else f"no such agent {a}"
        if k == "scale":
            try:
                return f"target {self.scale(int(cmd.value))}"
            except ValueError:
                return f"bad scale target {cmd.value!r}"
        if k == "stop":
            self._stop.set()
            with self._lock:
                for s in self._slots.values():
                    s.stop.set()
                    s.resume.set()
            return "stopping fleet"
        return "noted"

    # ── the agent thread ─────────────────────────────────────────────────
    def _record_outcome(self, out: AgentOutcome) -> None:
        with self._lock:
            self._completed.append(out)
            self._cost += out.cost_usd
            self._live_ideas = [i for i in self._live_ideas if i.id != out.idea.id]

    def _run_agent(self, agent_id: str) -> None:
        spec = self._spec
        seeds = list(spec.seeds)
        slot = self._slots[agent_id]
        collisions = 0
        while not self.should_stop(agent_id) and self._within_budget():
            if not self.wait_if_paused(agent_id):
                break
            self.report(agent_id, status="thinking", activity="choosing an idea")
            agent: AgentService = self.make_agent(agent_id, self)
            seed = seeds.pop(0) if seeds else None
            try:
                idea = agent.propose(seed=seed, live_ideas=self.live_ideas())
            except Exception as e:
                self.report(agent_id, status="failed", note=f"propose: {e}")
                break
            if self.claim_idea(agent_id, idea) is None:
                # Backs off, then gives up on being different. Diversity is a
                # heuristic, not a correctness property, and an agent that
                # re-seeds forever costs a model call per spin while producing
                # nothing. Better a near-duplicate that runs than a hot loop.
                collisions += 1
                if collisions >= self.max_reseeds:
                    with self._lock:
                        self._live_ideas.append(idea)
                    self.report(agent_id, note="accepted a near-duplicate idea "
                                               f"after {collisions} collisions")
                else:
                    self.report(agent_id,
                                activity=f"idea duplicated another agent's "
                                         f"({collisions}/{self.max_reseeds}); reseeding")
                    time.sleep(min(2.0, 0.1 * collisions))
                    continue
            collisions = 0
            self.report(agent_id, status="thinking", idea_title=idea.title,
                        idea_hypothesis=idea.hypothesis, attempt=0,
                        activity="starting")
            try:
                out = agent.run(idea, spec.agent_budget)
            except Exception as e:                  # one agent must not kill the fleet
                self.report(agent_id, status="failed", note=f"{type(e).__name__}: {e}")
                with self._lock:
                    self._live_ideas = [i for i in self._live_ideas if i.id != idea.id]
                continue
            self._record_outcome(out)
            self.report(agent_id, status="idle", cost_usd=out.cost_usd,
                        attempts_total=slot.view.attempts_total + len(out.attempts),
                        idle_s=out.idle_s,
                        best_delta_pct=(out.best.delta.get("bill_per_1k_pct")
                                        if out.best else None),
                        note=f"{out.stop} (idle {out.idle_s:.0f}s)")
        # Not unconditionally "done": an agent that broke out of the loop after
        # a failure has already reported why, and overwriting that hid a real
        # SQLite error behind a green row on the dashboard.
        with self._lock:
            slot = self._slots.get(agent_id)
            failed = slot is not None and slot.view.status == "failed"
        self.report(agent_id, activity="") if failed else \
            self.report(agent_id, status="done", activity="")
