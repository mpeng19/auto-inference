"""Reference EvalService: a priority queue and a small worker pool.

The design goal is that **no agent is ever idle because a GPU is busy**.
Three mechanisms, in descending order of how much throughput they buy:

**1. Tiers.** A screening run (one concurrency level, short window) costs a
fraction of a full sweep and kills most candidates. Running every proposal at
full size is the single largest waste available, and no amount of clever
scheduling recovers it. `screen_fraction` reserves capacity for screens so a
queue full of expensive confirmations cannot starve them.

**2. Dedup by content.** A stack digest is a content hash, so two agents
proposing the same diff join one ticket. In a fleet seeded around a shared
baseline this is common, and each hit is a whole GPU-hour not spent. Only
*successes* are cached: a failure is a property of the moment rather than of
the code, and memoising one turns an infrastructure retry into an infinite
loop.

**3. Fair ordering.** FIFO within a priority band. A plain semaphore lets a
fast agent barge repeatedly; here the only way to jump the queue is to be
explicitly marked worth it.

What this deliberately does *not* do is make agents wait more politely. The
agent loop submits and keeps working -- preparing the next candidate, reading
other agents' traces, writing up what it learned. `est_wait_s` is published so
an agent can decide whether it is worth proposing at all right now.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import replace

from ..contracts.evaluation import (
    EvalRecord,
    EvalRequest,
    EvalService,
    EvalTicket,
    QueueStats,
)


class EvalBroker(EvalService):
    """Queue in front of the GPUs. `runner(request) -> (ok, metrics, failure)`."""

    def __init__(self, runner, *, capacity: int = 3, screen_fraction: float = 0.34,
                 timeout_s: float = 4 * 3600):
        self.runner = runner
        self.capacity = max(1, capacity)
        # At least one slot kept for screens unless capacity is 1: otherwise a
        # queue of full sweeps starves the cheap tier that does the filtering.
        self.screen_slots = max(1, round(self.capacity * screen_fraction)) \
            if self.capacity > 1 else 0
        self.timeout_s = timeout_s

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._heap: list = []                       # (-priority, seq, ticket_id)
        self._seq = itertools.count()
        self._tickets: dict[str, EvalTicket] = {}
        self._records: dict[str, EvalRecord] = {}
        self._by_key: dict[str, str] = {}           # dedup_key -> ticket id
        self._running: dict[str, float] = {}        # ticket id -> started_at
        self._durations: dict[str, list[float]] = {"screen": [], "full": []}
        self._deduped = self._completed = 0
        self._busy_time = 0.0
        self._started = time.time()
        self._stop = threading.Event()
        self._workers = [
            threading.Thread(target=self._worker, name=f"eval-{i}", daemon=True)
            for i in range(self.capacity)]
        for w in self._workers:
            w.start()

    # ── submit / collect ─────────────────────────────────────────────────
    def submit(self, req: EvalRequest) -> EvalTicket:
        with self._cv:
            existing = self._by_key.get(req.dedup_key)
            if existing is not None:
                # Same code, same tier: join the in-flight ticket rather than
                # rent a second GPU to learn the same thing.
                self._deduped += 1
                t = EvalTicket(request=req, deduped_from=existing)
                self._tickets[t.id] = t
                self._records[t.id] = EvalRecord(ticket_id=t.id, status="deduped")
                self._cv.notify_all()
                return t
            t = EvalTicket(request=req)
            self._tickets[t.id] = t
            self._by_key[req.dedup_key] = t.id
            self._records[t.id] = EvalRecord(ticket_id=t.id, status="queued")
            heapq.heappush(self._heap, (-req.priority, next(self._seq), t.id))
            self._cv.notify_all()
            return t

    def poll(self, ticket_id: str) -> EvalRecord:
        with self._lock:
            rec = self._records.get(
                ticket_id, EvalRecord(ticket_id=ticket_id, status="failed",
                                      note="unknown ticket"))
            src = self._tickets.get(ticket_id)
            if rec.status == "deduped" and src is not None and src.deduped_from:
                other = self._records.get(src.deduped_from)
                if other is not None and other.status in ("done", "failed"):
                    return replace(other, ticket_id=ticket_id, status=other.status)
            return rec

    def collect(self, ticket_id: str, timeout_s: float | None = None) -> EvalRecord:
        deadline = time.time() + (timeout_s if timeout_s is not None else self.timeout_s)
        with self._cv:
            while True:
                rec = self.poll(ticket_id)
                if rec.status in ("done", "failed", "cancelled"):
                    return rec
                if time.time() >= deadline:
                    return replace(rec, status="failed", failure="timeout",
                                   note="collect timed out")
                self._cv.wait(timeout=min(1.0, max(0.01, deadline - time.time())))

    def cancel(self, ticket_id: str) -> bool:
        """Queued tickets are droppable. A running one is already paid for."""
        with self._cv:
            rec = self._records.get(ticket_id)
            if rec is None or rec.status != "queued":
                return False
            self._records[ticket_id] = replace(rec, status="cancelled",
                                               failure="cancelled")
            self._heap = [x for x in self._heap if x[2] != ticket_id]
            heapq.heapify(self._heap)
            t = self._tickets.get(ticket_id)
            if t and t.request:
                self._by_key.pop(t.request.dedup_key, None)
            self._cv.notify_all()
            return True

    # ── introspection ────────────────────────────────────────────────────
    def stats(self) -> QueueStats:
        with self._lock:
            elapsed = max(1e-9, time.time() - self._started)
            live = sum(time.time() - s for s in self._running.values())
            return QueueStats(
                queued=len(self._heap), running=len(self._running),
                capacity=self.capacity, completed=self._completed,
                deduped=self._deduped, est_wait_s=self._est_wait(),
                utilisation=min(1.0, (self._busy_time + live)
                                / (elapsed * self.capacity)))

    def _est_wait(self) -> float:
        """Queue depth times mean duration over capacity. Rough on purpose."""
        durs = [d for v in self._durations.values() for d in v]
        mean = sum(durs) / len(durs) if durs else 0.0
        ahead = len(self._heap)
        return round(mean * (ahead + len(self._running)) / self.capacity, 1)

    def shutdown(self, wait: bool = True) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if wait:
            for w in self._workers:
                w.join(timeout=5)

    # ── the workers ──────────────────────────────────────────────────────
    def _take(self, worker_index: int) -> str | None:
        """Pop the next ticket this worker may run.

        Workers above `screen_slots` take anything; the reserved ones prefer a
        screen and only fall through to a full sweep when no screen is waiting.
        That keeps the cheap filtering tier alive under a backlog of expensive
        confirmations, which is when it matters most.
        """
        reserved = worker_index < self.screen_slots
        if not self._heap:
            return None
        if not reserved:
            return heapq.heappop(self._heap)[2]
        for i, (_, _, tid) in enumerate(self._heap):
            t = self._tickets.get(tid)
            if t and t.request and t.request.tier == "screen":
                self._heap.pop(i)
                heapq.heapify(self._heap)
                return tid
        return heapq.heappop(self._heap)[2]

    def _worker(self) -> None:
        idx = int(threading.current_thread().name.rsplit("-", 1)[-1])
        while not self._stop.is_set():
            with self._cv:
                tid = self._take(idx)
                while tid is None:
                    if self._stop.is_set():
                        return
                    self._cv.wait(timeout=0.05)
                    tid = self._take(idx)
                t = self._tickets[tid]
                started = time.time()
                self._running[tid] = started
                self._records[tid] = replace(self._records[tid], status="running",
                                             started_at=started)
                self._cv.notify_all()

            ok, metrics, failure = False, {}, "infra"
            try:
                ok, metrics, failure = self.runner(t.request)
            except Exception as e:                  # a bad runner must not kill a slot
                metrics, failure = {"error": f"{type(e).__name__}: {e}"}, "infra"

            with self._cv:
                ended = time.time()
                dur = ended - started
                self._running.pop(tid, None)
                self._busy_time += dur
                self._durations.setdefault(t.request.tier, []).append(dur)
                self._completed += 1
                self._records[tid] = EvalRecord(
                    ticket_id=tid, status="done" if ok else "failed", ok=ok,
                    metrics=metrics, failure="" if ok else failure,
                    started_at=started, ended_at=ended,
                    queued_s=round(started - t.submitted_at, 3),
                    cost_usd=float(metrics.get("cost_usd", 0.0)))
                # A success is cached: an identical proposal later reads the
                # answer instead of renting a GPU. A **failure is not**, and
                # that distinction is load-bearing. An infrastructure failure
                # is a property of the moment, not of the code, so an agent
                # retrying the same diff must genuinely re-run -- caching it
                # made the retry join the cached failure and loop forever.
                if not ok:
                    self._by_key.pop(t.request.dedup_key, None)
                self._cv.notify_all()
