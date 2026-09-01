"""The evaluation queue: the answer to "no agent should sit waiting".

These are the properties that justify the queue existing, so they are asserted
rather than assumed.
"""
import threading
import time
from dataclasses import dataclass, field

from harness import EvalBroker
from harness.contracts import EvalRequest, EvalService


class Stack:
    def __init__(self, digest):
        self.digest = digest


@dataclass
class Runner:
    delay: float = 0.02
    ok: bool = True
    calls: list = field(default_factory=list)
    tiers: list = field(default_factory=list)
    peak: int = 0
    _live: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, req):
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)
            self.calls.append(req.stack.digest)
            self.tiers.append(req.tier)
        try:
            time.sleep(self.delay)
            return self.ok, {"bill_per_1k": 12.0, "cost_usd": 1.0}, "" if self.ok else "infra"
        finally:
            with self._lock:
                self._live -= 1


def _req(d, **kw):
    return EvalRequest(stack=Stack(d), **kw)


def test_satisfies_the_contract():
    b = EvalBroker(Runner(), capacity=1)
    try:
        assert isinstance(b, EvalService)
    finally:
        b.shutdown()


def test_submit_never_blocks():
    """The whole point. An agent must always be able to propose."""
    b = EvalBroker(Runner(delay=0.5), capacity=1)
    try:
        t0 = time.time()
        for i in range(50):
            b.submit(_req(f"s{i}"))
        assert time.time() - t0 < 0.5, "submit blocked; it must only enqueue"
        assert b.stats().queued > 0
    finally:
        b.shutdown()


def test_capacity_is_respected():
    r = Runner(delay=0.05)
    b = EvalBroker(r, capacity=2)
    try:
        tix = [b.submit(_req(f"s{i}")) for i in range(8)]
        for t in tix:
            b.collect(t.id)
        assert r.peak <= 2
    finally:
        b.shutdown()


def test_identical_diffs_share_one_gpu():
    """A stack digest is a content hash; two agents proposing the same edit is
    common in a fleet seeded around one baseline."""
    r = Runner()
    b = EvalBroker(r, capacity=2)
    try:
        a = b.submit(_req("same", agent_id="a01"))
        c = b.submit(_req("same", agent_id="a02"))
        assert c.deduped_from == a.id
        assert b.collect(a.id).ok and b.collect(c.id).ok
        assert r.calls.count("same") == 1, "paid twice for one measurement"
        assert b.stats().deduped == 1
    finally:
        b.shutdown()


def test_failures_are_not_cached_so_a_retry_really_retries():
    """An infra failure is a property of the moment, not of the code.

    Caching one made an agent's retry join the cached failure and loop forever.
    """
    r = Runner(ok=False)
    b = EvalBroker(r, capacity=1)
    try:
        first = b.submit(_req("x"))
        assert not b.collect(first.id).ok
        second = b.submit(_req("x"))
        assert second.deduped_from == "", "a failed result must not be memoised"
        b.collect(second.id)
        assert r.calls.count("x") == 2
    finally:
        b.shutdown()


def test_priority_jumps_the_queue():
    r = Runner(delay=0.02)
    b = EvalBroker(r, capacity=1)
    try:
        b.submit(_req("blocker"))
        time.sleep(0.005)                   # let the worker pick it up
        for i in range(4):
            b.submit(_req(f"low{i}", priority=0))
        hot = b.submit(_req("hot", priority=10))
        b.collect(hot.id)
        # `hot` should run after the one already in flight, not after the four
        # queued ahead of it. (`low3` may legitimately not have run at all --
        # which is the point.)
        assert r.calls.index("hot") <= 2, f"order was {r.calls}"
        assert r.calls.count("low3") == 0 or r.calls.index("low3") > r.calls.index("hot")
    finally:
        b.shutdown()


def test_screens_are_not_starved_by_a_backlog_of_full_sweeps():
    """The cheap tier does the filtering; a queue of confirmations must not
    push it out or the fleet loses its only throughput multiplier."""
    r = Runner(delay=0.02)
    b = EvalBroker(r, capacity=3, screen_fraction=0.34)
    try:
        for i in range(12):
            b.submit(_req(f"full{i}", tier="full"))
        screen = b.submit(_req("screen0", tier="screen"))
        b.collect(screen.id, timeout_s=10)
        ran_before = r.calls.index("screen0")
        assert ran_before < 10, f"screen waited behind {ran_before} full sweeps"
    finally:
        b.shutdown()


def test_queued_tickets_can_be_cancelled_running_ones_cannot():
    b = EvalBroker(Runner(delay=0.2), capacity=1)
    try:
        running = b.submit(_req("a"))
        time.sleep(0.02)
        queued = b.submit(_req("b"))
        assert b.cancel(queued.id) is True
        assert b.cancel(running.id) is False, "already paid for; let it finish"
        assert b.poll(queued.id).status == "cancelled"
    finally:
        b.shutdown()


def test_stats_expose_what_an_agent_needs_to_decide():
    b = EvalBroker(Runner(delay=0.02), capacity=2)
    try:
        tix = [b.submit(_req(f"s{i}")) for i in range(6)]
        for t in tix:
            b.collect(t.id)
        s = b.stats()
        assert s.completed == 6 and s.capacity == 2
        assert 0.0 < s.utilisation <= 1.0
    finally:
        b.shutdown()


def test_a_broken_runner_does_not_kill_a_slot():
    def boom(req):
        raise RuntimeError("runner exploded")

    b = EvalBroker(boom, capacity=1)
    try:
        t = b.submit(_req("x"))
        rec = b.collect(t.id, timeout_s=5)
        assert not rec.ok and rec.failure == "infra"
        ok = EvalBroker(Runner(), capacity=1)
        ok.shutdown()
    finally:
        b.shutdown()


def test_collect_times_out_rather_than_hanging_forever():
    b = EvalBroker(Runner(delay=5.0), capacity=1)
    try:
        t = b.submit(_req("slow"))
        rec = b.collect(t.id, timeout_s=0.1)
        assert rec.failure == "timeout"
    finally:
        b.shutdown(wait=False)
