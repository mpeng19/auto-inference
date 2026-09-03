"""The Modal bill line: formatted from a summary, and never a blocking call."""
import time

from harness import billing


def test_the_line_says_usage_credits_and_the_coming_payment():
    b = billing.Bill(start="2026-09-01", end="2026-10-01", metered_usd=404.71,
                     billed_usd=372.92, credits_usd=30.0)
    assert b.line() == ("modal this cycle  $404.71 used  ·  $30.00 credits"
                        "  ·  next payment $372.92 on 2026-10-01")


def test_fetch_is_none_when_modal_fails_and_never_raises(monkeypatch):
    import sys
    import types

    class Workspace:
        @staticmethod
        def from_context():
            raise RuntimeError("no token")

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Workspace=Workspace))
    t0 = time.time()
    assert billing.fetch(timeout_s=5.0) is None
    assert time.time() - t0 < 1.0


def test_cached_refreshes_in_the_background_and_keeps_the_last_value():
    calls = []

    def fake():
        calls.append(1)
        time.sleep(0.02)
        return billing.Bill("s", "e", 1.0, 1.0, 0.0) if len(calls) == 1 else None

    c = billing.Cached(ttl_s=0.0, fetcher=fake)
    c.get()                                # only starts the refresh; never blocks
    for _ in range(50):
        if c.bill is not None:
            break
        time.sleep(0.01)
    assert c.bill is not None and c.bill.metered_usd == 1.0
    for _ in range(50):
        if len(calls) >= 2:
            break
        c.get()
        time.sleep(0.01)
    assert c.get() is not None             # a failed refresh keeps the last bill
