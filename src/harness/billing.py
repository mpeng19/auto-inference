"""Modal's own bill for the current cycle: the number every harness dollar
figure is reconciled against.

`fetch()` asks the Modal client for the workspace billing summary (the same
call the dashboard makes): metered usage, credits applied, what is billed
so far and when the cycle ends. It needs the local Modal token and the
network, takes about a second, and returns None on any failure -- the
header shows a dash rather than the TUI waiting on Modal. `Cached` refreshes
it every five minutes from a thread so no refresh tick ever blocks on it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

TTL_S = 300.0


@dataclass(frozen=True)
class Bill:
    start: str                      # cycle start, ISO date
    end: str                        # cycle end = next payment date, ISO date
    metered_usd: float              # usage this cycle before credits
    billed_usd: float               # what will be charged at `end`
    credits_usd: float              # credits applied this cycle
    breakdown: dict = field(default_factory=dict)
    fetched_at: float = 0.0

    def line(self) -> str:
        """One header line: usage, credits, and the payment that is coming."""
        return (f"modal this cycle  ${self.metered_usd:,.2f} used"
                f"  ·  ${self.credits_usd:,.2f} credits"
                f"  ·  next payment ${self.billed_usd:,.2f} on {self.end}")


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def fetch(timeout_s: float = 15.0) -> Bill | None:
    """The live summary, or None when Modal cannot be reached.
    `HARNESS_BILLING=off` skips the call entirely (the test suite sets it:
    a Modal client thread started under blocked sockets complains at
    interpreter exit)."""
    import os

    if os.environ.get("HARNESS_BILLING", "").lower() == "off":
        return None
    out: list = []

    def go():
        try:
            from modal import Workspace

            s = Workspace.from_context().billing.summary()
            adj = dict(s.adjustments or {})
            out.append(Bill(
                start=str(s.start)[:10], end=str(s.end)[:10],
                metered_usd=round(_f(s.metered_cost), 2),
                billed_usd=round(_f(s.billed_cost), 2),
                credits_usd=round(abs(_f(adj.get("Credits", 0))), 2),
                breakdown={k: round(_f(v), 2) for k, v in
                           dict(s.metered_cost_breakdown or {}).items()},
                fetched_at=time.time()))
        except Exception:
            out.append(None)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout_s)
    return out[0] if out else None


class Cached:
    """The last bill seen, refreshed in the background every `ttl_s`."""

    def __init__(self, ttl_s: float = TTL_S, fetcher=fetch):
        self.ttl_s, self._fetch = ttl_s, fetcher
        self.bill: Bill | None = None
        self._at = 0.0
        self._lock = threading.Lock()
        self._busy = False

    def get(self) -> Bill | None:
        """Never blocks: returns what is known and refreshes if stale."""
        if time.time() - self._at >= self.ttl_s:
            with self._lock:
                if not self._busy:
                    self._busy = True
                    threading.Thread(target=self._refresh, daemon=True).start()
        return self.bill

    def _refresh(self) -> None:
        try:
            b = self._fetch()
            if b is not None:
                self.bill = b
        finally:
            self._at = time.time()
            self._busy = False
