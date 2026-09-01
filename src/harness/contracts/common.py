"""Types every service shares.

The contracts package holds **no logic**. It exists so that each service can be
replaced wholesale -- a better memory store, a sandboxed agent runner, a
different orchestrator -- without any other service noticing. If a change here
is needed to make an implementation work, that is a signal the boundary is
wrong, not that the contract should grow.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ── identifiers ──────────────────────────────────────────────────────────
# Prefixed so a bare id in a log or a database row says what it is.

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def digest(obj: Any) -> str:
    """Stable content hash. The cache key for anything worth not repeating."""
    import json
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def now() -> float:
    return time.time()


@dataclass(frozen=True)
class Provenance:
    """Where a claim came from, and how to tell when it has gone stale.

    Every fact in this system is derived from a measurement that was made
    under specific conditions. A fact without provenance cannot be invalidated,
    and a store full of un-invalidatable facts poisons every agent that reads
    it -- `agent-db` measured that hazard directly (stale-fact noise) before
    fixing it with exactly this field.
    """
    harness_commit: str = ""
    stack_digest: str = ""          # which inference stack produced it
    eval_digest: str = ""           # which measurement configuration
    run_id: str = ""
    ts: float = field(default_factory=now)

    def is_stale(self, current_stack: str = "", current_eval: str = "") -> bool:
        if current_stack and self.stack_digest and current_stack != self.stack_digest:
            return True
        return bool(current_eval and self.eval_digest and current_eval != self.eval_digest)
