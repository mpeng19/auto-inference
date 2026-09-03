"""Spend an agent causes outside the evaluation queue.

`harness tool gpu-run`, `ncu` and `equivalence` each hold an H100 for
minutes and run from the agent's own shell, so the fleet never saw them: on
build-4 the agents made 128 such calls, 14 GPU-hours, against a reported
fleet spend of $23 that covered the 14 evaluations only. The tools now
append what each call cost to `<agent>/spend.jsonl`, and the agent loop
drains that file after every model call into its cost and its budget.

The drain keeps a byte offset in `spend.seen`, so a daemon restarted over an
existing agent directory counts nothing twice.
"""
from __future__ import annotations

import json
import pathlib
import time

LEDGER = "spend.jsonl"
SEEN = "spend.seen"


def append(agent_root: str | pathlib.Path, tool: str, cost_usd: float, *,
           elapsed_s: float = 0.0, gpu: str = "", where: str = "") -> None:
    root = pathlib.Path(agent_root)
    root.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": round(time.time(), 3), "tool": tool,
                       "cost_usd": round(float(cost_usd or 0.0), 4),
                       "elapsed_s": round(float(elapsed_s or 0.0), 1),
                       "gpu": gpu, "where": where})
    with open(root / LEDGER, "a") as f:
        f.write(line + "\n")


def drain(agent_root: str | pathlib.Path) -> float:
    """Dollars appended since the last drain. Zero when there is no ledger."""
    root = pathlib.Path(agent_root)
    p = root / LEDGER
    if not p.is_file():
        return 0.0
    seen = root / SEEN
    try:
        offset = int(seen.read_text().strip() or 0)
    except (OSError, ValueError):
        offset = 0
    data = p.read_bytes()
    if offset > len(data):          # ledger was truncated; start over
        offset = 0
    usd = 0.0
    for raw in data[offset:].splitlines():
        try:
            usd += float(json.loads(raw).get("cost_usd") or 0.0)
        except (ValueError, AttributeError):
            continue
    seen.write_text(str(len(data)))
    return round(usd, 4)


def total(agent_root: str | pathlib.Path) -> float:
    """Everything in the ledger, drained or not. For reports."""
    p = pathlib.Path(agent_root) / LEDGER
    if not p.is_file():
        return 0.0
    usd = 0.0
    for raw in p.read_text().splitlines():
        try:
            usd += float(json.loads(raw).get("cost_usd") or 0.0)
        except (ValueError, AttributeError):
            continue
    return round(usd, 4)
