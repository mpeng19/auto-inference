"""Modal calls a fleet left running when it stopped.

Every sweep and workbench call writes its `call_id` beside where its result
will land. A daemon killed with `--force`, or one that exits on a signal,
leaves those calls running to completion on a GPU nobody will read: build-3
was killed at 01:09 with sweeps in flight, and they billed until they
finished. This finds the ids with no result yet and cancels them.
"""
from __future__ import annotations

import pathlib


def pending_call_ids(root: str | pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """(call id, directory) for every call under `root` without a result."""
    out = []
    for p in sorted(pathlib.Path(root).glob("**/call_id")):
        if len(p.relative_to(root).parts) > 5:
            continue
        d = p.parent
        if (d / "result.json").is_file() or (d / "cancelled").is_file():
            continue
        cid = p.read_text().strip()
        if cid:
            out.append((cid, d))
    return out


def cancel_pending(root: str | pathlib.Path, cancel=None) -> list[str]:
    """Cancel them. `cancel(call_id)` defaults to Modal's; returns the ids
    that were cancelled. Errors on one id never stop the rest."""
    if cancel is None:
        import modal

        def cancel(cid: str) -> None:
            modal.FunctionCall.from_id(cid).cancel()
    done = []
    for cid, d in pending_call_ids(root):
        try:
            cancel(cid)
        except Exception:
            continue
        (d / "cancelled").write_text("cancelled by the fleet on stop\n")
        done.append(cid)
    return done
