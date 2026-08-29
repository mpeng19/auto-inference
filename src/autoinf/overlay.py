"""Source overlays: modify SGLang's serving logic, not just its CLI flags.

The knobs SGLang exposes are a small slice of the serving-architecture design
space. A better scheduler, a different KV eviction policy, a smarter batch
former or a new dataloader are not expressible as flags — they are code. This
module makes that code the unit of experiment.

How it works. SGLang's serving layer is pure Python (`srt/managers/*`,
`srt/mem_cache/*`); only the kernels are compiled, and they live in separate
packages (`sgl_kernel`, `sgl_deep_gemm`, ...). So any module under `sglang/`
can be replaced by copying a file over it before the server starts. No
recompilation, and — because Modal mounts the overlay directory at runtime
rather than baking it into the image — no image rebuild between experiments.

    overlays/sglang/srt/managers/schedule_policy.py
        replaces
    site-packages/sglang/srt/managers/schedule_policy.py

**Drift detection.** Each overlay records the SHA of the upstream file it was
derived from. If SGLang is upgraded and that file changes underneath us, the
overlay is stale — it silently reverts upstream fixes and may not even reflect
the code we think we modified. `apply()` refuses to run in that case rather
than producing a plausible-looking but meaningless result.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil

UPSTREAM_MANIFEST = "UPSTREAM.json"


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def sglang_root() -> pathlib.Path:
    import sglang
    return pathlib.Path(sglang.__file__).parent


def _pairs(overlay_dir: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """(overlay file, its destination inside the installed sglang package)."""
    src_root = overlay_dir / "sglang"
    if not src_root.is_dir():
        return []
    dst_root = sglang_root()
    out = []
    for f in sorted(src_root.rglob("*.py")):
        out.append((f, dst_root / f.relative_to(src_root)))
    return out


def status(overlay_dir: str | os.PathLike) -> dict:
    """Report what would be overlaid and whether any overlay has gone stale."""
    d = pathlib.Path(overlay_dir)
    manifest = {}
    mf = d / UPSTREAM_MANIFEST
    if mf.exists():
        manifest = json.loads(mf.read_text())

    files, stale = {}, []
    for src, dst in _pairs(d):
        rel = str(dst.relative_to(sglang_root()))
        recorded = manifest.get(rel, {}).get("upstream_sha")
        actual = _sha(dst) if dst.exists() else None
        entry = {
            "overlay_sha": _sha(src),
            "upstream_sha_recorded": recorded,
            "upstream_sha_actual": actual,
            "target_exists": dst.exists(),
        }
        # Stale if we recorded an upstream SHA and the installed file no longer
        # matches it -- SGLang changed under the overlay.
        if recorded and actual and recorded != actual:
            entry["stale"] = True
            stale.append(rel)
        files[rel] = entry

    import sglang
    return {
        "sglang_version": getattr(sglang, "__version__", "unknown"),
        "n_overlays": len(files),
        "files": files,
        "stale": stale,
        "digest": hashlib.sha256(
            json.dumps({k: v["overlay_sha"] for k, v in files.items()},
                       sort_keys=True).encode()).hexdigest()[:12],
    }


def apply(overlay_dir: str | os.PathLike, allow_stale: bool = False) -> dict:
    """Copy overlays over the installed sglang package. Call before launching.

    Returns the provenance dict that belongs in the run record: without it, a
    result cannot be attributed to a specific version of the serving code.
    """
    d = pathlib.Path(overlay_dir)
    st = status(d)

    if st["stale"] and not allow_stale:
        raise RuntimeError(
            f"overlay(s) stale against installed sglang {st['sglang_version']}: "
            f"{st['stale']}. The upstream file changed since the overlay was "
            f"taken, so the overlay would revert upstream changes. Re-vendor "
            f"and re-apply your edits, or pass allow_stale=True deliberately."
        )

    applied = []
    for src, dst in _pairs(d):
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
        # Keep one pristine copy so a run can diff against stock.
        backup = dst.with_suffix(dst.suffix + ".stock")
        if dst.exists() and not backup.exists():
            shutil.copy2(dst, backup)
        shutil.copy2(src, dst)
        applied.append(str(dst.relative_to(sglang_root())))

    st["applied"] = applied
    return st
