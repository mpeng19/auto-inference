"""Pull a stock SGLang module into overlays/ so it can be modified.

    PYTHONPATH=src uv run modal run scripts/vendor.py --path srt/managers/schedule_policy.py
    PYTHONPATH=src uv run modal run scripts/vendor.py::ls --pattern managers

Reads the file out of the built image (so it is exactly the code that will run,
not whatever GitHub's main branch says today), writes it to
`overlays/sglang/<path>`, and records its upstream SHA in
`overlays/UPSTREAM.json`. That SHA is what lets `overlay.apply()` detect later
that SGLang has moved underneath the edit.
"""
from __future__ import annotations

import json
import pathlib

import modal

from autoinf.modal_app import image

app = modal.App("auto-inference-vendor", image=image)
OVERLAYS = pathlib.Path(__file__).resolve().parents[1] / "overlays"


@app.function(cpu=1.0, timeout=300)
def _read(path: str) -> dict:
    import hashlib
    import sglang
    root = pathlib.Path(sglang.__file__).parent
    f = root / path
    if not f.exists():
        cands = [str(p.relative_to(root)) for p in root.rglob(pathlib.Path(path).name)]
        return {"error": f"{path} not found", "candidates": cands[:20]}
    b = f.read_bytes()
    return {"path": path, "text": b.decode("utf-8", "replace"),
            "sha": hashlib.sha256(b).hexdigest()[:16],
            "lines": len(b.splitlines()),
            "sglang_version": getattr(sglang, "__version__", "unknown")}


@app.function(cpu=1.0, timeout=300)
def _ls(pattern: str = "") -> list[str]:
    import sglang
    root = pathlib.Path(sglang.__file__).parent
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.py")
                  if pattern in str(p.relative_to(root)))[:200]


@app.local_entrypoint()
def main(path: str):
    r = _read.remote(path)
    if "error" in r:
        print(r["error"])
        for c in r.get("candidates", []):
            print("  candidate:", c)
        return

    dst = OVERLAYS / "sglang" / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(r["text"])

    mf = OVERLAYS / "UPSTREAM.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    manifest[path] = {"upstream_sha": r["sha"], "sglang_version": r["sglang_version"]}
    mf.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"vendored {path}")
    print(f"  -> {dst}  ({r['lines']} lines, upstream sha {r['sha']})")
    print(f"  edit it, then any bench run picks it up with no image rebuild.")


@app.local_entrypoint()
def ls(pattern: str = ""):
    for p in _ls.remote(pattern):
        print(p)
