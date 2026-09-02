"""The inference stack under test: stock SGLang plus a set of diffs.

The knobs SGLang exposes are a small slice of the serving design space. A
better scheduler, a different KV eviction policy, a smarter batch former are
not expressible as flags -- they are code. This module makes that code the unit
of experiment.

SGLang's serving layer is pure Python (`srt/managers/*`, `srt/mem_cache/*`);
only the kernels are compiled and they live in separate packages
(`sgl_kernel`, `sgl_deep_gemm`). So any module under `sglang/` can be replaced
by writing over it before the server starts -- no recompilation, no image
rebuild.

A stack carries its files **by value**, not by path. The whole set travels to
the runner as text inside the call, which means an evaluation is reproducible
from the record alone and needs no mounted directory, no image layer and no
shared filesystem. A scheduler is ~60 KB; a realistic stack is a handful of
those.

**Drift detection.** Every file records the SHA of the upstream it was derived
from. If SGLang is upgraded and that file changes underneath, the stack is
stale: it would silently revert upstream fixes while still looking like a valid
experiment. `apply()` refuses rather than producing a plausible wrong number.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

UPSTREAM_MANIFEST = "UPSTREAM.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def sglang_root() -> pathlib.Path:
    import sglang
    return pathlib.Path(sglang.__file__).parent


@dataclass(frozen=True)
class InferenceStack:
    """Stock SGLang, optionally with files replaced or patches applied.

        InferenceStack.stock()                     # the baseline
        InferenceStack.from_dir("my-diff/")        # a mirrored sglang/ tree
        InferenceStack.from_files({"srt/managers/schedule_policy.py": path})

    `files` maps a path *relative to the sglang package root* to the full text
    that should live there. `patches` maps the same kind of key to unified-diff
    text applied with `git apply` against the installed file.
    """
    files: dict[str, str] = field(default_factory=dict)
    patches: dict[str, str] = field(default_factory=dict)
    upstream_sha: dict[str, str] = field(default_factory=dict)
    label: str = ""

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def stock(cls) -> InferenceStack:
        return cls(label="stock")

    @classmethod
    def from_dir(cls, root: str | pathlib.Path) -> InferenceStack:
        """Read a directory that mirrors the sglang package under `sglang/`.

        `<root>/sglang/srt/managers/schedule_policy.py` replaces
        `site-packages/sglang/srt/managers/schedule_policy.py`. Any `.patch` or
        `.diff` beside it is applied to the same relative target instead.
        """
        d = pathlib.Path(root)
        src = d / "sglang"
        files, patches = {}, {}
        if src.is_dir():
            for f in sorted(src.rglob("*")):
                if not f.is_file():
                    continue
                rel = str(f.relative_to(src))
                if f.suffix == ".py":
                    files[rel] = f.read_text()
                elif f.suffix in (".patch", ".diff"):
                    patches[rel.rsplit(".", 1)[0]] = f.read_text()
        manifest = {}
        mf = d / UPSTREAM_MANIFEST
        if mf.exists():
            manifest = {k: v.get("upstream_sha")
                        for k, v in json.loads(mf.read_text()).items()}
        return cls(files=files, patches=patches,
                   upstream_sha={k: v for k, v in manifest.items() if v},
                   label=str(d))

    @classmethod
    def from_files(cls, mapping: dict[str, str | pathlib.Path],
                   label: str = "") -> InferenceStack:
        return cls(files={k: pathlib.Path(v).read_text() for k, v in mapping.items()},
                   label=label or f"{len(mapping)} file(s)")

    # ── identity ────────────────────────────────────────────────────────
    @property
    def is_stock(self) -> bool:
        return not self.files and not self.patches

    @property
    def digest(self) -> str:
        """Stable content hash. Two stacks with this digest are the same code.

        It is what makes an evaluation cacheable: a research loop that proposes
        the same diff twice should not spend 25 GPU-minutes twice.
        """
        body = {"files": {k: _sha(v.encode()) for k, v in sorted(self.files.items())},
                "patches": {k: _sha(v.encode()) for k, v in sorted(self.patches.items())}}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]

    def describe(self) -> str:
        if self.is_stock:
            return "stock SGLang"
        n = len(self.files) + len(self.patches)
        return f"{n} modified file(s) [{self.digest}]: " + ", ".join(
            sorted({*self.files, *self.patches}))

    def as_dict(self) -> dict:
        return {"files": self.files, "patches": self.patches,
                "upstream_sha": self.upstream_sha, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> InferenceStack:
        return cls(files=d.get("files", {}), patches=d.get("patches", {}),
                   upstream_sha=d.get("upstream_sha", {}), label=d.get("label", ""))

    # ── application, inside the container ───────────────────────────────
    def apply(self, allow_stale: bool = False, root: pathlib.Path | None = None) -> dict:
        """Write the stack over the installed sglang package.

        Returns provenance for the run record: a result that cannot be
        attributed to a specific version of the serving code is worthless to a
        search loop.

        **Starts from stock, every time.** The container that runs this is
        reused between calls, so whatever the previous stack wrote is still
        on disk. Until 2026-09-02 nothing put it back: each evaluation ran on
        top of the last one's diff, a stock run in a warm container was not
        stock, and a stack touching a file the previous one touched was
        refused as stale. `restored` in the provenance says what was undone.
        """
        if root is None:
            root = sglang_root()
        try:
            import sglang
            version = getattr(sglang, "__version__", "unknown")
        except ImportError:
            version = "unknown"
        prov: dict = {"sglang_version": version,
                      "digest": self.digest, "label": self.label,
                      "stock": self.is_stock, "applied": [], "stale": [],
                      "restored": restore_stock(root)}
        if self.is_stock:
            return prov

        for rel in sorted({*self.files, *self.patches}):
            dst = root / rel
            if not dst.exists():
                raise FileNotFoundError(
                    f"{rel} does not exist in sglang {prov['sglang_version']}; "
                    "the stack targets a file this version does not have")
            actual = _sha(dst.read_bytes())
            recorded = self.upstream_sha.get(rel)
            if recorded and recorded != actual:
                prov["stale"].append(rel)

        if prov["stale"] and not allow_stale:
            raise RuntimeError(
                f"stack is stale against sglang {prov['sglang_version']}: "
                f"{prov['stale']}. The upstream file changed since the diff was "
                "taken, so applying it would revert upstream changes while still "
                "looking like a valid experiment. Re-derive the diff, or pass "
                "allow_stale=True deliberately.")

        for rel, text in sorted(self.files.items()):
            dst = root / rel
            _backup(dst)
            dst.write_text(text)
            prov["applied"].append(rel)
        for rel, patch in sorted(self.patches.items()):
            dst = root / rel
            _backup(dst)
            _git_apply(root, rel, patch)
            prov["applied"].append(rel + " (patch)")
        return prov


def restore_stock(root: pathlib.Path) -> list[str]:
    """Put back every file `_backup` saved. Returns what was restored."""
    out = []
    for b in sorted(root.rglob("*.stock")):
        dst = b.with_name(b.name[: -len(".stock")])
        if not dst.exists() or dst.read_bytes() != b.read_bytes():
            shutil.copy2(b, dst)
            out.append(str(dst.relative_to(root)))
    return out


def _backup(dst: pathlib.Path) -> None:
    """Keep one pristine copy, so a run can always diff against stock."""
    b = dst.with_suffix(dst.suffix + ".stock")
    if dst.exists() and not b.exists():
        shutil.copy2(dst, b)


def _git_apply(root: pathlib.Path, rel: str, patch: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(patch if patch.endswith("\n") else patch + "\n")
        p = f.name
    r = subprocess.run(["git", "apply", "--unsafe-paths", f"--directory={root}",
                        "-p1", p], capture_output=True, text=True, cwd=str(root))
    if r.returncode != 0:
        raise RuntimeError(f"patch for {rel} did not apply: {r.stderr.strip()}")
