"""Where a stock SGLang source file comes from.

An agent's whole job is "here is the current file, propose a better one", so
something has to serve the current file. Two ways, in preference order:

1. **The installed package**, if SGLang is importable here. Exact, free.
2. **The pinned wheel**, fetched once and cached. SGLang's serving layer is
   pure Python -- the kernels live in separate packages (`sgl_kernel`,
   `sgl_deep_gemm`) -- so the wheel carries every file an agent can usefully
   edit, and it does not require a GPU box or a 3 GB install to read one.

The wheel is fetched **by URL, not through pip**, and that is not a shortcut:
SGLang 0.5.18 publishes only `manylinux` wheels and no sdist, so `pip download`
on a Mac correctly refuses every file it finds and an agent could not read a
line of the code it is meant to improve. We want `.py` text, not an install,
and the Python sources are identical across platform wheels -- so resolving a
platform is the wrong question to ask.


The cache is **shared across the fleet**. Ten agents each downloading the same
wheel is ten times the wait and ten copies on disk, and they would still all be
reading identical bytes.

The version is pinned to whatever the runner will apply the diff against.
Reading 0.5.18 and applying to 0.5.19 is the drift `InferenceStack` refuses at
apply time; catching it here instead means catching it before a GPU is rented.

**A base is a third source.** A compounding fleet starts every agent from the
best stack of the last one, not from stock. `BaseSource` layers that stack's
files over one of the two above: an agent asking for "the current file" gets
the base's version where the base has one, so its edits -- and its diff, and
"did anything change" -- are relative to the base. The upstream hash for drift
detection still comes from the real stock underneath (`stock_sha`).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from simulator.stack import InferenceStack

SGLANG_VERSION = "0.5.18"
CACHE_ROOT = pathlib.Path.home() / ".cache" / "auto-inference" / "sglang"


@runtime_checkable
class StockSource(Protocol):
    version: str

    def read(self, rel: str) -> str:
        """`srt/managers/schedule_policy.py` -> its stock text."""
        ...

    def ls(self, prefix: str = "srt") -> tuple[str, ...]: ...

    def sha(self, rel: str) -> str: ...


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class InstalledSglang:
    """Read from the SGLang installed in this interpreter."""
    version: str = ""

    def __post_init__(self):
        import sglang
        self.root = pathlib.Path(sglang.__file__).parent
        self.version = self.version or getattr(sglang, "__version__", "unknown")

    @staticmethod
    def available() -> bool:
        try:
            import sglang  # noqa: F401
            return True
        except Exception:
            return False

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text()

    def ls(self, prefix: str = "srt") -> tuple[str, ...]:
        base = self.root / prefix
        return tuple(sorted(str(p.relative_to(self.root))
                            for p in base.rglob("*.py"))) if base.is_dir() else ()

    def sha(self, rel: str) -> str:
        return _sha(self.read(rel))


@dataclass
class WheelSource:
    """Read from a pinned wheel, extracted once into a shared cache."""
    version: str = SGLANG_VERSION
    cache_root: pathlib.Path = field(default=CACHE_ROOT)

    @property
    def root(self) -> pathlib.Path:
        return self.cache_root / self.version / "sglang"

    index_url: str = "https://pypi.org/pypi"

    def _wheel_url(self) -> str:
        """Any wheel for this version: the .py sources are platform-identical."""
        import sys
        meta = json.load(urllib.request.urlopen(
            f"{self.index_url}/sglang/{self.version}/json", timeout=60))
        files = [f for f in meta["urls"] if f["packagetype"] == "bdist_wheel"]
        files = files or [f for f in meta["urls"] if f["packagetype"] == "sdist"]
        if not files:
            raise RuntimeError(f"sglang {self.version} publishes no usable file")
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        return min(files, key=lambda f: (tag not in f["filename"], f["size"]))["url"]

    def ensure(self) -> pathlib.Path:
        """Fetch and extract if we do not already have it. Idempotent.

        Extracts into a staging directory and renames, so two agents starting
        at the same instant cannot leave a half-written tree that later reads
        would treat as complete.
        """
        if self.root.is_dir():
            return self.root
        d = self.cache_root / self.version
        d.mkdir(parents=True, exist_ok=True)
        staging = d.with_name(f"{d.name}.partial.{os.getpid()}")
        staging.mkdir(parents=True, exist_ok=True)
        blob = urllib.request.urlopen(self._wheel_url(), timeout=600).read()
        n = 0
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                if name.startswith("sglang/") and name.endswith(".py"):
                    out = staging / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
                    n += 1
        if not n:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"sglang {self.version} wheel contained no sources")
        # Rename the extracted package into place, not the staging root: the
        # version directory may already exist (a previous partial attempt, or
        # another agent), and renaming onto an existing directory raises --
        # which an over-broad `except OSError` once turned into silently
        # discarding a good extraction and reporting success.
        try:
            (staging / "sglang").rename(self.root)
        except OSError:
            if not self.root.is_dir():
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if not self.root.is_dir():
            raise RuntimeError(f"extraction produced nothing at {self.root}")
        return self.root

    def read(self, rel: str) -> str:
        return (self.ensure() / rel).read_text()

    def ls(self, prefix: str = "srt") -> tuple[str, ...]:
        base = self.ensure() / prefix
        return tuple(sorted(str(p.relative_to(self.root))
                            for p in base.rglob("*.py"))) if base.is_dir() else ()

    def sha(self, rel: str) -> str:
        return _sha(self.read(rel))


def stock(version: str = SGLANG_VERSION) -> StockSource:
    """The fleet's stock source. Installed package if present, else the wheel."""
    if InstalledSglang.available():
        s = InstalledSglang()
        if s.version == version or version == "":
            return s
    return WheelSource(version=version)


@dataclass
class BaseSource:
    """A saved stack's files layered over a stock source.

    `read(rel)` is the base's text when the base carries that file, else
    stock's. A base's *patches* are not rendered here (that would need `git
    apply` on the agent's machine); they travel with the composed stack and
    are applied in the container, so an agent editing a patched file edits
    its pre-patch text. Bases are saved from `Workspace.stack()`, which
    carries whole files, so in practice the case does not arise.
    """
    base: InferenceStack
    stock: StockSource
    version: str = ""

    def __post_init__(self):
        self.version = self.version or self.stock.version

    def read(self, rel: str) -> str:
        text = self.base.files.get(rel)
        return text if text is not None else self.stock.read(rel)

    def ls(self, prefix: str = "srt") -> tuple[str, ...]:
        have = set(self.stock.ls(prefix))
        have.update(r for r in self.base.files if r.startswith(prefix.rstrip("/") + "/"))
        return tuple(sorted(have))

    def sha(self, rel: str) -> str:
        return _sha(self.read(rel))

    def stock_sha(self, rel: str) -> str:
        """The hash of the *upstream* file, for drift detection: the base's
        own record of it if it has one, else the stock text underneath.
        Raises like `read` when the file is not in stock at all."""
        rec = self.base.upstream_sha.get(rel)
        return rec or self.stock.sha(rel)
