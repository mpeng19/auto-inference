"""Git for experiments: a shadow repository per agent.

The run directory stays the record of a measurement (`stack.json` and the
rest, keyed by content digest); git is the *history*. An agent's working
copy holds only the files it materialised, so the repository is a shadow
tree at `<agent>/repo/`: the full package as the root commit (stock, or the
fleet's base), with the agent's overlay synced onto it at every checkpoint
the loop already has:

  eval <tier>: attempt N <digest> -> <run dir>     tag eval/<digest>
  abandoned: attempt N (stalled | cancelled)        the diff a reset would lose
  win/<digest>                                      a replicated win

`git diff <root>..HEAD` is exactly the stack; `git log` is the agent's
story; a fleet started `--base` from a run clones that agent's repository at
the commit the run was measured from, so a campaign reads as one branch of
stacked wins. Everything here is best effort: no git, no repository, and
the loop never notices.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import time
from dataclasses import dataclass

GIT_ENV = {"GIT_AUTHOR_NAME": "harness", "GIT_AUTHOR_EMAIL": "harness@local",
           "GIT_COMMITTER_NAME": "harness", "GIT_COMMITTER_EMAIL": "harness@local",
           "GIT_CONFIG_NOSYSTEM": "1", "HOME": ""}
SYNCED = "harness-synced.json"      # under .git: which rels the overlay last wrote


def git_available() -> bool:
    return shutil.which("git") is not None


@dataclass
class Repo:
    path: pathlib.Path

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def open_or_init(cls, agent_root, source, label: str = "stock",
                     base_run_dir: str | pathlib.Path | None = None) -> Repo | None:
        """The agent's repository, created on first use. With `base_run_dir`
        naming a run that carries a `commit` file and whose agent has a
        repository, that repository is cloned at that commit, so history
        continues across fleets; otherwise the root commit is the full
        source tree."""
        if not git_available():
            return None
        path = pathlib.Path(agent_root) / "repo"
        repo = cls(path)
        if (path / ".git").is_dir():
            return repo
        try:
            if base_run_dir and repo._clone_base(pathlib.Path(base_run_dir)):
                return repo
            path.mkdir(parents=True, exist_ok=True)
            _materialise_tree(source, path)
            (path / ".gitignore").write_text("__pycache__/\n*.pyc\n")
            repo._git("init", "-q")
            repo._git("add", "-A")
            repo._git("commit", "-q", "--allow-empty", "-m", f"base: {label}")
            return repo
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            return None

    def _clone_base(self, run_dir: pathlib.Path) -> bool:
        commit_file = run_dir / "commit"
        src = run_dir.parent.parent / "repo"
        if not commit_file.is_file() or not (src / ".git").is_dir():
            return False
        sha = commit_file.read_text().strip()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", str(src), str(self.path)],
                       check=True, capture_output=True, env=_env())
        self._git("reset", "-q", "--hard", sha)
        self._git("commit", "-q", "--allow-empty", "-m",
                  f"base: continued from {src.parent.name} at {sha[:12]} ({run_dir.name})")
        return True

    # ── checkpoints ─────────────────────────────────────────────────────
    def checkpoint(self, workspace, message: str, tag: str = "") -> str | None:
        """Sync the workspace's overlay onto the tree and commit. Returns the
        head sha, or None when git failed; never raises into the loop."""
        try:
            self.sync(workspace)
            return self.commit(message, tag)
        except Exception:
            return None

    def sync(self, workspace) -> None:
        """Write every touched file into the tree; put back the source's
        version of anything the previous sync wrote that is no longer
        touched (or delete it when the source never had it)."""
        touched = tuple(workspace.touched())
        marker = self.path / ".git" / SYNCED
        try:
            before = set(json.loads(marker.read_text()))
        except (OSError, ValueError):
            before = set()
        for rel in touched:
            dst = self.path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text((workspace.candidates / rel).read_text())
        for rel in before - set(touched):
            dst = self.path / rel
            try:
                dst.write_text(workspace.source.read(rel))
            except Exception:
                dst.unlink(missing_ok=True)
        marker.write_text(json.dumps(sorted(touched)))

    def commit(self, message: str, tag: str = "") -> str:
        self._git("add", "-A")
        changed = subprocess.run(["git", "-C", str(self.path), "diff", "--cached", "--quiet"],
                                 env=_env()).returncode != 0
        if changed:
            self._git("commit", "-q", "-m", message)
        if tag:
            self._git("tag", "-f", tag)
        return self.head()

    def tag(self, name: str) -> None:
        with_suppress(self._git, "tag", "-f", name)

    # ── reading ─────────────────────────────────────────────────────────
    def head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def root(self) -> str:
        return self._git("rev-list", "--max-parents=0", "HEAD").strip().splitlines()[-1]

    def log(self, n: int = 50) -> list[dict]:
        out = self._git("log", f"-{n}", "--format=%H%x1f%s%x1f%ct%x1f%D")
        rows = []
        for line in out.splitlines():
            sha, subject, ts, refs = [*line.split("\x1f"), "", "", "", ""][:4]
            rows.append({"sha": sha, "subject": subject, "at": int(ts or 0),
                         "tags": [r.strip()[4:].strip() for r in refs.split(",") if r.strip().startswith("tag:")]})
        return rows

    def diff_from_root(self) -> str:
        return self._git("diff", f"{self.root()}..HEAD")

    def _git(self, *args: str) -> str:
        r = subprocess.run(["git", "-C", str(self.path), *args], check=True,
                           capture_output=True, text=True, env=_env())
        return r.stdout


def with_suppress(fn, *args):
    import contextlib

    with contextlib.suppress(Exception):
        fn(*args)


def _env() -> dict:
    import os

    env = {**os.environ, **GIT_ENV}
    env["HOME"] = os.environ.get("HOME", "")     # keep git's own config lookups sane
    return env


def _materialise_tree(source, dest: pathlib.Path) -> None:
    """The full source tree: a stock wheel is copied, a base is the wheel
    with the base's files written over it, a test source is copied."""
    stock = getattr(source, "stock", None)
    base = getattr(source, "base", None)
    root = getattr(stock if stock is not None else source, "root", None)
    if root is not None and pathlib.Path(root).is_dir():
        shutil.copytree(root, dest, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    else:
        for rel in source.ls(""):
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(source.read(rel))
    if base is not None:
        for rel, text in getattr(base, "files", {}).items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)


def stamp(run_dir, sha: str | None) -> None:
    """`<run dir>/commit`: the commit a measurement was taken from."""
    if not sha:
        return
    try:
        d = pathlib.Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "commit").write_text(sha + "\n")
    except OSError:
        pass


def now() -> float:
    return time.time()
