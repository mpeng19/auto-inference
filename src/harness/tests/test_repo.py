"""Git for experiments: the run directory records, the repository remembers."""
import json
import subprocess

import pytest

from harness import EvalBroker, IterativeAgent, Workspace
from harness.agent.repo import Repo, git_available
from harness.contracts import AgentBudget, Idea

from .test_workspace import FakeStock

P = "srt/managers/schedule_policy.py"
pytestmark = pytest.mark.skipif(not git_available(), reason="git is not installed")


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout


def test_the_root_commit_is_the_full_source_tree(tmp_path, stock_dir):
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=FakeStock(stock_dir))
    repo = Repo.open_or_init(ws.root, ws.source, label="stock")
    assert repo is not None and (repo.path / ".git").is_dir()
    assert repo.root() == repo.head()
    tracked = _git(repo.path, "ls-files").split()
    assert P in tracked and any(t.endswith(".py") for t in tracked)
    assert repo.log()[0]["subject"] == "base: stock"


def test_checkpoints_commit_exactly_the_overlay(tmp_path, stock_dir):
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=FakeStock(stock_dir))
    ws.materialise(P)
    repo = Repo.open_or_init(ws.root, ws.source)
    # materialised but identical: no commit
    assert repo.checkpoint(ws, "check: attempt 0") == repo.root()
    ws.edit(P, ws.read(P) + "\nCHUNK = 16384\n")
    sha = repo.checkpoint(ws, "eval screen: attempt 0 abc", tag="eval/abc")
    assert sha != repo.root()
    diff = repo.diff_from_root()
    assert "+CHUNK = 16384" in diff and diff.count("diff --git") == 1
    assert repo.log()[0]["tags"] == ["eval/abc"]
    # reset puts the source back; the next checkpoint reverts the file
    ws.reset()
    ws.materialise(P)
    repo.checkpoint(ws, "abandoned: attempt 1")
    assert "CHUNK = 16384" not in (repo.path / P).read_text()
    assert repo.diff_from_root() == ""
    # a tag moves to the newest commit that carries the digest
    ws.edit(P, ws.read(P) + "\nCHUNK = 4096\n")
    repo.checkpoint(ws, "eval full: attempt 2 abc", tag="eval/abc")
    assert _git(repo.path, "rev-parse", "eval/abc").strip() == repo.head()


def test_a_new_file_in_the_overlay_is_added_and_removed_with_it(tmp_path, stock_dir):
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=FakeStock(stock_dir))
    repo = Repo.open_or_init(ws.root, ws.source)
    new = "srt/managers/new_kernel.py"
    (ws.candidates / new).parent.mkdir(parents=True, exist_ok=True)
    (ws.candidates / new).write_text("def k(): pass\n")
    repo.checkpoint(ws, "eval: new file")
    assert new in _git(repo.path, "ls-files").split()
    ws.reset()
    repo.checkpoint(ws, "abandoned")
    assert new not in _git(repo.path, "ls-files").split()


def test_a_base_run_continues_the_previous_agents_history(tmp_path, stock_dir):
    """A fleet started --base from a run clones that agent's repository at
    the commit the run was measured from: the campaign is one branch."""
    a = Workspace(tmp_path / "s1" / "a02", agent_id="a02", source=FakeStock(stock_dir))
    a.materialise(P)
    a.edit(P, a.read(P) + "\nCHUNK = 16384\n")
    repo = Repo.open_or_init(a.root, a.source)
    sha = repo.checkpoint(a, "eval full: attempt 1 d1 -> runs/attempt-001", tag="eval/d1")
    run = a.root / "runs" / "attempt-001"
    run.mkdir(parents=True)
    (run / "commit").write_text(sha + "\n")

    b = Workspace(tmp_path / "s2" / "a00", agent_id="a00", source=FakeStock(stock_dir))
    b.base_path = str(run)
    repo2 = Repo.open_or_init(b.root, b.source, label="base d1", base_run_dir=run)
    subjects = [r["subject"] for r in repo2.log()]
    assert subjects[0].startswith("base: continued from a02 at " + sha[:12])
    assert "eval full: attempt 1 d1 -> runs/attempt-001" in subjects
    assert "CHUNK = 16384" in (repo2.path / P).read_text()


def test_the_loop_commits_at_submit_and_stamps_the_run_dir(tmp_path, stock_dir, memory, context):
    from .test_agent_calls import _runner
    from .test_stop_kill import BlockingProposer  # noqa: F401  (import check only)

    class Editing:
        def seed(self, live_ideas, brief):
            return Idea(title="chunk", hypothesis="tune chunk", targets=(P,))

        def edit(self, ws, idea, brief, attempt, history, cancel=None):
            ws.edit(P, ws.read(P) + "\nCHUNK = 16384\n")
            return "raised CHUNK"

        def study(self, ws, idea, brief, history, cancel=None):
            return "note"

    ws = Workspace(tmp_path / "a01", agent_id="a01", source=FakeStock(stock_dir))
    broker = EvalBroker(_runner(), capacity=1)
    agent = IterativeAgent(agent_id="a01", workspace=ws, memory=memory, context=context,
                           proposer=Editing(), evals=broker, baseline={"bill_per_1k": 12.23})
    try:
        out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                        AgentBudget(max_attempts=1, patience=1, screen_first=False,
                                    replicate_wins=False))
    finally:
        broker.shutdown()
    assert out.attempts
    repo = Repo(ws.root / "repo")
    subjects = [r["subject"] for r in repo.log()]
    assert any(s.startswith("eval full: attempt 0 ") for s in subjects), subjects
    run_dir = ws.run_dir(0)
    assert (run_dir / "commit").read_text().strip() == repo.head()
    tags = [t for r in repo.log() for t in r["tags"]]
    assert any(t.startswith("eval/") for t in tags)
    assert json.loads((repo.path / ".git" / "harness-synced.json").read_text()) == [P]
