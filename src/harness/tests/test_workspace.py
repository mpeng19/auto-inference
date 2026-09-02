"""The agent's diff API: the only interface between an agent and the code."""
import pathlib
from dataclasses import dataclass

import pytest

from harness.agent.workspace import Workspace

P = "srt/managers/schedule_policy.py"


@dataclass
class FakeStock:
    root: pathlib.Path
    version: str = "test"

    def read(self, rel):
        return (self.root / rel).read_text()

    def ls(self, prefix="srt"):
        return tuple(sorted(str(p.relative_to(self.root))
                            for p in (self.root / prefix).rglob("*.py")))

    def sha(self, rel):
        import hashlib
        return hashlib.sha256(self.read(rel).encode()).hexdigest()[:16]


@pytest.fixture
def ws(tmp_path, stock_dir):
    return Workspace(tmp_path / "a01", agent_id="a01", source=FakeStock(stock_dir))


def test_reads_stock_until_edited(ws):
    assert "CHUNK = 8192" in ws.read(P)
    ws.replace(P, "8192", "16384")
    assert "CHUNK = 16384" in ws.read(P)
    assert "CHUNK = 8192" in ws.stock_text(P), "stock must stay pristine"


def test_ambiguous_edit_is_refused(ws):
    """An edit meant for one place that matches three produces a plausible diff
    with different behaviour -- the most expensive kind of wrong here."""
    with pytest.raises(ValueError, match="occurs 0 times"):
        ws.replace(P, "not_present_anywhere", "x")
    ws.edit(P, "A = 1\nA = 1\n")
    with pytest.raises(ValueError, match="occurs 2 times"):
        ws.replace(P, "A = 1", "A = 2")


def test_syntax_errors_are_caught_before_a_gpu_is_rented(ws):
    ws.edit(P, "def broken(:\n")
    ok, why = ws.check()
    assert not ok and "syntax error" in why


def test_a_no_op_is_not_a_stack(ws):
    """A stack of unmodified files is a no-op wearing a diff's clothes.

    Exactly what the deleted `overlays/` turned out to be: two pristine
    vendored files whose "application" changed nothing.
    """
    ok, why = ws.check()
    assert not ok and "no files changed" in why
    # An agent editing in place starts from stock copies; those alone are still
    # not a diff.
    ws.materialise(P)
    ok, why = ws.check()
    assert not ok and "no files changed" in why
    assert ws.touched() == ()


def test_materialise_never_discards_work(ws):
    ws.materialise(P)
    ws.replace(P, "8192", "16384")
    ws.materialise(P)                     # called again by the next attempt
    assert "16384" in ws.read(P)


def test_a_syntax_error_anywhere_blocks_the_stack(ws):
    """Even in a file the agent later reverted: it is still six GPU-minutes."""
    ws.replace(P, "8192", "16384")
    ws.edit("srt/mem_cache/radix_cache.py", "class Broken(:\n")
    ok, why = ws.check()
    assert not ok and "syntax error" in why


def test_stack_carries_upstream_hashes(ws):
    """Without them a stale diff silently reverts upstream changes while still
    looking like a valid experiment."""
    ws.replace(P, "8192", "16384")
    st = ws.stack()
    assert st.files and list(st.upstream_sha) == [P]
    assert st.upstream_sha[P] == ws.source.sha(P)


def test_run_dirs_exist_before_the_simulator_needs_them(ws):
    """The replicate of a win failed all night as 'infra' because its run
    directory was a string with a suffix, never a directory."""
    assert ws.run_dir(1).is_dir()
    rep = ws.run_dir(1, "-rep1")
    assert rep.is_dir() and rep.name == "attempt-001-rep1"


def test_stack_refuses_to_build_from_a_broken_workspace(ws):
    ws.edit(P, "def broken(:\n")
    with pytest.raises(ValueError, match="not a valid stack"):
        ws.stack()


def test_diff_is_reviewable(ws):
    ws.replace(P, "8192", "16384")
    d = ws.diff()
    assert "-CHUNK = 8192" in d and "+CHUNK = 16384" in d


def test_reset_returns_to_stock(ws):
    ws.replace(P, "8192", "16384")
    ws.reset()
    assert ws.touched() == () and "CHUNK = 8192" in ws.read(P)


def test_each_attempt_gets_its_own_run_dir(ws):
    a, b = ws.run_dir(0), ws.run_dir(1)
    assert a != b and a.is_dir() and b.is_dir()


def test_serving_json_is_the_launch_line(ws):
    """Every launch argument is the candidate's to set; a typo is caught
    before a GPU is rented; the launch overrides travel with the stack."""
    import json

    ok, why = ws.check()
    assert not ok and "serving.json" in why
    (ws.candidates / "serving.json").write_text(json.dumps(
        {"serving": {"chunked_prefill_size": 16384, "schedule_policy": "lpm"},
         "env": {"SGLANG_FOO": 1}}))
    ok, why = ws.check()
    assert ok, why
    st = ws.stack()
    assert st.serving == {"chunked_prefill_size": 16384, "schedule_policy": "lpm"}
    assert st.env == {"SGLANG_FOO": "1"} and not st.files
    assert "chunked_prefill_size=16384" in ws.diff()
    (ws.candidates / "serving.json").write_text(json.dumps({"model": "other"}))
    ok, why = ws.check()
    assert not ok and "not allowed" in why
    (ws.candidates / "serving.json").write_text("{not json")
    ok, why = ws.check()
    assert not ok and "not JSON" in why
