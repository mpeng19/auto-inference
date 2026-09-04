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
    """A stack of unmodified files is a no-op wearing a diff's clothes."""
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


def test_tools_run_from_inside_candidate_find_the_agent_root(ws):
    """`harness tool ... --workspace .` from candidate/ nested a second
    workspace there and re-materialised the targets over the agent's edits."""
    from harness.agent.workspace import Workspace

    assert Workspace.locate(ws.candidates) == ws.root            # candidate/sglang
    assert Workspace.locate(ws.candidates.parent) == ws.root     # candidate
    assert Workspace.locate(ws.root) == ws.root
    again = Workspace(ws.candidates, source=ws.source)
    assert again.root == ws.root
    assert not (ws.candidates / "candidate").exists()
    assert not (ws.candidates.parent / "candidate").exists()


def test_missing_targets_are_skipped_and_named(ws):
    present = ws.materialise("srt/managers/schedule_policy.py", "srt/nope/gone.py")
    assert present == ("srt/managers/schedule_policy.py",)
    assert ws.missing_targets == ("srt/nope/gone.py",)


def test_a_nested_sglang_directory_is_refused_not_shipped(ws):
    """An agent that treats the package root as a repo root writes
    `sglang/srt/...`; the container would never load it."""
    ws.materialise("srt/managers/schedule_policy.py")
    p = ws.candidates / "sglang" / "srt" / "x.py"
    p.parent.mkdir(parents=True)
    p.write_text("x = 1\n")
    ok, why = ws.check()
    assert not ok and "package root" in why
    assert "sglang/srt/x.py" not in ws.touched()


def test_new_files_have_no_upstream_sha(ws):
    ws.materialise("srt/managers/schedule_policy.py")
    (ws.candidates / "srt" / "layers").mkdir(parents=True, exist_ok=True)
    (ws.candidates / "srt" / "layers" / "new_kernel.py").write_text("k = 1\n")
    st = ws.stack()
    assert "srt/layers/new_kernel.py" in st.files
    assert "srt/layers/new_kernel.py" not in st.upstream_sha


def test_skills_are_installed_where_claude_code_looks(ws):
    paths = ws.install_skills({"tracedb": "---\nname: tracedb\n---\nquery it", "empty": ""})
    assert [p.parent.name for p in paths] == ["tracedb"]
    assert (ws.candidates / ".claude" / "skills" / "tracedb" / "SKILL.md").read_text().endswith("query it")
    ws.materialise("srt/managers/schedule_policy.py")
    assert ws.touched() == () and ws.misplaced() == ()      # not part of the diff


# ── compounding: a workspace built on a saved stack ─────────────────────

BASE_ONLY = "srt/new/base_only.py"


def _base(stock_dir):
    """The best stack of a previous fleet: one edited file, one new file, a
    launch line and an env var."""
    from simulator import InferenceStack

    return InferenceStack(
        files={P: "CHUNK = 4096\n\n\nclass SchedulePolicy:\n    pass\n",
               BASE_ONLY: "B = 1\n"},
        upstream_sha={P: FakeStock(stock_dir).sha(P)},
        serving={"chunked_prefill_size": 4096, "extra_args": ["--a"]},
        env={"SGLANG_X": "1"}, label="build-1 best")


@pytest.fixture
def based(tmp_path, stock_dir):
    return Workspace(tmp_path / "a02", agent_id="a02", source=FakeStock(stock_dir),
                     base=_base(stock_dir))


def test_base_files_show_through_read(based):
    """The agent's "current file" is the base's version, not stock's."""
    assert "CHUNK = 4096" in based.read(P) and "CHUNK = 4096" in based.stock_text(P)
    assert based.read(BASE_ONLY) == "B = 1\n"
    assert based.materialise(P, BASE_ONLY) == (P, BASE_ONLY)
    assert based.touched() == (), "base copies are not a diff"
    ok, why = based.check()
    assert not ok and "identical to the base" in why


def test_touched_is_relative_to_the_base_not_stock(based):
    """Reverting the base's edit back to stock text IS a change."""
    based.edit(P, "CHUNK = 8192\n\n\nclass SchedulePolicy:\n    pass\n")
    assert based.touched() == (P,)
    assert "-CHUNK = 4096" in based.diff() and "+CHUNK = 8192" in based.diff()
    assert "base/" + P in based.diff()


def test_stack_is_the_full_base_plus_the_edits(based, tmp_path, stock_dir):
    """The runner gets base files it never saw plus the agent's, and the
    digest is different from the same edit made on stock."""
    based.replace(P, "4096", "2048")
    st = based.stack()
    assert "CHUNK = 2048" in st.files[P] and st.files[BASE_ONLY] == "B = 1\n"
    # Drift is measured against the installed package, not the base's text.
    assert st.upstream_sha[P] == FakeStock(stock_dir).sha(P)
    assert BASE_ONLY not in st.upstream_sha
    assert st.serving == {"chunked_prefill_size": 4096, "extra_args": ["--a"]}
    assert st.env == {"SGLANG_X": "1"}
    assert st.base == based.base.digest and "build-1 best" in st.label
    plain = Workspace(tmp_path / "a03", source=FakeStock(stock_dir))
    plain.edit(P, st.files[P])
    assert plain.stack().digest != st.digest


def test_serving_and_env_layer_over_the_base(based):
    import json

    (based.candidates / "serving.json").write_text(json.dumps(
        {"serving": {"schedule_policy": "lpm", "extra_args": ["--b"]},
         "env": {"SGLANG_Y": "2"}}))
    ok, why = based.check()
    assert ok, why
    st = based.stack()
    assert st.serving == {"chunked_prefill_size": 4096, "schedule_policy": "lpm",
                          "extra_args": ["--a", "--b"]}
    assert st.env == {"SGLANG_X": "1", "SGLANG_Y": "2"}
    assert st.files[P] == based.base.files[P], "base files travel even with no edit"
    # Base value replaced, not appended: with_overrides semantics.
    (based.candidates / "serving.json").write_text(json.dumps(
        {"chunked_prefill_size": 16384}))
    assert based.stack().serving["chunked_prefill_size"] == 16384


def test_reset_returns_to_the_base_not_stock(based):
    """The loop resets before every attempt, and after a stalled call; with
    a base that must land on the base's files, or the next attempt would
    silently be measured against stock."""
    based.materialise(P)
    based.replace(P, "4096", "16384")
    (based.candidates / "serving.json").write_text('{"serving": {"max_running_requests": 8}}')
    (based.candidates / "ablation.env").write_text("X=1\n")
    assert based.touched() == (P,)
    based.reset()
    assert based.touched() == () and "CHUNK = 4096" in based.read(P)
    assert based.read(BASE_ONLY) == "B = 1\n"
    assert not (based.candidates / "serving.json").exists()
    assert not (based.candidates / "ablation.env").exists()
    assert based.base is not None and (based.root / "base.json").is_file()


def test_the_base_is_found_from_base_json(based, stock_dir):
    """`harness tool` builds `Workspace(root)` from the agent's shell with no
    base argument; it must see the same base the daemon set."""
    assert (based.root / "base.json").is_file()
    again = Workspace(based.root, source=FakeStock(stock_dir))
    assert again.base is not None and again.base.digest == based.base.digest
    assert "CHUNK = 4096" in again.read(P)
    again.set_base(None)
    assert not (based.root / "base.json").exists()
    assert "CHUNK = 8192" in Workspace(based.root, source=FakeStock(stock_dir)).read(P)


def test_compose_is_the_rule_the_workspace_follows():
    from simulator import InferenceStack

    base = InferenceStack(files={"a.py": "1", "b.py": "1"}, patches={"c.py": "p"},
                          upstream_sha={"a.py": "sa"}, serving={"x": 1, "extra_args": ["--a"]},
                          env={"E": "1", "F": "1"}, label="base")
    over = InferenceStack(files={"b.py": "2", "d.py": "2"}, upstream_sha={"b.py": "sb"},
                          serving={"x": 2, "extra_args": ["--b"]}, env={"F": "2"}, label="edit")
    full = InferenceStack.compose(base, over)
    assert full.files == {"a.py": "1", "b.py": "2", "d.py": "2"}
    assert full.patches == {"c.py": "p"}
    assert full.upstream_sha == {"a.py": "sa", "b.py": "sb"}
    assert full.serving == {"x": 2, "extra_args": ["--a", "--b"]}
    assert full.env == {"E": "1", "F": "2"}
    assert full.base == base.digest and full.label == "edit on base"
    # Round trip keeps the provenance; a plain stack's dict is unchanged.
    assert InferenceStack.from_dict(full.as_dict()) == full
    assert "base" not in base.as_dict()
    assert InferenceStack.compose(base, InferenceStack()).digest == base.digest
