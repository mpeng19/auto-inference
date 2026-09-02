"""The tools an agent uses between sweeps."""
import pytest

from harness import tools
from harness.contracts import Experiment
from harness.memory import SqliteMemory


def test_roofline_reproduces_the_measured_calibration():
    """docs/methodology.md 8.3: the KV read runs at 0.22-0.28 of bandwidth."""
    r = tools.roofline(context=20583, batch=12, n_gpu=1)
    assert 0.20 <= r.f_kv <= 0.32
    assert 0.60 <= r.f_weights <= 0.80
    assert r.weights_gb == pytest.approx(23.1, abs=0.2)
    assert r.per_seq_gb == pytest.approx(1.504, abs=0.01)


def test_cost_per_token_falls_toward_a_floor_not_to_zero():
    """The per-sequence term does not amortise, which is the whole finding."""
    small = tools.roofline(batch=4).usd_per_m_output
    big = tools.roofline(batch=64).usd_per_m_output
    huge = tools.roofline(batch=512).usd_per_m_output
    assert big < small
    assert huge > 0.5 * big, "cost must approach a floor, not collapse"


def test_more_gpus_lower_the_step_time():
    assert tools.roofline(batch=12, n_gpu=2).measured_step_ms < \
        tools.roofline(batch=12, n_gpu=1).measured_step_ms


def test_preflight_blocks_an_undefined_name(tmp_path, stock_dir):
    """A NameError costs six GPU-minutes to discover; ruff finds it free."""
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.edit("srt/managers/schedule_policy.py",
            "CHUNK = 8192\n\n\nclass SchedulePolicy:\n    x = undefined_thing\n")
    rep = tools.preflight(ws.root, source=src)
    assert not rep["ok"]
    assert any("F821" in ln for ln in rep["lint"])


def test_preflight_passes_a_clean_edit(tmp_path, stock_dir):
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.replace("srt/managers/schedule_policy.py", "8192", "16384")
    rep = tools.preflight(ws.root, source=src)
    assert rep["ok"] and rep["touched"] == ["srt/managers/schedule_policy.py"]
    assert "lint_skipped" not in rep, "the lint must actually have run"
    assert rep["diff_lines"] > 0


def test_preflight_rejects_an_empty_workspace(tmp_path, stock_dir):
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    assert not tools.preflight(ws.root, source=src)["ok"]


def test_recall_reaches_the_fleet_memory(tmp_path):
    db = tmp_path / "memory.db"
    m = SqliteMemory(db)
    m.record(Experiment(hypothesis="raise chunked_prefill_size to 16384",
                        verdict="loss", summary="TTFT rose 40%"))
    rep = tools.recall("I want to raise chunked prefill size", root=tmp_path)
    assert rep["found"]
    assert "did NOT work" in rep["brief"]
    assert rep["hits"] and rep["hits"][0]["verdict"] == "loss"


def test_recall_says_so_when_there_is_no_memory(tmp_path):
    rep = tools.recall("anything", root=tmp_path / "nothing-here")
    assert not rep["found"] and rep["brief"] == ""


def test_preflight_degrades_when_ruff_is_missing(tmp_path, stock_dir, monkeypatch):
    """An agent's machine is not guaranteed to have ruff. A clean report must
    not then be mistaken for a thorough one."""
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    monkeypatch.setattr(tools, "_ruff", lambda: None)
    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.replace("srt/managers/schedule_policy.py", "8192", "16384")
    rep = tools.preflight(ws.root, source=src)
    assert rep["ok"] and "ruff not found" in rep["lint_skipped"]


def test_a_broken_linter_is_reported_not_read_as_clean(tmp_path, stock_dir,
                                                       monkeypatch):
    """Selecting a rule ruff had removed made it exit 2 and report nothing.

    A tool whose job is catching mistakes must never fail silently clean.
    """
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    monkeypatch.setattr(tools, "_run_ruff",
                        lambda ruff, paths: __import__("subprocess").run(
                            [*ruff, "check", "--not-a-real-flag", *paths],
                            capture_output=True, text=True))
    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.replace("srt/managers/schedule_policy.py", "8192", "16384")
    rep = tools.preflight(ws.root, source=src)
    assert "lint_skipped" in rep and "exited" in rep["lint_skipped"]
