"""The tools an agent uses between sweeps."""
import json
import pathlib

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


# ── the GPU workbench ────────────────────────────────────────────────────

def _fake_workbench(monkeypatch, seen):
    """Stand in for the Modal call, recording what would have been run."""
    from simulator import Simulator

    async def fake(self, text, files=None, timeout_s=600):
        seen.update(text=text, files=files, timeout_s=timeout_s,
                    stack_digest=self.stack.digest, root=str(self.root))
        return {"ok": True, "exit_code": 0, "stdout": text, "stderr": "",
                "elapsed_s": 1.0, "gpu": "NVIDIA H100 80GB HBM3",
                "cost_usd": 0.05, "dir": str(self.root / "workbench-0")}

    monkeypatch.setattr(Simulator, "workbench", fake)


def test_gpu_run_refuses_a_missing_script(tmp_path):
    """The cheapest failure available: no GPU is rented to discover it."""
    rep = tools.gpu_run(tmp_path / "not-written-yet.py")
    assert not rep["ok"] and "no script at" in rep["error"]
    assert "not-written-yet.py" in rep["error"]


def test_gpu_run_refuses_a_workspace_that_will_not_parse(tmp_path, stock_dir):
    """Exactly preflight's argument -- a GPU is an expensive place to find a
    syntax error, and this one rents an H100 for minutes."""
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.edit("srt/managers/schedule_policy.py", "def broken(:\n")
    (tmp_path / "p.py").write_text("print(1)")
    rep = tools.gpu_run(tmp_path / "p.py", workspace=ws.root, source=src)
    assert not rep["ok"] and "syntax error" in rep["error"]


def test_gpu_run_carries_the_candidate_stack_to_the_gpu(tmp_path, stock_dir,
                                                        monkeypatch):
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.replace("srt/managers/schedule_policy.py", "8192", "16384")
    (tmp_path / "p.py").write_text("print('bench')")

    seen = {}
    _fake_workbench(monkeypatch, seen)
    rep = tools.gpu_run(tmp_path / "p.py", workspace=ws.root, timeout_s=120,
                        source=src)
    assert rep["ok"] and rep["cost_usd"] == 0.05
    assert seen["text"] == "print('bench')" and seen["timeout_s"] == 120
    assert seen["stack_digest"] == ws.stack().digest
    assert seen["root"] == str(ws.root), "artifacts belong in the agent's dir"
    assert "note" not in rep


def test_gpu_run_on_an_unedited_workspace_runs_stock_and_says_so(tmp_path,
                                                                 stock_dir,
                                                                 monkeypatch):
    """Measuring what stock's kernel costs is the right first move, so this is
    not an error -- but a result that did not say which code it measured would
    be worse than useless."""
    from harness.agent.workspace import Workspace
    from simulator import InferenceStack

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    (tmp_path / "p.py").write_text("print(1)")

    seen = {}
    _fake_workbench(monkeypatch, seen)
    rep = tools.gpu_run(tmp_path / "p.py", workspace=ws.root, source=src)
    assert rep["ok"] and "stock" in rep["note"]
    assert seen["stack_digest"] == InferenceStack.stock().digest


def test_gpu_run_writes_helper_files_beside_the_script(tmp_path, monkeypatch):
    (tmp_path / "p.py").write_text("import kern")
    seen = {}
    _fake_workbench(monkeypatch, seen)
    tools.gpu_run(tmp_path / "p.py", workspace=tmp_path,
                  files={"kern.py": "K = 1\n"})
    assert seen["files"] == {"kern.py": "K = 1\n"}


def test_equivalence_refuses_a_workspace_that_will_not_parse(tmp_path, stock_dir):
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.edit("srt/mem_cache/radix_cache.py", "class Broken(:\n")
    rep = tools.equivalence(workspace=ws.root, source=src)
    assert not rep["ok"] and "syntax error" in rep["error"]


def test_equivalence_reaches_the_measurement_with_its_thresholds(tmp_path,
                                                                 stock_dir,
                                                                 monkeypatch):
    from harness.agent.workspace import Workspace
    from simulator.measure import equivalence as eq

    from .test_workspace import FakeStock

    src = FakeStock(stock_dir)
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=src)
    ws.replace("srt/managers/schedule_policy.py", "8192", "16384")

    seen = {}

    async def fake(sim, **kw):
        seen.update(kw, stack_digest=sim.stack.digest)
        return {"ok": True, "regressed": False, "why": "", "cost_usd": 0.9,
                "summary": "2016 positions"}

    monkeypatch.setattr(eq, "measure", fake)
    rep = tools.equivalence(workspace=ws.root, source=src)
    assert rep["ok"] and rep["cost_usd"] == 0.9
    assert seen["min_agreement"] == eq.MIN_AGREEMENT
    assert seen["max_mean_dlogprob"] == eq.MAX_MEAN_DLOGPROB
    assert seen["stack_digest"] == ws.stack().digest


def test_ncu_builds_a_driver_and_parses_the_counters(tmp_path, stock_dir, monkeypatch):
    """The driver runs the agent's script under ncu without clock locking
    (the container cannot lock clocks) and turns the CSV into per-kernel
    rows; a run with no parseable output is reported as not ok."""
    import simulator
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    ws = Workspace(tmp_path / "a01", agent_id="a01", source=FakeStock(stock_dir))
    ws.materialise("srt/managers/schedule_policy.py")
    script = tmp_path / "bench.py"
    script.write_text("print('hi')\n")
    seen = {}

    class Stub:
        def __init__(self, **kw):
            seen["stack"] = kw["stack"]

        async def workbench(self, text, files=None, timeout_s=0):
            seen["driver"] = text
            seen["files"] = files
            return {"ok": True, "stdout": 'NCU_JSON {"rc": 0, "kernels": {"k0": {"launches": 2, '
                                          '"gpu__time_duration.sum": 176.3, '
                                          '"dram__throughput.avg.pct_of_peak_sustained_elapsed": 21.6}}}',
                    "stderr": "", "cost_usd": 0.02, "gpu": "H100"}

    real = simulator.Simulator
    simulator.Simulator = Stub
    try:
        rep = tools.ncu(script, workspace=ws.root, kernel="gemm", source=ws.source)
    finally:
        simulator.Simulator = real
    assert rep["ok"] and rep["ncu"]["kernels"]["k0"]["dram__throughput.avg.pct_of_peak_sustained_elapsed"] == 21.6
    assert "--clock-control" in seen["driver"] and "'gemm'" in seen["driver"]
    assert seen["files"] == {"target.py": "print('hi')\n"}
    assert tools.ncu(tmp_path / "missing.py", workspace=ws.root, source=ws.source)["ok"] is False


# ── ablation: the mechanism's share of the delta ─────────────────────────

class _FakeEval:
    def __init__(self, ok, bill, n_star=12, reason=""):
        self.ok, self.bill_per_1k, self.n_star, self.reason = ok, bill, n_star, reason
        self.record = {"serving": {"gpu": "H100", "n_gpu": 1}, "started_at": 100.0,
                       "finished_at": 700.0, "levels": []}


def _stub_simulator(prices: dict, seen: dict):
    """A Simulator that prices a stack by whether its env carries the kill
    switch, and records what it was asked to run."""

    class Stub:
        def __init__(self, root_dir, stack, levels=(), seconds_per_level=0.0, **kw):
            assert pathlib.Path(root_dir).is_dir(), "the tool makes the run dirs first"
            self.root, self.stack = pathlib.Path(root_dir), stack
            seen.setdefault("runs", []).append(
                {"dir": str(root_dir), "env": dict(stack.env), "levels": tuple(levels),
                 "seconds": seconds_per_level, "digest": stack.digest})

        async def submit_async(self):
            return f"call-{len(seen['runs'])}"

        async def collect(self, call_id):
            seen.setdefault("collected", []).append(call_id)
            key = "disabled" if "SGLANG_DISABLE_X" in self.stack.env else "as_is"
            p = prices[key]
            return _FakeEval(p is not None, p, reason="" if p is not None else "no level met the SLO")

    return Stub


def _edited_ws(tmp_path, stock_dir, fleet_baseline=None):
    from harness.agent.workspace import Workspace

    from .test_workspace import FakeStock

    root = tmp_path / "agents" / "S"
    root.mkdir(parents=True)
    if fleet_baseline is not None:
        (root / "fleet.json").write_text(json.dumps({"baseline": fleet_baseline}))
    ws = Workspace(root / "a01", agent_id="a01", source=FakeStock(stock_dir))
    ws.materialise("srt/managers/schedule_policy.py")
    ws.edit("srt/managers/schedule_policy.py", "CHUNK = 16384\n\n\nclass SchedulePolicy:\n    pass\n")
    return ws


def test_ablate_prices_the_stack_with_and_without_its_kill_switch(tmp_path, stock_dir):
    """Two sweeps at the tier's grid, one with the env applied; the record
    says how much of the delta against baseline the mechanism explains, and
    the verdict says it in a paragraph. Screen baseline $17.52, as-is $15.00,
    disabled $17.40: the mechanism explains 84% and the disabled stack sits
    within noise of baseline."""
    import simulator

    ws = _edited_ws(tmp_path, stock_dir,
                    fleet_baseline={"bill_per_1k": 14.96, "screen": {"bill_per_1k": 17.52}})
    seen: dict = {}
    real = simulator.Simulator
    simulator.Simulator = _stub_simulator({"as_is": 15.0, "disabled": 17.4}, seen)
    try:
        rep = tools.ablate(ws.root, env={"SGLANG_DISABLE_X": "1"}, tier="screen", source=ws.source)
    finally:
        simulator.Simulator = real
    assert rep["ok"] and rep["tier"] == "screen"
    assert [r["levels"] for r in seen["runs"]] == [(8, 12), (8, 12)]
    assert {r["seconds"] for r in seen["runs"]} == {60.0}
    assert seen["runs"][0]["env"] == {} and seen["runs"][1]["env"] == {"SGLANG_DISABLE_X": "1"}
    assert len(seen["collected"]) == 2
    assert rep["as_is"]["bill_per_1k"] == 15.0 and rep["disabled"]["bill_per_1k"] == 17.4
    assert rep["as_is"]["n_star"] == 12
    assert rep["baseline_bill_per_1k"] == 17.52                       # the screen baseline
    assert rep["delta_pct"] == pytest.approx(-13.79, abs=0.01)
    assert rep["total_pct"] == pytest.approx(-14.38, abs=0.01)
    assert rep["explained_pct"] == pytest.approx(95.2, abs=0.1)
    assert rep["explains"] is True
    assert rep["stack_digest"] != rep["disabled_digest"]
    assert rep["cost_usd"] > 0
    assert "accounts for 95% of that -14.4% delta" in rep["verdict"]
    assert "within the 3% noise floor" in rep["verdict"] and "$" in rep["verdict"]
    on_disk = json.loads((ws.root / "ablations" / "0" / "ablation.json").read_text())
    assert on_disk["env"] == {"SGLANG_DISABLE_X": "1"} and on_disk["stack_digest"] == rep["stack_digest"]
    assert (ws.root / "ablations" / "0" / "as-is").is_dir() and (ws.root / "ablations" / "0" / "disabled").is_dir()
    ledger = (ws.root / "spend.jsonl").read_text()
    assert '"ablate"' in ledger
    # a second run gets its own directory
    simulator.Simulator = _stub_simulator({"as_is": 15.0, "disabled": 15.2}, {})
    try:
        again = tools.ablate(ws.root, env={"SGLANG_DISABLE_X": "1"}, tier="screen", source=ws.source)
    finally:
        simulator.Simulator = real
    assert again["dir"].endswith("ablations/1") and again["explains"] is False
    assert "cannot claim it" in again["verdict"]


def test_ablate_without_a_baseline_reports_prices_but_no_share(tmp_path, stock_dir):
    import simulator

    ws = _edited_ws(tmp_path, stock_dir)
    real = simulator.Simulator
    simulator.Simulator = _stub_simulator({"as_is": 15.0, "disabled": 17.4}, {})
    try:
        rep = tools.ablate(ws.root, env={"SGLANG_DISABLE_X": "1"}, tier="full", source=ws.source)
    finally:
        simulator.Simulator = real
    assert rep["ok"] and rep["levels"] == [4, 8, 12, 16, 24] and rep["seconds_per_level"] == 120.0
    assert rep["explained_pct"] is None and rep["explains"] is None
    assert "No baseline" in rep["verdict"] and rep["delta_pct"] == pytest.approx(-13.79, abs=0.01)


def test_ablate_refuses_before_renting_a_gpu(tmp_path, stock_dir):
    import simulator

    ws = _edited_ws(tmp_path, stock_dir)
    assert "kill switch" in tools.ablate(ws.root, env={}, source=ws.source)["error"]
    assert "unknown tier" in tools.ablate(ws.root, env={"X": "1"}, tier="huge", source=ws.source)["error"]
    ws.edit("srt/managers/schedule_policy.py", "def (:\n")
    assert "not runnable" in tools.ablate(ws.root, env={"X": "1"}, source=ws.source)["error"]
    # a failed sweep is reported, not scored
    ws.edit("srt/managers/schedule_policy.py", "CHUNK = 1\n")
    real = simulator.Simulator
    simulator.Simulator = _stub_simulator({"as_is": 15.0, "disabled": None}, {})
    try:
        rep = tools.ablate(ws.root, env={"SGLANG_DISABLE_X": "1"}, source=ws.source)
    finally:
        simulator.Simulator = real
    assert not rep["ok"] and rep["explains"] is None
    assert "disabled sweep did not price" in rep["verdict"]


def test_ablate_tiers_match_the_daemon():
    from harness.daemon import FleetConfig

    cfg = FleetConfig(session_id="x", root="/tmp/x")
    assert tools.TIERS["screen"] == (cfg.screen_levels, cfg.screen_seconds)
    assert tools.TIERS["full"] == (cfg.levels, cfg.seconds_per_level)


def test_env_flags_are_parsed_in_every_shape_the_cli_hands_over():
    assert tools._parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    assert tools._parse_env("A=1,B=2") == {"A": "1", "B": "2"}
    assert tools._parse_env({"A": 1}) == {"A": "1"}
    assert tools._parse_env(None) == {}


def test_equivalence_keeps_its_record_under_the_agent(tmp_path, stock_dir, monkeypatch):
    """`results.py` reads decode agreement from `<agent>/equivalence/`; the
    record has to land there, keyed by the stack digest."""
    from simulator.measure import equivalence as eq

    ws = _edited_ws(tmp_path, stock_dir)

    async def fake(sim, timeout_s, min_agreement, max_mean_dlogprob):
        return {"ok": True, "stack": "x", "stack_digest": sim.stack.digest, "cost_usd": 0.4,
                "regressed": False, "why": "", "lossless": False, "decode_agreement": 0.77,
                "summary": "s", "result": {"decode_agreement": 0.77}}

    monkeypatch.setattr(eq, "measure", fake)
    rep = tools.equivalence(workspace=ws.root, source=ws.source)
    files = list((ws.root / "equivalence").glob("*.json"))
    assert len(files) == 1 and files[0].name.startswith(rep["stack_digest"])
    assert json.loads(files[0].read_text())["decode_agreement"] == 0.77
    assert rep["path"] == str(files[0])
