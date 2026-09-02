"""The GPU workbench: one script on an H100, minutes instead of a sweep.

No Modal call is made here -- the function handle is faked the same way
`test_api.py::test_submit_records_the_call_id_on_both_paths` fakes the sweep's.
What is asserted is the part that stays on this side of the wire: what gets
written, where, and in what order.
"""
import json

import pytest

from simulator import InferenceStack, Simulator

RESULT = {"ok": True, "exit_code": 0, "stdout": "hello\n", "stderr": "",
          "elapsed_s": 3.5, "gpu": "NVIDIA H100 80GB HBM3",
          "stack": {"digest": "abc", "applied": []}, "cost_usd": 0.004}


class Getter:
    """`call.get.aio(timeout=0)` is how Modal's async client spells a fetch."""

    def __init__(self, result):
        self._result = result

    async def aio(self, timeout=None):
        return self._result


class Call:
    object_id = "fc-workbench-1"

    def __init__(self, result):
        self.get = Getter(result)


def fake_fn(result=None, seen=None):
    """A stand-in for `modal.Function`, recording what was spawned."""
    class Spawn:
        async def aio(self, *a):
            if seen is not None:
                seen.append(a)
            return Call(RESULT if result is None else result)

    class Fn:
        spawn = Spawn()

    return lambda self, timeout_s: Fn()


def test_every_artifact_of_a_run_lands_in_its_own_directory(root, monkeypatch):
    import asyncio

    monkeypatch.setattr(Simulator, "_workbench_fn", fake_fn())
    sim = Simulator(root_dir=root)
    rec = asyncio.run(sim.workbench("print('hello')"))

    d = root / "workbench-0"
    assert rec["dir"] == str(d)
    assert (d / "script.py").read_text() == "print('hello')"
    assert (d / "stdout.txt").read_text() == "hello\n"
    assert (d / "stderr.txt").read_text() == ""
    assert json.loads((d / "result.json").read_text())["exit_code"] == 0
    for name in ("script", "stdout", "stderr", "result"):
        assert name in rec["artifacts"]


def test_runs_are_numbered_in_the_order_they_were_made(root, monkeypatch):
    """An agent iterating on one kernel wants them in the order it ran them,
    not sorted by a timestamp it has to decode."""
    import asyncio

    monkeypatch.setattr(Simulator, "_workbench_fn", fake_fn())
    sim = Simulator(root_dir=root)
    assert asyncio.run(sim.workbench("a"))["dir"].endswith("workbench-0")
    assert asyncio.run(sim.workbench("b"))["dir"].endswith("workbench-1")
    assert (root / "workbench-0" / "script.py").read_text() == "a"
    assert (root / "workbench-1" / "script.py").read_text() == "b"


def test_the_script_and_call_id_are_on_disk_before_the_wait_begins(root, monkeypatch):
    """A script that hangs still has to leave something to look at."""
    import asyncio

    monkeypatch.setattr(Simulator, "_workbench_fn", fake_fn())
    sim = Simulator(root_dir=root)
    asyncio.run(sim.workbench("print(1)"))
    assert (root / "workbench-0" / "call_id").read_text() == "fc-workbench-1"


def test_the_stack_and_the_helpers_reach_the_runner(root, monkeypatch):
    """A workbench run measures the candidate or it measures nothing."""
    import asyncio

    seen = []
    monkeypatch.setattr(Simulator, "_workbench_fn", fake_fn(seen=seen))
    stack = InferenceStack(files={"srt/managers/schedule_policy.py": "X = 1\n"},
                           label="candidate")
    sim = Simulator(root_dir=root, stack=stack)
    asyncio.run(sim.workbench("import kern", files={"kern.py": "K = 2\n"},
                              timeout_s=120))
    (sent_stack, script, timeout, files), = seen
    assert sent_stack["files"]["srt/managers/schedule_policy.py"] == "X = 1\n"
    assert script == "import kern" and timeout == 120
    assert files == {"kern.py": "K = 2\n"}


def test_a_failed_run_still_writes_what_it_printed(root, monkeypatch):
    """The stderr of a kernel that would not compile is the entire result."""
    import asyncio

    bad = {**RESULT, "ok": False, "exit_code": 1, "stdout": "",
           "stderr": "triton.CompilationError: ...\n"}
    monkeypatch.setattr(Simulator, "_workbench_fn", fake_fn(result=bad))
    rec = asyncio.run(Simulator(root_dir=root).workbench("boom"))
    assert not rec["ok"]
    assert "CompilationError" in (root / "workbench-0" / "stderr.txt").read_text()


# ── what the runner itself does with the script's neighbours ─────────────

def test_a_helper_file_may_not_escape_the_scratch_directory(tmp_path):
    from simulator.runner.modal_runner import _write_helpers

    with pytest.raises(ValueError, match="escapes"):
        _write_helpers(tmp_path, {"../evil.py": "x"})


def test_a_helper_file_may_not_shadow_the_applied_sglang(tmp_path):
    """Python puts the script's own directory ahead of PYTHONPATH, so a
    `sglang.py` beside it would silently be the package under test."""
    from simulator.runner.modal_runner import _write_helpers

    for name in ("sglang.py", "sglang/srt/x.py"):
        with pytest.raises(ValueError, match="shadow"):
            _write_helpers(tmp_path, {name: "x"})


def test_helpers_are_written_where_the_script_can_import_them(tmp_path):
    from simulator.runner.modal_runner import _write_helpers

    assert _write_helpers(tmp_path, {"pkg/kern.py": "K = 1\n"}) == ["pkg/kern.py"]
    assert (tmp_path / "pkg" / "kern.py").read_text() == "K = 1\n"


def test_the_workbench_is_billed_for_four_vcpus_not_sixteen(tmp_path):
    """vCPUs are billed on top of the GPU. The sweep needs 16 to drive its load
    generator; a workbench script has nothing to feed, and 12 idle cores would
    be $1.64/hour of nothing."""
    from simulator import costs
    from simulator.runner import modal_runner as r

    assert r.WORKBENCH_VCPU == 4.0
    out = r._bill({}, __import__("time").perf_counter() - 3600.0)
    assert out["cost_usd"] == pytest.approx(
        costs.container_rate("H100", 1, vcpu=4.0), rel=0.01)
    assert out["cost_usd"] < costs.container_rate("H100", 1, vcpu=16.0)


# ── the command line over it ─────────────────────────────────────────────

def test_cli_refuses_a_script_that_is_not_there(root, capsys):
    """Before renting anything: the cheapest failure there is."""
    from simulator.cli import main

    assert main(["workbench", "--root", str(root), "no-such-file.py"]) == 2
    assert "no script at" in capsys.readouterr().err


def test_cli_sends_the_named_script_and_the_timeout(root, tmp_path, monkeypatch,
                                                    capsys):
    from simulator.cli import main

    seen = {}

    async def fake(self, text, files=None, timeout_s=600):
        seen.update(text=text, timeout_s=timeout_s)
        return {**RESULT, "dir": str(root)}

    monkeypatch.setattr(Simulator, "workbench", fake)
    script = tmp_path / "probe.py"
    script.write_text("print('kernel')")
    assert main(["workbench", "--root", str(root), "--timeout", "42",
                 str(script)]) == 0
    assert seen == {"text": "print('kernel')", "timeout_s": 42}
    assert "hello" in capsys.readouterr().out


def test_cli_exits_nonzero_when_the_script_failed(root, tmp_path, monkeypatch):
    from simulator.cli import main

    async def fake(self, text, files=None, timeout_s=600):
        return {**RESULT, "ok": False, "exit_code": 1, "dir": str(root)}

    monkeypatch.setattr(Simulator, "workbench", fake)
    (tmp_path / "p.py").write_text("x")
    assert main(["workbench", "--root", str(root), str(tmp_path / "p.py")]) == 1


def test_cli_equivalence_passes_its_thresholds_through(root, monkeypatch, capsys):
    """They are provisional and meant to be re-set from a measured noise floor,
    so they have to be reachable without editing the module."""
    from simulator.cli import main

    seen = {}

    async def fake(self, **kw):
        seen.update(kw)
        return {"ok": True, "regressed": False, "why": "", "cost_usd": 0.12,
                "summary": "2016 positions", "reference_path": "/results/r.json"}

    monkeypatch.setattr(Simulator, "equivalence", fake)
    assert main(["equivalence", "--root", str(root), "--min-agreement", "0.99",
                 "--max-mean-dlogprob", "0.01"]) == 0
    assert seen["min_agreement"] == 0.99 and seen["max_mean_dlogprob"] == 0.01
    assert "equivalent" in capsys.readouterr().out


def test_cli_equivalence_exits_nonzero_on_a_regression(root, monkeypatch):
    from simulator.cli import main

    async def fake(self, **kw):
        return {"ok": True, "regressed": True, "why": "top-1 agreement 0.4",
                "cost_usd": 0.12, "summary": "s", "reference_path": "/r.json"}

    monkeypatch.setattr(Simulator, "equivalence", fake)
    assert main(["equivalence", "--root", str(root)]) == 1


def test_results_path_guard_resolves_the_mount_too(tmp_path):
    """/results is a symlink inside the container; the guard must resolve
    both sides or it rejects every real file (it did)."""
    from simulator.runner.modal_runner import _inside

    real = tmp_path / "vol"
    (real / "equivalence").mkdir(parents=True)
    (real / "equivalence" / "ref.json").write_text("{}")
    link = tmp_path / "results"
    link.symlink_to(real)
    assert _inside(str(link), str(link / "equivalence" / "ref.json"))
    assert not _inside(str(link), str(link / ".." / "vol" / ".." / "elsewhere"))
    assert not _inside(str(link), str(tmp_path / "other.json"))
