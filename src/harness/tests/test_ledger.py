"""Tool spend reaches the fleet, and killed fleets cancel their GPU calls."""
import json

from harness.agent import ledger
from harness.inflight import cancel_pending, pending_call_ids


def test_ledger_drains_once_and_survives_a_restart(tmp_path):
    ledger.append(tmp_path, "gpu-run", 0.9, elapsed_s=400, gpu="H100")
    ledger.append(tmp_path, "ncu", 0.6)
    assert ledger.drain(tmp_path) == 1.5
    assert ledger.drain(tmp_path) == 0.0            # nothing new
    ledger.append(tmp_path, "equivalence", 0.4)
    assert ledger.drain(tmp_path) == 0.4            # only the new line
    # a fresh loop over the same directory starts from the recorded offset
    assert (tmp_path / ledger.SEEN).read_text().strip() == str((tmp_path / ledger.LEDGER).stat().st_size)


def test_missing_or_garbled_ledger_is_zero(tmp_path):
    assert ledger.drain(tmp_path) == 0.0
    (tmp_path / ledger.LEDGER).write_text("not json\n" + json.dumps({"cost_usd": 0.25}) + "\n")
    assert ledger.drain(tmp_path) == 0.25


def test_pending_calls_are_the_ones_without_a_result(tmp_path):
    def call(rel, cid, done=False, cancelled=False):
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "call_id").write_text(cid + "\n")
        if done:
            (d / "result.json").write_text("{}")
        if cancelled:
            (d / "cancelled").write_text("x")
    call("a00/runs/attempt-000", "fc-1", done=True)
    call("a00/runs/attempt-001", "fc-2")
    call("a00/workbench-3", "fc-3")
    call("a01/workbench-0", "fc-4", cancelled=True)
    assert [c for c, _ in pending_call_ids(tmp_path)] == ["fc-2", "fc-3"]
    seen = []
    done = cancel_pending(tmp_path, cancel=lambda cid: seen.append(cid) or (cid == "fc-3" and (_ for _ in ()).throw(RuntimeError("gone"))))
    assert seen == ["fc-2", "fc-3"]
    assert done == ["fc-2"]                          # the failed cancel is left for next time
    assert (tmp_path / "a00/runs/attempt-001/cancelled").is_file()
    assert not (tmp_path / "a00/workbench-3/cancelled").is_file()
    assert [c for c, _ in pending_call_ids(tmp_path)] == ["fc-3"]


def test_a_sweep_is_billed_for_the_whole_call_not_just_load_and_levels():
    from harness.agent.evaluator import sweep_cost, sweep_seconds

    rec = {"serving": {"gpu": "H100", "n_gpu": 1}, "model_load_s": 300.0,
           "levels": [{"wall_s": 63.0}, {"wall_s": 63.0}],
           "started_at": 1000.0, "finished_at": 1700.0}
    assert sweep_seconds(rec) == 700.0                 # not 426
    assert sweep_seconds({"model_load_s": 10.0, "levels": [{"wall_s": 5.0}]}) == 15.0
    assert sweep_cost(rec) > sweep_cost({**rec, "finished_at": None})
    assert sweep_cost({}) == 0.0
