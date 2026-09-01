"""End to end on a real stored sweep, with no GPU."""
import json

import pytest

from simulator import Simulator
from simulator.slo import SLO


def sim(root, **kw):
    kw.setdefault("levels", (4, 8, 12, 16, 24))
    return Simulator(root_dir=root, n_gpu=1, **kw)


def test_reproduces_the_handoff_numbers(root, sweep):
    """HANDOFF 6c/6d, straight through the public API."""
    res = sim(root).analyse(sweep)
    assert res.ok and res.n_star == 12
    assert res.effective_in_per_m == pytest.approx(0.0294, abs=5e-4)
    assert res.out_per_m == pytest.approx(5.6021, abs=5e-3)
    assert res.bill_per_1k == pytest.approx(12.23, abs=0.02)
    assert res.share_per_node == pytest.approx(0.0042, abs=2e-4)
    assert res.best.batch == pytest.approx(5.0, abs=0.1)
    assert res.best.hit_rate == pytest.approx(0.748, abs=1e-3)


def test_ranks_both_ways(root, sweep):
    r = sim(root).analyse(sweep).rank()
    assert (r["rank_eff_in"], r["rank_bill"], r["of"]) == (1, 9, 12)


def test_root_dir_must_already_exist(tmp_path):
    with pytest.raises(NotADirectoryError):
        Simulator(root_dir=tmp_path / "nope")


def test_every_artifact_lands_in_root(root, sweep):
    res = sim(root).finish(sweep)
    for name in ("sweep", "result", "config", "report",
                 "slo_frontier", "price_vs_share"):
        assert name in res.artifacts, name
        assert (root / res.artifacts[name].rsplit("/", 1)[-1]).exists()
    assert "N* = 12" in (root / "report.txt").read_text()


def test_result_json_is_self_describing(root, sweep):
    sim(root).finish(sweep)
    d = json.loads((root / "result.json").read_text())
    assert d["n_star"] == 12 and len(d["curve"]) == 5
    assert d["rank"]["rank_bill"] == 9


def test_rescoring_at_a_different_slo_costs_no_gpu(root, sweep):
    """The whole reason every percentile is stored."""
    tight = sim(root, slo=SLO.parse("ttft:p99:1000,tpot:p90:25,tpot:mean:20"))
    res = tight.analyse(sweep)
    assert res.n_star == 8
    assert res.bill_per_1k == pytest.approx(15.06, abs=0.03)


def test_no_price_when_no_level_holds(root, sweep):
    hard = sim(root, slo=SLO.parse("tpot:mean:1"))
    res = hard.analyse(sweep)
    assert not res.ok and "no level met the SLO" in res.reason
    assert res.curve, "the curve is still reported so the caller can see why"


def test_no_price_when_the_device_timer_was_off(root, sweep):
    for lv in sweep["levels"]:
        lv["server_counters"] = {}
    res = sim(root).analyse(sweep)
    assert not res.ok and "DEVICE_TIMER" in res.reason


def test_failed_sweep_propagates_its_reason(root):
    res = sim(root).analyse({"status": "failed", "failure": "boom"})
    assert not res.ok and res.reason == "boom"


def test_flags_a_sweep_that_never_found_its_frontier(root, sweep):
    """If every level passes, N* is the top of the sweep and the price is an
    upper bound -- the 2xH100 run had exactly this problem."""
    loose = sim(root, slo=SLO(bounds=SLO.parse("tpot:mean:9999").bounds,
                              min_good_frac=0.0))
    res = loose.analyse(sweep)
    assert res.ok and res.n_star == 24
    assert "every level passed" in res.reason


def test_digest_tracks_what_would_change_the_answer(root, sweep):
    a = sim(root)
    assert a.digest() == sim(root).digest()
    assert a.digest() != sim(root, levels=(4, 8)).digest()
    assert a.digest() != sim(root, slo=SLO.parse("tpot:mean:19")).digest()


def test_utilisation_is_reported_not_hidden(root, sweep):
    """It is the largest lever and cannot be measured, so it must be visible."""
    sim(root, utilisation=0.25).finish(sweep)
    cfg = json.loads((root / "config.json").read_text())
    assert cfg["utilisation"] == 0.25
    assert "utilisation 25%" in (root / "report.txt").read_text()


def test_good_frac_is_a_runtime_gate_not_a_rescoring_one(root, sweep):
    """It was computed against the SLO the sweep ran with and cannot be
    recovered offline, so a rescore that relaxes the bounds can still be held
    back by it. Documented in SLO.judge; asserted here so it stays known."""
    loose = sim(root, slo=SLO.parse("tpot:mean:9999"))
    assert loose.analyse(sweep).n_star == 16      # N=24 held back by good_frac
    free = sim(root, slo=SLO(bounds=SLO.parse("tpot:mean:9999").bounds,
                             min_good_frac=0.0))
    assert free.analyse(sweep).n_star == 24
