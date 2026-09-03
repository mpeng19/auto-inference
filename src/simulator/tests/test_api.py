"""End to end on a real stored sweep, with no GPU."""
import json

import pytest

from simulator import Simulator
from simulator.slo import SLO


def sim(root, **kw):
    kw.setdefault("levels", (4, 8, 12, 16, 24))
    return Simulator(root_dir=root, n_gpu=1, **kw)


def test_reproduces_the_handoff_numbers(root, sweep):
    """The 1xH100 baseline, straight through the public API.

    These are the numbers in `docs/examples/baseline-1xh100/report.txt`, which
    is generated from this same fixture -- so the example and the test cannot
    drift apart without one of them failing.
    """
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
                 "price_vs_share", "price_vs_demand"):
        assert name in res.artifacts, name
        assert (root / res.artifacts[name].rsplit("/", 1)[-1]).exists()
    # One SLO figure per bound, each its own file: three panels crammed into
    # one image are three plots nobody can read.
    slo_figs = [k for k in res.artifacts if k.startswith("slo_")]
    assert len(slo_figs) == len(sim(root).slo.bounds)
    for k in slo_figs:
        assert (root / res.artifacts[k].rsplit("/", 1)[-1]).exists()
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
    assert cfg["assumptions"]["utilisation"] == 0.25
    assert "utilisation 25%" in (root / "report.txt").read_text()


def test_every_assumption_is_recorded(root, sweep):
    """None of these can be validated from inside the harness, and all of them
    scale the answer, so a result that does not carry them is unreadable."""
    sim(root).finish(sweep)
    a = json.loads((root / "config.json").read_text())["assumptions"]
    for k in ("rate_per_gpu_hour", "gpu_provider", "cost_basis", "utilisation",
              "margin", "market_as_of", "market_requests_per_day",
              "in_per_request", "out_per_request"):
        assert k in a, k


def test_rate_comes_from_the_catalog_when_a_provider_is_named(root):
    from simulator import Simulator
    assert Simulator(root_dir=root).rate_per_gpu_hour == 3.00
    s = Simulator(root_dir=root, gpu_provider="nebius-committed")
    assert s.rate_per_gpu_hour == 2.50
    assert "nebius-committed" in s.cost_basis


def test_serverless_retail_is_refused_as_a_serving_basis(root):
    from simulator import Simulator
    with pytest.raises(ValueError, match="retail"):
        Simulator(root_dir=root, gpu="L40S", gpu_provider="modal")
    ok = Simulator(root_dir=root, gpu="L40S", gpu_provider="modal",
                   allow_retail_rate=True)
    assert ok.rate_per_gpu_hour == 1.95


def test_good_frac_is_a_runtime_gate_not_a_rescoring_one(root, sweep):
    """It was computed against the SLO the sweep ran with and cannot be
    recovered offline, so a rescore that relaxes the bounds can still be held
    back by it. Documented in SLO.judge; asserted here so it stays known."""
    strict = sim(root, slo=SLO(bounds=SLO.parse("tpot:mean:9999").bounds,
                               min_good_frac=0.99))
    assert strict.analyse(sweep).n_star == 16     # N=24 held back by good_frac
    free = sim(root, slo=SLO(bounds=SLO.parse("tpot:mean:9999").bounds,
                             min_good_frac=0.0))
    assert free.analyse(sweep).n_star == 24


def test_profiling_is_off_by_default_and_plumbed_when_asked(root):
    """It perturbs what it measures, so it must never be on by accident."""
    from simulator import Simulator

    assert Simulator(root_dir=root).profile_level == 0
    s = Simulator(root_dir=root, profile_level=8, profile_steps=30)
    args = s._args()
    assert 8 in args and 30 in args, "profile settings must reach the runner"


def test_quality_is_measured_by_default(root):
    """An agent maximising goodput can serve worse answers faster, and nothing
    in the price model sees it."""
    from simulator import Simulator

    s = Simulator(root_dir=root)
    assert s.quality_suites == ("gsm8k", "longbench") and s.quality_n > 0
    assert tuple(s.quality_suites) in s._args()


def test_a_quality_regression_is_reported_beside_the_price(root, sweep):
    """Both are facts: a price was computed *and* accuracy fell. Collapsing
    them into one boolean loses the number a caller needs."""
    from simulator import Simulator

    sweep["quality"] = [{"suite": "gsm8k", "n": 50, "correct": 30,
                         "accuracy": 0.60, "baseline_accuracy": 0.72,
                         "delta_pct": -12.0, "errors": 0, "regressed": True,
                         "why": "accuracy fell 12.0 points"}]
    res = Simulator(root_dir=root, n_gpu=1, levels=(4, 8, 12, 16, 24)).analyse(sweep)
    assert res.ok, "the price is still computed and still reported"
    assert res.quality_regressed
    assert "accuracy fell" in res.quality_note
    assert "REGRESSION" in res.summary()


def test_clean_quality_does_not_flag(root, sweep):
    from simulator import Simulator

    sweep["quality"] = [{"suite": "gsm8k", "n": 50, "correct": 36,
                         "accuracy": 0.72, "baseline_accuracy": 0.72,
                         "delta_pct": 0.0, "errors": 0, "regressed": False,
                         "why": ""}]
    res = Simulator(root_dir=root, n_gpu=1, levels=(4, 8, 12, 16, 24)).analyse(sweep)
    assert res.ok and not res.quality_regressed
    assert "quality gsm8k: 72.0%" in res.summary()


def test_submit_records_the_call_id_on_both_paths(root, monkeypatch):
    """`eval()` awaits the sweep from an event loop, so it must use Modal's
    async spawn; `submit()` stays blocking for the CLI. Both leave `call_id`
    behind so a dead client can still `collect`."""
    import asyncio

    from simulator import Simulator

    class Call:
        object_id = "fc-123"

    class Spawn:
        def __call__(self, *a):
            return Call()

        async def aio(self, *a):
            return Call()

    class Fn:
        spawn = Spawn()

    monkeypatch.setattr(Simulator, "_fn", lambda self: Fn())
    s = Simulator(root_dir=root)
    assert s.submit() == "fc-123"
    (root / "call_id").unlink()
    assert asyncio.run(s.submit_async()) == "fc-123"
    assert (root / "call_id").read_text() == "fc-123"


def test_interpolated_frontier_sits_inside_the_grid_step(root, sweep):
    """N* is quantised to the grid; the crossing is not. On the stored
    baseline mean TPOT is 19.3 ms at N=12 and 22.0 at N=16, so the 20 ms line
    is crossed a quarter of the way to 16 and the bill there is a shade below
    the N=12 price."""
    res = sim(root).analyse(sweep)
    i = res.interpolated
    assert i is not None and i["binding"] == "mean TPOT"
    assert 12 < i["n_star"] < 14 and i["between"] == [12, 16]
    assert res.bill_per_1k > i["bill_per_1k"] > res.curve[3].bill_per_1k
    assert res.as_dict()["interpolated"] == i
    # a sweep that passes everywhere has no crossing to interpolate
    from simulator.slo import SLO
    loose = sim(root, slo=SLO.parse("tpot:mean:9999")).analyse(sweep)
    assert loose.interpolated is None


def test_the_app_name_override_reaches_the_client_side_too(monkeypatch):
    """`make deploy` names the app from SIMULATOR_APP_NAME, so every lookup has
    to read the same variable. It did not: the runner honoured the override and
    the client asked for "auto-inference" regardless, which on a fresh account
    deploying under another name is a deploy that works and a run that cannot
    find it."""
    import importlib

    import simulator.api as api
    import simulator.runner.modal_runner as runner

    monkeypatch.setenv("SIMULATOR_APP_NAME", "someone-elses-app")
    assert importlib.reload(api).APP_NAME == "someone-elses-app"
    assert importlib.reload(runner).APP_NAME == "someone-elses-app"

    monkeypatch.delenv("SIMULATOR_APP_NAME")
    assert importlib.reload(api).APP_NAME == "auto-inference"
    assert importlib.reload(runner).APP_NAME == "auto-inference"
