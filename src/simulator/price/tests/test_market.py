"""Ground truth: OpenRouter publishes both listed and *realised* prices.

That gives us something we could not construct ourselves -- an external check on
the effective-price formula, and a real distribution of achievable cache hit
rates. Both were assumptions until this data existed.
"""
import pytest

from simulator.price.direct import effective_in
from simulator.price.market import (
    MARKET_BEST_EFF_IN,
    MARKET_INPUT_OUTPUT_RATIO,
    MARKET_QWEN38_27B,
    MARKET_REALISED,
    MARKET_SNAPSHOTS,
    MARKET_WEIGHTED_IN,
)

_LISTED = {n: (i, o, c) for n, i, o, c in MARKET_QWEN38_27B}


def test_effective_price_formula_matches_published_values():
    """eff = h*cache_read + (1-h)*listed must reproduce their numbers.

    This is the one step of the pricing chain we can validate against reality
    rather than assume.
    """
    errs = []
    for name, eff_pub, hit, _share in MARKET_REALISED:
        listed_in, _out, cache_read = _LISTED[name]
        pred = listed_in if cache_read is None else effective_in(
            listed_in, cache_read, hit)
        errs.append((name, abs(pred - eff_pub) / eff_pub))
    worst = max(e for _, e in errs)
    assert worst < 0.05, sorted(errs, key=lambda x: -x[1])[:3]
    # Most should be near-exact, not merely within tolerance.
    near = sum(1 for _, e in errs if e < 0.005)
    assert near >= 9, [(n, round(e, 4)) for n, e in errs]


def test_realised_hit_rates_stay_below_our_synthetic_assumption():
    """Our workloads hit 96%. The best real provider reaches 87%.

    The ceiling moves: on 2026-08-29 the best was Novita at 81.8%, and two days
    later the same provider realised 87.4%, which broke an earlier version of
    this test that hard-coded 85%. So assert the *relationship* that matters --
    no provider reaches the 95.6% our replayed traces achieve -- rather than a
    level that drifts. Sizing a business case on 95% would overstate the cache
    discount's benefit.
    """
    for date, snap in MARKET_SNAPSHOTS.items():
        hits = [h for _n, _e, h, _s in snap]
        assert max(hits) < 0.90, (date, max(hits))
        assert max(hits) > 0.75, (date, max(hits))   # the best do achieve a lot


def test_the_cheapest_provider_changes_between_snapshots():
    """The target moves, and not because anyone repriced.

    Listed prices were identical across both snapshots; the leader changed from
    Chutes to Novita purely because their cache hit rates moved. Any
    "we rank Nth" claim is against a moving target and needs its date.
    """
    best = [min(s, key=lambda r: r[1])[0] for s in MARKET_SNAPSHOTS.values()]
    assert len(set(best)) > 1, best


def test_hit_rate_is_provider_controlled_not_workload_luck():
    """Same model, same marketplace traffic, hit rates from 0% to 82%.

    The strongest available evidence that realised cache hit rate is a serving
    -system property -- which is what makes it an optimisation target rather
    than an input.
    """
    by = {n: h for n, _e, h, _s in MARKET_REALISED}
    assert by["Venice"] == 0.0 and by["Cloudflare"] == 0.0
    assert by["Novita"] > 0.8
    # And it dominates effective price: zero-cache providers are the worst.
    eff = {n: e for n, e, _h, _s in MARKET_REALISED}
    assert eff["Venice"] > eff["Novita"] * 2.5


def test_weighted_average_is_consistent_with_the_table():
    tot = sum(s for *_x, s in MARKET_REALISED)
    wavg = sum(e * s for _n, e, _h, s in MARKET_REALISED) / tot
    assert abs(wavg - MARKET_WEIGHTED_IN) / MARKET_WEIGHTED_IN < 0.10, wavg


def test_target_is_the_best_realised_price():
    best = min(e for _n, e, _h, _s in MARKET_REALISED)
    assert best == pytest.approx(MARKET_BEST_EFF_IN)


def test_traffic_is_input_dominated():
    """18:1 input:output, against ~2:1 in our synthetic suite."""
    assert MARKET_INPUT_OUTPUT_RATIO > 10
    # At market prices, input is the majority of revenue at this ratio.
    in_rev = MARKET_INPUT_OUTPUT_RATIO * MARKET_WEIGHTED_IN
    out_rev = 1 * 2.866
    assert in_rev > out_rev, (in_rev, out_rev)


def test_burst_utilisation_is_derived_not_assumed():
    """Utilisation for a single-model deployment comes from the traffic.

    Sizing for peak means running at mean/peak. On this model's 17-day
    OpenRouter series that is ~48%, which replaces the 60% that earlier
    price tables simply assumed.
    """
    import json

    from simulator.price.market import (
        _find_snapshot,
        burst_utilisation,
        fleet_utilisation,
    )
    v = [r["total_prompt_tokens"]
         for r in json.loads(_find_snapshot().read_text())["daily"]]
    b = burst_utilisation(v)
    assert b["available"]
    assert 0.40 < b["single_model_utilisation"] < 0.60, b
    assert b["peak_over_mean"] > 1.8

    # A fleet beats it purely through uncorrelated peaks, not through serving
    # more of this model.
    assert fleet_utilisation(b["cv"], 1) < 0.60
    assert fleet_utilisation(b["cv"], 100) > 0.85


def test_pricing_defaults_are_the_agreed_basis():
    """A hardcoded copy of the basis kept reporting the superseded numbers.

    Two now-deleted modules (a `launch.py` entrypoint and a `modal_app.price`
    helper) both restated $2.50/60%/25% inline, so changing the default to the
    agreed $3.00/50%/break-even silently did nothing to what runs printed. The
    scan below is what keeps that from coming back: call sites must inherit the
    basis, not restate it.
    """
    import pathlib

    from simulator.costs import rate
    from simulator.price.direct import DEFAULT_MARGIN, DEFAULT_UTILISATION

    assert rate() == 3.00
    assert DEFAULT_UTILISATION == 0.50
    assert DEFAULT_MARGIN == 0.0          # break-even; margin stated separately

    # No module outside `price.direct` may restate the basis, which is what the
    # two deleted modules above did.
    import simulator

    root = pathlib.Path(simulator.__file__).parent
    for f in sorted(root.rglob("*.py")):
        # `costs.py` owns the rates and `direct.py` owns the default; tests are
        # allowed to name a literal because asserting on one is the point.
        if f.name in ("direct.py", "costs.py") or "tests" in f.parts \
                or f.name.startswith("test_"):
            continue
        src = f.read_text()
        for banned in ('basis="nebius', "utilization=0.6", "margin=0.25",
                       "2.50", "0.60,"):
            assert banned not in src, f"{f} restates the cost basis: {banned}"


def test_price_direct_needs_no_decomposition():
    """Hit rate is an outcome, so the cached/uncached split is unnecessary.

    Splitting input cost exists only to re-blend at a competitor's hit rate.
    At our own, effective input price is input GPU-seconds over input tokens --
    and caching better simply lowers it, which is the point.
    """
    from simulator.price.direct import price_direct

    # Same work, same tokens, but the second system caches better: fewer of the
    # input tokens needed prefill, so it spent less GPU time on them.
    poor = price_direct(gpu_seconds_input=100.0, gpu_seconds_output=400.0,
                        input_tokens=20_583_000, output_tokens=2_076_000,
                        cached_tokens=0.30 * 20_583_000)
    good = price_direct(gpu_seconds_input=40.0, gpu_seconds_output=400.0,
                        input_tokens=20_583_000, output_tokens=2_076_000,
                        cached_tokens=0.85 * 20_583_000)
    assert good.effective_in_per_m < poor.effective_in_per_m
    assert good.hit_rate > poor.hit_rate
    assert good.out_per_m == poor.out_per_m        # output untouched by caching
    # break-even by default: no margin folded into a cost figure
    assert good.margin == 0.0 and good.utilization == 0.50


def test_price_direct_rejects_impossible_inputs():
    import pytest

    from simulator.price.direct import price_direct

    with pytest.raises(ValueError):
        price_direct(1.0, 1.0, 0, 100, 0)                  # no input tokens
    with pytest.raises(ValueError):
        price_direct(1.0, 1.0, 100, 100, 0, utilization=0)  # divide by zero


def test_counters_keep_the_category_label():
    """Summing across labels discarded the prefill/decode split.

    `forward_execution_seconds_total` carries a `category` label. Folding it
    into one number threw away exactly the phase breakdown that makes direct
    pricing possible -- and the aggregate still looked perfectly sensible, so
    nothing flagged it.
    """
    from simulator.measure.server import Snapshot

    snap = Snapshot.parse(
        'sglang:forward_execution_seconds_total{m="x",category="prefill"} 41.5\n'
        'sglang:forward_execution_seconds_total{m="x",category="decode"} 61.6\n')
    c = snap.counters
    assert c["sglang:forward_execution_seconds_total"] == pytest.approx(103.1)
    assert c["sglang:forward_execution_seconds_total[prefill]"] == pytest.approx(41.5)
    assert c["sglang:forward_execution_seconds_total[decode]"] == pytest.approx(61.6)


def test_forward_time_is_already_gpu_seconds(tmp_path):
    """The counter is summed across TP ranks; multiplying by n_gpu doubles it.

    Caught because the affine model predicted $5.53/M output while the direct
    measurement said $10.97/M -- exactly 2x on a 2-GPU run. Physically forward
    time can never exceed wall x n_gpu, and the measured ratio was 1.00, not the
    0.50 that per-rank timing would give.

    Asserted on behaviour rather than by grepping source: a level on 2 GPUs
    must price the same as the identical level on 1 GPU, because the counter
    already carries the node total.
    """
    from simulator import Simulator

    def rec(n_gpu):
        return {"status": "ok", "serving": {"n_gpu": n_gpu},
                "levels": [{"n_users": 8, "wall_s": 200.0, "goodput_rps": 1.0,
                            "good_frac": 1.0, "n_failed": 0,
                            "ttft_ms": {"n": 400, "p90": 500.0},
                            "tpot_ms": {"n": 400, "p90": 10.0, "mean": 9.0},
                            "prompt_tokens": 1e6, "cached_tokens": 5e5,
                            "output_tokens": 1e5, "cache_hit_rate": 0.5,
                            "batch": {"running": {"mean": 8.0}},
                            "server_counters": {
                                "sglang:forward_execution_seconds_total[extend]": 20.0,
                                "sglang:forward_execution_seconds_total[decode]": 180.0,
                            }}]}

    d = tmp_path / "r"
    d.mkdir()
    one = Simulator(root_dir=d, n_gpu=1, levels=(8,)).analyse(rec(1))
    two = Simulator(root_dir=d, n_gpu=2, levels=(8,)).analyse(rec(2))
    assert one.ok and two.ok
    assert one.out_per_m == two.out_per_m
    assert one.effective_in_per_m == two.effective_in_per_m


def test_bounds_at_different_quantiles_catch_what_one_p99_cannot():
    """A single p99 threshold reports "usually slow, rarely terrible" as passing.

    This replaces an earlier two-tier SLO that still forced TTFT and TPOT onto
    the same percentile. The market pins a TTFT tail and only a TPOT middle, so
    the bounds have to be independent.
    """
    from simulator.slo import SLO

    loose = SLO.parse("ttft:p99:1000,tpot:p99:45")
    real = SLO.parse("ttft:p90:400,tpot:p90:25,tpot:p99:45")
    level = {"ttft_ms": {"n": 500, "p90": 380, "p99": 900},
             "tpot_ms": {"n": 500, "p90": 32, "p99": 41},
             "good_frac": 1.0, "n_failed": 0}
    assert loose.judge(level).ok, "p99 alone should accept it"
    v = real.judge(level)
    assert not v.ok and v.binding == "p90 TPOT"


def test_default_environment_is_one_h100_on_the_target_model():
    """The agreed experiment environment, pinned so it cannot drift.

    A100 80GB is cheaper per hour but has no FP8 (SM80), and L40S is cheaper
    still while costing ~1.9x per token, because decode is bandwidth-bound.
    """
    from simulator.config import ServingConfig
    from simulator.costs import rate
    from simulator.price.direct import DEFAULT_UTILISATION
    from simulator.specs import HARDWARE

    sc = ServingConfig()
    assert sc.model == "Qwen/Qwen3.8-27B-FP8"
    assert (sc.gpu, sc.n_gpu) == ("H100", 1)
    assert not sc.validate(), sc.validate()      # tp=1 needs no --ep
    assert rate() == 3.00
    assert DEFAULT_UTILISATION == 0.50
    # the model's FP8 weights must fit with room for a usable KV pool
    from simulator.specs import MODELS
    m = MODELS[sc.model]
    kv = HARDWARE["H100"].hbm_bytes * 0.85 - m.active_params
    assert kv / m.bytes_per_seq(20_583) > 20, "too few conversations fit"


def test_rank_vs_market_scores_everyone_at_one_hit_rate():
    """The matched comparison, kept as a diagnostic (§5e). Every provider is
    re-blended to OUR hit rate, which is exactly what makes it a diagnostic and
    not the headline: it isolates cost per token from cache achievement."""
    from simulator.price.market import rank_vs_market

    board = [("cheap", 0.40, 3.0, 0.04), ("dear", 0.50, 3.0, 0.10),
             ("nocache", 0.45, 3.0, None)]
    got = rank_vs_market(0.10, board, hit_rate=0.8)
    # 0.8*0.04 + 0.2*0.40 = 0.112; 0.8*0.10 + 0.2*0.50 = 0.180; nocache stays 0.45.
    assert got["best_competitor"] == "cheap"
    assert got["best_competitor_price"] == 0.112
    assert got["rank"] == 1 and got["of"] == 4
    assert [r["provider"] for r in got["table"]] == ["cheap", "dear", "nocache"]


def test_rank_counts_only_providers_strictly_cheaper():
    from simulator.price.market import rank_vs_market

    board = [("a", 0.40, 3.0, None), ("b", 0.50, 3.0, None)]
    assert rank_vs_market(0.45, board, hit_rate=0.0)["rank"] == 2
    assert rank_vs_market(0.60, board, hit_rate=0.0)["rank"] == 3


def test_blended_per_m_is_the_bill_expressed_per_token():
    """Same money, different denominator -- so it must agree with `bill_per_1k`
    exactly, or one of the two is quoting a different request."""
    from simulator.price.market import Market

    m = Market.load()
    eff_in, out = 0.0126, 6.45
    per_m = m.blended_per_m(eff_in, out)
    tokens = m.in_per_request + m.out_per_request
    assert per_m * tokens / 1e6 * 1000 == pytest.approx(m.bill_per_1k(eff_in, out))
    # It sits between the two prices it blends, weighted 9.9:1 toward input.
    assert eff_in < per_m < out


def test_direct_price_bills_a_request_of_any_shape():
    """`Market.bill_per_1k` is the market-sized request; this is the same
    arithmetic for an arbitrary one, which is what a per-level check needs."""
    from simulator.price.direct import price_direct

    p = price_direct(gpu_seconds_input=100.0, gpu_seconds_output=50.0,
                     input_tokens=1e6, output_tokens=1e5, cached_tokens=4e5,
                     usd_per_gpu_hour=3.00, utilization=0.5)
    assert p.hit_rate == pytest.approx(0.4)
    got = p.bill_per_request(20_583, 2_076)
    want = (p.effective_in_per_m * 20_583 + p.out_per_m * 2_076) / 1e6
    assert got == pytest.approx(want)
    assert p.bill_per_request(0, 0) == 0.0
