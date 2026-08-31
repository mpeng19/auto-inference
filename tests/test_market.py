"""Ground truth: OpenRouter publishes both listed and *realised* prices.

That gives us something we could not construct ourselves -- an external check on
the effective-price formula, and a real distribution of achievable cache hit
rates. Both were assumptions until this data existed.
"""
import pytest

from autoinf.modal_app import (MARKET_BEST_EFF_IN, MARKET_INPUT_OUTPUT_RATIO,
                               MARKET_SNAPSHOTS,
                               MARKET_QWEN38_27B, MARKET_REALISED,
                               MARKET_WEIGHTED_IN)
from autoinf.pricing import effective_in

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
