"""Cost attribution must recover true per-token costs, and say when it can't."""
import pytest

from autoinf.pricing import (Attribution, COST_BASES, Observation, attribute,
                             conditioning, effective_in, prices, rank_vs_market)

# Ground truth: an uncached input token costs 10x a cached one, and an output
# token costs 30x. Deliberately the shape we expect on a real stack.
A_UNCACHED, A_CACHED, A_OUT = 2.0e-5, 0.2e-5, 60.0e-5


def _obs(name, uin, cin, out, noise=0.0):
    g = uin * A_UNCACHED + cin * A_CACHED + out * A_OUT
    return Observation(name, uin, cin, out, g * (1 + noise))


def _spanning():
    """Mixes that differ the way the suite's workloads differ."""
    return [
        _obs("prefill_heavy", 500_000, 20_000, 5_000),     # input dominated
        _obs("decode_heavy", 20_000, 2_000, 350_000),      # output dominated
        _obs("prefix_heavy", 60_000, 500_000, 60_000),     # cache dominated
        _obs("short_chat", 40_000, 5_000, 60_000),         # balanced
        _obs("sustained", 200_000, 30_000, 90_000),
    ]


def test_recovers_known_coefficients():
    a = attribute(_spanning())
    assert a.per_uncached_in == pytest.approx(A_UNCACHED, rel=0.05), a
    assert a.per_cached_in == pytest.approx(A_CACHED, rel=0.15), a
    assert a.per_out == pytest.approx(A_OUT, rel=0.05), a
    assert a.r2 > 0.999


def test_recovers_the_cache_discount():
    """The ratio is the number the business case rests on."""
    a = attribute(_spanning())
    assert a.cache_discount == pytest.approx(A_CACHED / A_UNCACHED, rel=0.2)
    assert a.cache_discount < 0.2


def test_tolerates_measurement_noise():
    import random
    rng = random.Random(0)
    noisy = [_obs(o.name, o.uncached_in, o.cached_in, o.out,
                  noise=rng.uniform(-0.03, 0.03)) for o in _spanning()]
    a = attribute(noisy)
    assert a.per_out == pytest.approx(A_OUT, rel=0.15)
    assert a.r2 > 0.99


def test_coefficients_are_never_negative():
    """A token cannot give GPU-seconds back; an unconstrained fit would allow it."""
    a = attribute(_spanning())
    assert a.per_uncached_in >= 0 and a.per_cached_in >= 0 and a.per_out >= 0


def test_refuses_too_few_workloads():
    with pytest.raises(ValueError, match="differing token mixes"):
        attribute(_spanning()[:2])


def test_conditioning_flags_a_degenerate_design():
    """Near-identical workloads give a confident fit that means nothing."""
    good = conditioning(_spanning())
    assert good["well_conditioned"], good

    same = [_obs(f"w{i}", 100_000 * (1 + i * 0.01), 10_000 * (1 + i * 0.01),
                 50_000 * (1 + i * 0.01)) for i in range(5)]
    bad = conditioning(same)
    assert not bad["well_conditioned"], bad
    assert bad["condition_number"] > good["condition_number"]


def test_price_scales_inversely_with_utilisation():
    a = attribute(_spanning())
    p50 = prices(a, utilization=0.5)
    p100 = prices(a, utilization=1.0)
    assert p50["price_out_per_mtok"] == pytest.approx(
        2 * p100["price_out_per_mtok"], rel=1e-6)


def test_modal_is_not_the_default_basis():
    """Serverless retail is our spend, not a serving cost basis."""
    from autoinf.pricing import DEFAULT_BASIS
    assert "modal" not in DEFAULT_BASIS
    assert COST_BASES[DEFAULT_BASIS][0] < COST_BASES["modal-h100"][0]


def test_effective_price_is_dominated_by_cache_at_high_hit_rate():
    # Cheap cached, expensive uncached.
    assert effective_in(0.40, 0.04, 0.95) == pytest.approx(0.058)
    # At a low hit rate the raw input price dominates instead.
    assert effective_in(0.40, 0.04, 0.10) == pytest.approx(0.364)


MARKET = [("Reka", 0.250, 3.000, 0.025), ("Chutes", 0.350, 2.750, 0.035),
          ("AkashML", 0.350, 2.550, 0.050), ("Phala", 0.400, 3.000, 0.150),
          ("Io Net", 0.480, 3.400, 0.250)]


def test_ranking_against_the_live_table():
    top = rank_vs_market(0.03, MARKET, hit_rate=0.9)
    assert top["rank"] == 1
    mid = rank_vs_market(0.10, MARKET, hit_rate=0.9)
    assert 2 <= mid["rank"] <= 5
    last = rank_vs_market(9.99, MARKET, hit_rate=0.9)
    assert last["rank"] == len(MARKET) + 1


def test_ranking_shifts_with_hit_rate():
    """Cache-read price reorders the table; raw input alone does not."""
    lo = rank_vs_market(0.20, MARKET, hit_rate=0.1)["rank"]
    hi = rank_vs_market(0.20, MARKET, hit_rate=0.95)["rank"]
    assert hi > lo, (lo, hi)      # same price ranks worse when caching matters
