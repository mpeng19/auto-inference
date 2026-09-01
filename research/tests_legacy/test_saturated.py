"""Rate-form attribution: the formulation that finally identifies token costs.

Three earlier attempts failed for what looked like three different reasons
(r2 -2.9, then all-zero coefficients, then r2 -36.7). The cause was the same
each time and was in the experiment, not the arithmetic: every mix ran for the
same fixed duration, so GPU-seconds was constant by construction while token
counts varied 35x. No linear model fits a constant target from varying features.

At saturation the information is in throughput composition, and dividing the
cost equation by GPU-seconds cancels duration entirely.
"""
import pytest

from autoinf.pricing import Observation, attribute_saturated, conditioning

A_UNCACHED, A_CACHED, A_OUT = 5.0e-5, 1.0e-5, 2.0e-4


def _mix(name, u_rate, c_rate, o_rate, seconds=100.0):
    """A saturated mix, specified by its per-GPU-second token composition."""
    return Observation(name, u_rate * seconds, c_rate * seconds,
                       o_rate * seconds, seconds)


def _spanning(seconds=100.0):
    """Compositions that span the space, as the real mixes do."""
    return [
        _mix("prefill_heavy", 14000, 3200, 1000, seconds),
        _mix("decode_heavy", 360, 4100, 4700, seconds),
        _mix("cache_heavy", 1960, 17000, 3700, seconds),
        _mix("balanced", 1470, 12500, 3700, seconds),
    ]


def _consistent(rates, seconds=100.0):
    """Build mixes whose rates actually satisfy 1 = a*u + b*c + d*o."""
    out = []
    for name, u, c, o in rates:
        # Scale the composition so it consumes exactly one GPU-second per second.
        k = 1.0 / (A_UNCACHED * u + A_CACHED * c + A_OUT * o)
        out.append(_mix(name, u * k, c * k, o * k, seconds))
    return out


def test_recovers_known_costs_from_rates():
    obs = _consistent([("prefill", 14000, 3200, 1000),
                       ("decode", 360, 4100, 4700),
                       ("cache", 1960, 17000, 3700),
                       ("balanced", 1470, 12500, 3700)])
    a = attribute_saturated(obs)
    assert a.per_uncached_in == pytest.approx(A_UNCACHED, rel=0.1), a
    assert a.per_cached_in == pytest.approx(A_CACHED, rel=0.25), a
    assert a.per_out == pytest.approx(A_OUT, rel=0.1), a
    assert a.r2 > 0.95, a


def test_result_is_independent_of_run_duration():
    """The bug that broke three attempts: duration must cancel out."""
    rates = [("prefill", 14000, 3200, 1000), ("decode", 360, 4100, 4700),
             ("cache", 1960, 17000, 3700), ("balanced", 1470, 12500, 3700)]
    short = attribute_saturated(_consistent(rates, seconds=30.0))
    long_ = attribute_saturated(_consistent(rates, seconds=600.0))
    assert short.per_uncached_in == pytest.approx(long_.per_uncached_in, rel=1e-6)
    assert short.per_cached_in == pytest.approx(long_.per_cached_in, rel=1e-6)
    assert short.per_out == pytest.approx(long_.per_out, rel=1e-6)


def test_recovers_the_cache_discount():
    obs = _consistent([("prefill", 14000, 3200, 1000),
                       ("decode", 360, 4100, 4700),
                       ("cache", 1960, 17000, 3700),
                       ("balanced", 1470, 12500, 3700)])
    a = attribute_saturated(obs)
    assert a.cache_discount == pytest.approx(A_CACHED / A_UNCACHED, rel=0.3)
    assert a.cache_discount < 0.4, "a cached token must be markedly cheaper"


def test_real_measured_mixes_fit_well():
    """The four saturated mixes actually observed, run 1788147598."""
    obs = [
        Observation("prefill_heavy", 1486961, 343173, 106909, 106.2),
        Observation("decode_heavy", 42132, 481798, 552887, 116.6),
        Observation("cache_heavy", 203440, 1765932, 381203, 103.4),
        Observation("balanced", 157425, 1343541, 399906, 107.3),
    ]
    a = attribute_saturated(obs)
    assert a.r2 > 0.9, a                       # worst residual under 10%
    assert conditioning(obs)["well_conditioned"]
    # Cached tokens must come out substantially cheaper than uncached.
    assert 0.05 < a.cache_discount < 0.5, a.cache_discount
    # And the costs must be physically ordered: output > uncached > cached.
    assert a.per_out > a.per_uncached_in > a.per_cached_in, a


def test_refuses_too_few_mixes():
    with pytest.raises(ValueError, match="saturated mixes"):
        attribute_saturated(_spanning()[:2])


def test_coefficients_stay_non_negative():
    a = attribute_saturated(_spanning())
    assert a.per_uncached_in >= 0 and a.per_cached_in >= 0 and a.per_out >= 0
