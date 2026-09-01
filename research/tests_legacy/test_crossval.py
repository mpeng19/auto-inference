"""Cross-validation must distinguish a correct model from a memorised one.

In-sample r2 cannot: a linear fit through four points with three free
parameters will look excellent whether or not the underlying system is
additive. Held-out prediction is the only internal check that tests the
*structure* rather than the fit, and it needs no knowledge of anyone's
utilisation, margin or hardware cost.
"""
import pytest

from autoinf.pricing import Observation, attribute, cross_validate

A_UNCACHED, A_CACHED, A_OUT = 2.0e-5, 0.2e-5, 60.0e-5


def _additive(name, uin, cin, out, noise=0.0):
    g = uin * A_UNCACHED + cin * A_CACHED + out * A_OUT
    return Observation(name, uin, cin, out, g * (1 + noise))


def _spanning(fn):
    return [
        fn("prefill_heavy", 500_000, 20_000, 5_000),
        fn("decode_heavy", 20_000, 2_000, 350_000),
        fn("cache_heavy", 60_000, 500_000, 60_000),
        fn("balanced", 150_000, 80_000, 90_000),
        fn("extra", 250_000, 40_000, 120_000),
    ]


def test_additive_system_passes_cross_validation():
    cv = cross_validate(_spanning(_additive))
    assert cv["available"]
    assert cv["structure_holds"], cv
    assert cv["worst_rel_error"] < 0.05, cv
    assert cv["ratio_stable"], cv


def test_needs_enough_workloads():
    cv = cross_validate(_spanning(_additive)[:3])
    assert not cv["available"]
    assert "hold one out" in cv["reason"]


def test_detects_a_non_additive_system():
    """If prefill and decode interact, held-out prediction should fail even
    though the in-sample fit still looks fine."""
    def interacting(name, uin, cin, out, noise=0.0):
        base = uin * A_UNCACHED + cin * A_CACHED + out * A_OUT
        # Contention: running prefill and decode together costs more than the
        # sum of the parts. Sized to add 3-40% depending on the mix, which is
        # the realistic magnitude for prefill stealing decode steps.
        base += 1e-9 * uin * out
        return Observation(name, uin, cin, out, base)

    obs = _spanning(interacting)
    in_sample = attribute(obs)
    cv = cross_validate(obs)
    # The in-sample fit still looks respectable...
    assert in_sample.r2 > 0.8, in_sample.r2
    # ...but held-out prediction exposes the missing term.
    assert not cv["structure_holds"], cv


def test_flags_an_unidentified_cache_ratio():
    """If one workload alone carries the cached-token signal, dropping it
    should swing the ratio -- and that instability must be visible."""
    obs = [
        _additive("a", 400_000, 1_000, 50_000),
        _additive("b", 380_000, 1_200, 60_000),
        _additive("c", 420_000, 900, 55_000),
        # Only this one has substantial cached tokens.
        _additive("cache_only", 50_000, 600_000, 40_000),
    ]
    cv = cross_validate(obs)
    assert cv["available"]
    assert not cv["ratio_stable"], cv


def test_noise_does_not_break_structure_check():
    import random
    rng = random.Random(3)
    obs = _spanning(lambda n, u, c, o: _additive(n, u, c, o,
                                                 noise=rng.uniform(-0.04, 0.04)))
    cv = cross_validate(obs)
    assert cv["structure_holds"], cv
    assert cv["worst_rel_error"] < 0.25, cv
