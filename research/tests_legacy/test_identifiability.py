"""Per-class identifiability: the check that catches a plausible wrong answer.

An 8xH100 run reported `cache_discount = 1.52` -- cached tokens costing more
than uncached -- with fit 0.95 and condition number 5.4. Every existing check
passed. The design was degenerate in one column only: output rates spanned
108-166 per GPU-second, a factor of 1.5, so the output coefficient was fitted
to noise while the overall fit still looked healthy.

Overall conditioning asks whether the matrix is invertible. This asks whether
each coefficient is individually pinned down, which is the question that
actually determines whether a number means anything.
"""
import pytest

from autoinf.pricing import (Observation, attribute_saturated, conditioning,
                             identifiability, usable)


def _obs(name, u_rate, c_rate, o_rate, gpu_s=960.0):
    return Observation(name, u_rate * gpu_s, c_rate * gpu_s, o_rate * gpu_s, gpu_s)


# The real mixes from the 8xH100 target run.
DEGENERATE = [
    _obs("prefill_heavy", 945.4, 1049.8, 107.7, 964),
    _obs("decode_heavy", 68.1, 132.0, 166.2, 976),
    _obs("cache_heavy", 155.4, 965.6, 132.2, 965),
    _obs("balanced", 195.1, 526.9, 136.4, 973),
]

# The real mixes from the Qwen3-4B run, which did span the space.
SPANNING = [
    Observation("prefill_heavy", 1486961, 343173, 106909, 106.2),
    Observation("decode_heavy", 42132, 481798, 552887, 116.6),
    Observation("cache_heavy", 203440, 1765932, 381203, 103.4),
    Observation("balanced", 157425, 1343541, 399906, 107.3),
]


def test_flags_the_degenerate_output_column():
    d = identifiability(DEGENERATE)
    assert d["available"]
    assert not d["identified"]
    assert d["weak"] == ["out"], d["weak"]
    # The other two classes really do vary; only output is the problem.
    assert d["columns"]["uncached_in"]["identified"]
    assert d["columns"]["cached_in"]["identified"]
    assert d["columns"]["out"]["span"] < 2.0


def test_accepts_a_spanning_design():
    d = identifiability(SPANNING)
    assert d["identified"], d["weak"]


def test_condition_number_alone_would_have_passed_it():
    """The precise failure: every prior check said the run was fine."""
    cond = conditioning(DEGENERATE)
    attr = attribute_saturated(DEGENERATE)
    assert cond["well_conditioned"], "condition number did not catch it"
    assert attr.r2 > 0.9, "fit quality did not catch it"
    # And the answer it produced was physically implausible.
    assert attr.cache_discount > 1.0
    # Only the per-class check rejects it.
    ok, why = usable(attr, cond, identifiability(DEGENERATE))
    assert not ok
    assert "out" in why


def test_usable_still_accepts_a_good_run():
    attr = attribute_saturated(SPANNING)
    ok, why = usable(attr, conditioning(SPANNING), identifiability(SPANNING))
    assert ok, why
    assert attr.cache_discount < 0.5


def test_needs_enough_observations():
    assert not identifiability(DEGENERATE[:2])["available"]


def test_new_mix_design_would_span_the_output_column():
    """The redesigned mixes differ 250x in output length, so the column moves.

    Rates below are the shape those mixes should produce: an uncached mix with
    6000-token prompts and 8-token outputs, a decode mix with 64-token prompts
    and 2000-token outputs, a heavily reused cache mix, and a middle case.
    """
    designed = [
        _obs("uncached", 3000.0, 30.0, 4.0),
        _obs("decode", 30.0, 5.0, 900.0),
        _obs("cached", 60.0, 2500.0, 5.0),
        _obs("balanced", 400.0, 500.0, 110.0),
    ]
    d = identifiability(designed)
    assert d["identified"], d["weak"]
    assert d["columns"]["out"]["span"] > 100
