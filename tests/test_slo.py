"""The SLO has to reproduce the frontier decision the sweep actually made."""
import pytest

from simulator.slo import MARKET_SLO, SLO, Bound

N12 = {"ttft_ms": {"n": 45, "p90": 571.4, "p99": 1017.9},
       "tpot_ms": {"mean": 19.3, "p90": 20.8}, "good_frac": 1.0, "n_failed": 0}
N16 = {"ttft_ms": {"n": 59, "p90": 562.4, "p99": 1564.2},
       "tpot_ms": {"mean": 22.0, "p90": 23.5}, "good_frac": 1.0, "n_failed": 0}


def test_market_slo_puts_n_star_at_12():
    s = SLO(bounds=MARKET_SLO)
    assert s.judge(N12).ok
    assert not s.judge(N16).ok


def test_mean_tpot_is_what_binds_not_ttft():
    """The finding in HANDOFF 6c: p90 TTFT is nowhere near binding on 1xH100.

    An earlier SLO judged both metrics at one percentile and made p90 TTFT the
    constraint, which sent us optimising prefill for nothing.
    """
    v = SLO(bounds=MARKET_SLO).judge(N16)
    assert v.binding == "mean TPOT"
    ttft = next(c for c in v.checks if c["label"] == "p90 TTFT")
    assert ttft["ok"] and ttft["value"] < ttft["limit"] / 4


def test_ttft_and_tpot_may_sit_at_different_quantiles():
    """Market data pins a TTFT tail and only a TPOT middle, so this must work."""
    s = SLO.parse("ttft:p90:2818,tpot:mean:20")
    assert [b.stat for b in s.bounds] == ["p90", "mean"]


def test_p99_bound_flags_under_sampling():
    """At 45 completions a 'p99' is the single worst request, not a tail."""
    v = SLO.parse("ttft:p99:1000").judge(N12)
    assert any("300 needed" in w for w in v.warnings)
    assert not v.ok            # 1017.9 > 1000, by 1.8%


def test_p90_bound_is_satisfied_by_our_sample_sizes():
    assert Bound("ttft", "p90", 1).min_samples() == 30
    assert Bound("ttft", "p99", 1).min_samples() == 300
    assert Bound("tpot", "mean", 1).min_samples() == 0
    assert not SLO(bounds=(Bound("ttft", "p90", 2818),)).judge(N16).warnings


def test_per_request_limits_are_the_loosest_bound():
    """good_frac is a blow-up detector, not a second copy of the percentile test.

    Using the tightest bound would fail ~40% of requests at a level whose p90
    sits comfortably inside its limit.
    """
    assert SLO(bounds=MARKET_SLO).per_request_limits() == (2818.0, 25.0)


def test_roundtrip_through_dict():
    s = SLO.parse("ttft:p95:1500,tpot:p90:25")
    assert SLO.from_dict(s.as_dict()).describe() == s.describe()


def test_rejects_a_metric_it_cannot_measure():
    with pytest.raises(ValueError):
        Bound("throughput", "p50", 49)
    with pytest.raises(ValueError):
        Bound("ttft", "p42", 100)
