"""Tests for arrival processes and the eval suite.

The load shapes are only useful if they are actually the shapes we claim. A
"bursty" workload that quietly delivers a different mean rate than "sustained"
would make every comparison between them meaningless.
"""
import statistics

from autoinf.config import WorkloadConfig
from autoinf.workload import build_trace, merge_traces, mixed_trace, suite


def _gaps(tr):
    return [b.arrival_s - a.arrival_s for a, b in zip(tr.requests, tr.requests[1:])]


def _cv(tr):
    g = _gaps(tr)
    return statistics.pstdev(g) / statistics.fmean(g)


def test_poisson_rate_matches_lambda():
    tr = build_trace(WorkloadConfig(arrival="poisson", request_rate=5.0,
                                    n_requests=20000, seed=1))
    assert abs(tr.observed_rate - 5.0) / 5.0 < 0.05, tr.observed_rate


def test_constant_is_clockwork():
    tr = build_trace(WorkloadConfig(arrival="constant", request_rate=5.0,
                                    n_requests=500, seed=1))
    assert _cv(tr) < 1e-9
    assert abs(tr.observed_rate - 5.0) / 5.0 < 0.02


def test_bursty_preserves_mean_rate():
    """Same mean as sustained, or the comparison between them is invalid."""
    common = dict(request_rate=4.0, n_requests=20000, seed=3)
    smooth = build_trace(WorkloadConfig(arrival="poisson", **common))
    burst = build_trace(WorkloadConfig(arrival="bursty", burst_factor=4.0,
                                       burst_on_s=2.0, **common))
    assert abs(burst.observed_rate - 4.0) / 4.0 < 0.10, burst.observed_rate
    assert abs(burst.observed_rate - smooth.observed_rate) / smooth.observed_rate < 0.10


def test_variance_ordering_constant_poisson_bursty():
    common = dict(request_rate=4.0, n_requests=6000, seed=5)
    c = _cv(build_trace(WorkloadConfig(arrival="constant", **common)))
    p = _cv(build_trace(WorkloadConfig(arrival="poisson", **common)))
    b = _cv(build_trace(WorkloadConfig(arrival="bursty", burst_factor=4.0, **common)))
    assert c < 0.01, c            # clockwork
    assert 0.8 < p < 1.2, p       # Poisson CV is 1 by construction
    assert b > 1.5 * p, (b, p)    # bursty is meaningfully burstier


def test_ramp_matches_the_analytic_integral():
    """Counts per window must match the integral of the rate function.

    For rate(t) = r0 + (r1-r0)*t/D, the expected count over [a,b] is
    the integral of that, so this checks the shape rather than just
    "later is busier".
    """
    r0, r1, D = 2.0, 32.0, 200.0
    cfg = WorkloadConfig(arrival="ramp", request_rate=r0, ramp_end_rate=r1,
                         duration_s=D, n_requests=None, seed=7)
    tr = build_trace(cfg)

    def expected(a, b):
        f = lambda t: r0 * t + (r1 - r0) * t * t / (2 * D)
        return f(b) - f(a)

    for a, b in [(0, 50), (50, 100), (100, 150), (150, 200)]:
        obs = sum(1 for r in tr.requests if a <= r.arrival_s < b)
        exp = expected(a, b)
        assert abs(obs - exp) / exp < 0.15, (a, b, obs, round(exp, 1))

    # And the rate must be monotonically increasing across quarters.
    q = [sum(1 for r in tr.requests if a <= r.arrival_s < a + 50)
         for a in (0, 50, 100, 150)]
    assert q == sorted(q), q


def test_spike_lands_in_the_right_window():
    cfg = WorkloadConfig(arrival="spike", request_rate=4.0, spike_rate=40.0,
                         spike_at_s=30.0, spike_dur_s=10.0, duration_s=90.0,
                         n_requests=None, seed=9)
    tr = build_trace(cfg)
    inside = [r for r in tr.requests if 30.0 <= r.arrival_s < 40.0]
    outside = [r for r in tr.requests if r.arrival_s < 30.0]
    rate_in = len(inside) / 10.0
    rate_out = len(outside) / 30.0
    assert rate_in > 4 * rate_out, (rate_in, rate_out)
    assert abs(rate_in - 40.0) / 40.0 < 0.35, rate_in


def test_every_suite_workload_builds_and_is_deterministic():
    for name, cfg in suite(seed=11).items():
        a, b = build_trace(cfg), build_trace(cfg)
        assert a.digest() == b.digest(), name
        assert len(a.requests) > 10, (name, len(a.requests))
        assert a.config.name == name


def test_suite_shapes_are_distinct():
    """No two suite workloads should produce the same trace."""
    digests = {n: build_trace(c).digest() for n, c in suite(seed=13).items()}
    assert len(set(digests.values())) == len(digests), digests


def test_prefill_and_decode_heavy_are_opposites():
    s = suite(seed=15)
    pf = build_trace(s["prefill_heavy"]).describe()
    dc = build_trace(s["decode_heavy"]).describe()
    assert pf["input_tokens"]["mean"] > 5 * dc["input_tokens"]["mean"]
    assert dc["output_tokens"]["mean"] > 5 * pf["output_tokens"]["mean"]


def test_prefix_heavy_actually_shares():
    tr = build_trace(suite(seed=17)["prefix_heavy"])
    assert tr.describe()["prefix_shared_frac"] > 0.8


def test_mixed_merges_rates_and_keeps_tags():
    m = mixed_trace(seed=19, scale=0.5)
    ts = [r.arrival_s for r in m.requests]
    assert ts == sorted(ts)                                  # merged in time order
    assert [r.idx for r in m.requests] == list(range(len(m.requests)))
    tags = {r.tag for r in m.requests}
    assert tags == {"sustained", "prefill_heavy", "decode_heavy", "prefix_heavy"}


def test_merge_adds_rates():
    a = build_trace(WorkloadConfig(name="a", request_rate=2.0, n_requests=4000, seed=1))
    b = build_trace(WorkloadConfig(name="b", request_rate=3.0, n_requests=6000, seed=2))
    m = merge_traces([a, b])
    assert abs(m.observed_rate - 5.0) / 5.0 < 0.10, m.observed_rate
