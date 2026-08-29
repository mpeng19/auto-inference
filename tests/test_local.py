"""Local tests: everything here runs without a GPU or a Modal call.

These cover the parts of the harness that are easy to get subtly wrong and
impossible to notice later — trace determinism, arrival-rate correctness, and
the goodput definition.
"""
import math

from autoinf.config import SLO, ServingConfig, WorkloadConfig
from autoinf.metrics import RequestResult, percentile, summarize
from autoinf.workload import build_trace


def test_trace_is_deterministic():
    a = build_trace(WorkloadConfig(seed=7))
    b = build_trace(WorkloadConfig(seed=7))
    c = build_trace(WorkloadConfig(seed=8))
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_arrival_rate_matches_lambda():
    cfg = WorkloadConfig(n_requests=20000, request_rate=5.0, seed=1)
    tr = build_trace(cfg)
    observed = cfg.n_requests / tr.duration_s
    assert abs(observed - 5.0) / 5.0 < 0.05, observed


def test_arrivals_are_monotonic():
    tr = build_trace(WorkloadConfig(n_requests=500, seed=3))
    ts = [r.arrival_s for r in tr.requests]
    assert ts == sorted(ts)


def test_prefix_sharing_reuses_pool():
    cfg = WorkloadConfig(n_requests=400, prefix_share_frac=1.0, n_shared_prefixes=4, seed=2)
    tr = build_trace(cfg)
    ids = {r.prefix_id for r in tr.requests}
    assert ids == {0, 1, 2, 3}
    # Every request sharing a prefix must literally start with it.
    by_id = {}
    for r in tr.requests:
        by_id.setdefault(r.prefix_id, []).append(r.prompt)
    for prompts in by_id.values():
        head = prompts[0][:200]
        assert all(p.startswith(head) for p in prompts)


def test_no_sharing_by_default():
    tr = build_trace(WorkloadConfig(n_requests=100, seed=4))
    assert all(r.prefix_id is None for r in tr.requests)


def test_percentile_matches_numpy_semantics():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert math.isclose(percentile(xs, 50), 2.5)
    assert math.isclose(percentile(xs, 75), 3.25)
    assert percentile([], 50) is None
    assert percentile([9.0], 99) == 9.0


def _r(idx, ttft_ms, tpot_ms, n_out=10, ok=True):
    """Build a result with the given TTFT/TPOT, dispatched at t=0."""
    first = ttft_ms / 1000.0
    end = first + (tpot_ms / 1000.0) * (n_out - 1)
    return RequestResult(idx, 0.0, 0.0, first, end, 100, n_out, ok)


def test_goodput_requires_both_slos():
    slo = SLO(ttft_ms=500, tpot_ms=40)
    rs = [
        _r(0, 100, 20),    # good
        _r(1, 900, 20),    # TTFT violation
        _r(2, 100, 90),    # TPOT violation
        _r(3, 900, 90),    # both
        _r(4, 100, 39.9),  # good, just inside
    ]
    s = summarize(rs, slo, window_s=1.0)
    assert s["n_good"] == 2, s["n_good"]
    assert math.isclose(s["goodput_rps"], 2.0)
    assert math.isclose(s["throughput_rps"], 5.0)


def test_failed_requests_are_never_good():
    slo = SLO(ttft_ms=500, tpot_ms=40)
    bad = RequestResult(0, 0.0, 0.0, None, None, None, 0, False, "connection reset")
    s = summarize([bad, _r(1, 50, 10)], slo, window_s=1.0)
    assert s["n_failed"] == 1
    assert s["n_good"] == 1
    assert s["errors"] == {"connection reset": 1}


def test_single_token_output_has_no_tpot_violation():
    slo = SLO(ttft_ms=500, tpot_ms=1)
    one = RequestResult(0, 0.0, 0.0, 0.1, 0.1, 50, 1, True)
    s = summarize([one], slo, window_s=1.0)
    assert s["n_good"] == 1


def test_dispatch_lag_is_reported():
    late = RequestResult(0, 1.0, 1.5, 1.6, 2.0, 10, 5, True)   # 500ms late
    s = summarize([late], SLO(), window_s=1.0)
    assert math.isclose(s["client_dispatch_lag_ms"]["max"], 500.0)


def test_sglang_args_render():
    args = ServingConfig(tp_size=8, ep_size=8, enable_prefix_caching=False).to_sglang_args()
    assert "--tp-size" in args and args[args.index("--tp-size") + 1] == "8"
    assert "--ep-size" in args
    assert "--disable-radix-cache" in args


def test_config_digest_is_stable_and_sensitive():
    a = ServingConfig()
    assert a.digest() == ServingConfig().digest()
    assert a.digest() != ServingConfig(max_running_requests=128).digest()
