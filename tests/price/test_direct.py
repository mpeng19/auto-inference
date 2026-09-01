"""Pricing arithmetic, and the gate that refuses to price what it should not."""
import pathlib

import pytest

from simulator.price.direct import gpu_seconds_per_request, price_direct, usable
from simulator.price.market import Economics, Market

LEVEL = {"server_counters": {
             "sglang:forward_execution_seconds_total[extend]": 12.1,
             "sglang:forward_execution_seconds_total[decode]": 367.2},
         "prompt_tokens": 683084, "output_tokens": 109244,
         "cached_tokens": 510947, "wall_s": 380.3}


def test_reproduces_the_1xh100_baseline():
    """The 1xH100 baseline: eff-in $0.0294/M, out $5.60/M at $3.00/hr, 50%."""
    p = price_direct(gpu_seconds_input=12.1, gpu_seconds_output=367.2,
                     input_tokens=683084, output_tokens=109244,
                     cached_tokens=510947, utilization=0.50, margin=0.0)
    assert p.effective_in_per_m == pytest.approx(0.0294, abs=5e-4)
    assert p.out_per_m == pytest.approx(5.6021, abs=5e-3)
    assert p.hit_rate == pytest.approx(0.748, abs=1e-3)


def test_utilisation_scales_every_class_equally():
    """It cannot fix a ratio, which is why it can never be the answer."""
    a = price_direct(12.1, 367.2, 683084, 109244, 510947, utilization=0.50)
    b = price_direct(12.1, 367.2, 683084, 109244, 510947, utilization=0.25)
    assert b.effective_in_per_m / a.effective_in_per_m == pytest.approx(2.0)
    assert b.out_per_m / a.out_per_m == pytest.approx(2.0)


def test_gate_refuses_without_the_device_timer():
    ok, why = usable({"server_counters": {}, "prompt_tokens": 1, "output_tokens": 1})
    assert not ok and "DEVICE_TIMER" in why


def test_gate_catches_the_n_gpu_double_count():
    """Forward time above wall x n_gpu is impossible. This bug doubled every
    output price once, and the affine model disagreeing by exactly 2x is what
    caught it."""
    bad = dict(LEVEL, wall_s=100.0)
    ok, why = usable(bad, n_gpu=1)
    assert not ok and "impossible" in why


def test_gate_refuses_an_idle_level():
    ok, why = usable(dict(LEVEL, wall_s=100000.0))
    assert not ok and "busy" in why


def test_gate_accepts_the_real_level():
    assert usable(LEVEL, n_gpu=1)[0]


def test_price_times_capacity_is_exactly_the_hardware_bill():
    """The identity that proves utilisation is not double-counted anywhere."""
    for n_gpu, rate, u in ((1, 3.00, 0.50), (2, 3.00, 0.53), (8, 2.50, 0.31)):
        e = Economics(gpu_s_per_request=7.341, n_gpu=n_gpu,
                      rate_per_gpu_hour=rate, utilisation=u)
        daily = e.price_per_1k / 1000 * e.capacity_per_node_per_day()
        assert daily == pytest.approx(n_gpu * 24 * rate, rel=1e-9)


def test_share_scales_linearly_with_nodes_at_flat_price():
    m = Market.load()
    e = Economics(gpu_s_per_request=7.341, n_gpu=1, utilisation=0.50)
    assert e.nodes_for_share(2 * e.share_per_node(m), m) == pytest.approx(2.0)


def test_below_saturation_price_is_a_hyperbola_the_stack_does_not_enter():
    """Two very different stacks idle at exactly the same cost."""
    m = Market.load()
    a = Economics(gpu_s_per_request=7.341, n_gpu=1, utilisation=0.50)
    b = Economics(gpu_s_per_request=3.000, n_gpu=1, utilisation=0.50)
    tiny = 0.0005
    assert a.price_at_share(tiny, m) == pytest.approx(b.price_at_share(tiny, m))
    # ...and the better stack still wins once it is saturated.
    assert b.price_per_1k < a.price_per_1k


def test_leaderboard_disagrees_with_itself_on_purpose():
    """1st on effective input, 9th on the bill. Both are true; quoting one
    alone is how you accidentally mislead."""
    m = Market.load()
    board = m.leaderboard(0.0294, 5.6021)
    us = next(r for r in board if r["us"])
    assert us["rank_eff_in"] == 1
    assert us["rank_bill"] == 9
    assert len(board) == 12


def test_gpu_seconds_per_request_bridges_to_market_size():
    g = gpu_seconds_per_request(12.1, 367.2, 683084, 109244, 20583, 2076)
    assert g == pytest.approx(7.34, abs=0.02)


def test_the_rate_actually_reaches_the_token_prices():
    """Regression: `price_direct` took a cost-basis *name* and ignored the
    Simulator's rate, so naming a cheaper provider changed nothing.

    It was invisible because the agreed default ($3.00) happened to equal the
    default basis. Caught by the example notebook printing an identical price
    for nebius-committed at $2.50/hr.
    """
    base = price_direct(12.1, 367.2, 683084, 109244, 510947,
                        usd_per_gpu_hour=3.00, utilization=0.50)
    cheap = price_direct(12.1, 367.2, 683084, 109244, 510947,
                         usd_per_gpu_hour=2.50, utilization=0.50)
    assert cheap.out_per_m / base.out_per_m == pytest.approx(2.50 / 3.00)
    assert cheap.effective_in_per_m / base.effective_in_per_m == pytest.approx(2.50 / 3.00)


def test_a_named_provider_changes_the_whole_bill(tmp_path):
    """End to end through the Simulator, which is where the bug actually bit."""
    import json

    from simulator import Simulator

    d = tmp_path / "r"
    d.mkdir()
    rec = json.loads((pathlib.Path(__file__).resolve().parents[1]
                      / "data" / "sweep-1xh100.json").read_text())
    lv = (4, 8, 12, 16, 24)
    default = Simulator(root_dir=d, n_gpu=1, levels=lv).analyse(rec)
    nebius = Simulator(root_dir=d, n_gpu=1, levels=lv,
                       gpu_provider="nebius-committed").analyse(rec)
    assert nebius.bill_per_1k / default.bill_per_1k == pytest.approx(2.50 / 3.00)
