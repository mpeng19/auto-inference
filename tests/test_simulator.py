"""The simulator's invariants, and proof that its calibration cannot cheat.

The fidelity claim rests on the model having two free parameters and four
configurations to predict. These tests pin the physics that makes that a real
test rather than a curve fit.
"""
import pytest

from autoinf.simulator import (Efficiency, Observation, Workload,
                               calibrate_decode, decode_bytes_per_step,
                               gpu_s_per_output_token, max_batch_at_tpot,
                               predict, validate_loo)
from autoinf.flops import HARDWARE, MODELS

M = MODELS["Qwen/Qwen3.8-27B-FP8"]
H = HARDWARE["H100"]
E = Efficiency()


def test_tensor_parallelism_does_not_change_cost_per_token():
    """The finding that undercuts 8xH100: TP buys latency, not cheaper tokens.

    It divides bytes and bandwidth by the same factor, so GPU-seconds per token
    is identical at a fixed batch. What TP buys is a bigger batch inside the
    TPOT budget -- which is where the real saving comes from.
    """
    costs = [gpu_s_per_output_token(M, H, n, 32, 20_583, E) for n in (1, 2, 4, 8)]
    assert max(costs) - min(costs) < 1e-12 * max(costs), costs


def test_cost_per_token_falls_with_batch():
    """Weights are re-read every step, so batching amortises them."""
    costs = [gpu_s_per_output_token(M, H, 8, b, 20_583, E) for b in (8, 32, 128)]
    assert costs[0] > costs[1] > costs[2]


def test_cost_per_token_rises_with_context():
    """KV is re-read per sequence per step -- the term batching cannot fix."""
    costs = [gpu_s_per_output_token(M, H, 8, 32, c, E)
             for c in (2_000, 20_583, 132_092)]
    assert costs[0] < costs[1] < costs[2]


def test_batching_returns_diminish_once_kv_dominates():
    """At 20.6k context, batch 32 already captures most of the saving.

    This is why 4xH100 lands within ~8% of 8xH100 and the case for eight GPUs
    is weaker than the 1-GPU failure suggested.
    """
    c32 = gpu_s_per_output_token(M, H, 8, 32, 20_583, E)
    c256 = gpu_s_per_output_token(M, H, 8, 256, 20_583, E)
    assert c32 / c256 < 1.6, c32 / c256


def test_more_gpus_allow_a_bigger_batch():
    batches = [max_batch_at_tpot(M, H, n, 20_583, 50.0, E) for n in (1, 2, 4, 8)]
    assert batches == sorted(batches) and batches[0] < batches[-1], batches


def test_calibration_recovers_a_known_efficiency():
    """Round-trip: synthesise an observation at a known f, recover f."""
    truth = 0.41
    b, ctx, n = 64, 20_583, 4
    ideal = decode_bytes_per_step(M, b, ctx) / (H.hbm_bandwidth * n) / b * n
    obs = Observation(n_gpu=n, batch=b, context=ctx, gpu_s_out=ideal / truth)
    assert calibrate_decode([obs]) == pytest.approx(truth, rel=1e-9)


def test_leave_one_out_is_exact_when_the_physics_holds():
    """If every config shares one efficiency, LOO error must be ~0.

    A non-trivial error on real data therefore means the physics is wrong or
    the configurations genuinely differ -- not that the fit is noisy.
    """
    truth = 0.2767
    obs = []
    for n, b in ((1, 16), (2, 49), (4, 114), (8, 245)):
        ideal = decode_bytes_per_step(M, b, 20_583) / (H.hbm_bandwidth * n) / b * n
        obs.append(Observation(n_gpu=n, batch=b, context=20_583,
                               gpu_s_out=ideal / truth))
    v = validate_loo(obs)
    assert v["available"]
    assert v["worst_abs_error"] < 1e-6, v
    assert v["decode_bw_all"] == pytest.approx(truth, rel=1e-6)


def test_loo_needs_two_configurations():
    assert not validate_loo([Observation(8, 96.7, 20_583, 1.775e-03)])["available"]


def test_prediction_is_infeasible_when_no_batch_fits():
    """A 1ms TPOT budget cannot be met by any batch -- say so, don't guess."""
    p = predict("Qwen/Qwen3.8-27B-FP8", "H100", 1, Workload(), 1.0)
    assert p.batch == 0 and p.tokens_per_s == 0.0


def test_efficiency_rejects_impossible_values():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Efficiency(decode_bw=bad)
