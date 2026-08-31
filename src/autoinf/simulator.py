"""Predict serving cost and latency, with a fidelity number attached.

The objective needs a simulator that turns (workload, serving system) into an
effective input price. A simulator nobody can check is worth nothing, so this
one is built to be falsifiable:

  * It has **two** free parameters -- the fraction of memory bandwidth decode
    realises, and the fraction of peak FLOP/s prefill realises. Everything else
    is architecture and hardware.
  * Two parameters cannot absorb four GPU counts. Calibrating on one
    configuration and predicting the others is therefore a real test, and
    `validate_loo` runs exactly that.

What it deliberately does not model: queueing. TTFT under load depends on the
scheduler's prefill/decode interleaving, which is a policy we intend to change,
so predicting it from first principles would bake in the thing being optimised.
TPOT and GPU-seconds per token are the predictions that carry the cost model,
and they are the ones validated here.

Measured anchor (run 1788203369, 8xH100, 20.6k context, batch 96.7):

    roofline  4.739e-04 GPU-s per output token
    measured  1.775e-03      -> decode realises 27% of HBM bandwidth
"""
from __future__ import annotations

from dataclasses import dataclass

from .flops import HARDWARE, MODELS, HardwareSpec, ModelSpec

# FP8 weights, 16-bit KV. SGLang stores KV at bf16 unless told otherwise.
WEIGHT_BYTES = 1.0
KV_BYTES = 2.0


@dataclass(frozen=True)
class Efficiency:
    """The whole calibration surface: two numbers."""
    decode_bw: float = 0.27       # fraction of HBM bandwidth realised
    prefill_flops: float = 0.30   # fraction of peak FLOP/s realised

    def __post_init__(self) -> None:
        for f in (self.decode_bw, self.prefill_flops):
            if not 0 < f <= 1:
                raise ValueError(f"efficiency must be in (0, 1], got {f}")


@dataclass(frozen=True)
class Workload:
    """One request's shape. Defaults are OpenRouter's observed traffic."""
    input_tokens: int = 20_583
    output_tokens: int = 2_076
    cache_hit: float = 0.394

    @property
    def uncached_tokens(self) -> float:
        return self.input_tokens * (1.0 - self.cache_hit)


def decode_bytes_per_step(m: ModelSpec, batch: int, context: int) -> float:
    """Weights are re-read every step; KV is re-read for every sequence.

    This is why cost per output token falls with batch and rises with context,
    and why tensor parallelism changes latency but not cost per token: it
    divides both the bytes and the bandwidth by the same factor.
    """
    weights = m.active_params * WEIGHT_BYTES
    kv = batch * context * m.kv_bytes_per_token(KV_BYTES)
    return weights + kv


def decode_step_s(m: ModelSpec, hw: HardwareSpec, n_gpu: int, batch: int,
                  context: int, eff: Efficiency) -> float:
    bw = hw.hbm_bandwidth * n_gpu * eff.decode_bw
    return decode_bytes_per_step(m, batch, context) / bw


def prefill_flops(m: ModelSpec, tokens: int) -> float:
    """Dense FLOPs are linear; attention is quadratic in the full-attn layers.

    Only `n_kv_layers` of 64 use full attention -- the rest are linear and
    scale with length, not length squared. `flops.attn_flops_per_token`
    charges all 64, which overstates prefill on this model by 4x.
    """
    dense = tokens * m.dense_flops_per_token()
    full_frac = (m.n_kv_layers or m.n_layers) / m.n_layers
    quad = m.attn_flops_per_token(1) * full_frac * tokens ** 2 / 2
    return dense + quad


def prefill_s(m: ModelSpec, hw: HardwareSpec, n_gpu: int, tokens: int,
              eff: Efficiency) -> float:
    peak = hw.peak_flops_dense * n_gpu * eff.prefill_flops
    return prefill_flops(m, tokens) / peak


def gpu_s_per_output_token(m: ModelSpec, hw: HardwareSpec, n_gpu: int,
                           batch: int, context: int, eff: Efficiency) -> float:
    return n_gpu * decode_step_s(m, hw, n_gpu, batch, context, eff) / batch


def gpu_s_per_input_token(m: ModelSpec, hw: HardwareSpec, n_gpu: int,
                          tokens: int, eff: Efficiency) -> float:
    return n_gpu * prefill_s(m, hw, n_gpu, tokens, eff) / tokens


def max_batch_at_tpot(m: ModelSpec, hw: HardwareSpec, n_gpu: int, context: int,
                      tpot_ms: float, eff: Efficiency, cap: int = 2048) -> int:
    """Largest batch whose decode step still fits the TPOT budget.

    Also bounded by KV capacity: weights and cache share the same HBM.
    """
    usable = hw.hbm_bytes * n_gpu * 0.85 - m.active_params * WEIGHT_BYTES
    kv_per_seq = context * m.kv_bytes_per_token(KV_BYTES)
    mem_cap = int(max(0, usable) // max(1, kv_per_seq))
    best = 0
    for b in range(1, min(cap, max(1, mem_cap)) + 1):
        if decode_step_s(m, hw, n_gpu, b, context, eff) * 1e3 <= tpot_ms:
            best = b
        else:
            break
    return best


@dataclass
class Prediction:
    n_gpu: int
    batch: int
    tpot_ms: float
    gpu_s_out: float
    gpu_s_in: float
    tokens_per_s: float

    def as_dict(self) -> dict:
        return {"n_gpu": self.n_gpu, "batch": self.batch,
                "tpot_ms": round(self.tpot_ms, 2),
                "gpu_s_out": self.gpu_s_out, "gpu_s_in": self.gpu_s_in,
                "tokens_per_s": round(self.tokens_per_s, 1)}


def predict(model: str, gpu: str, n_gpu: int, wl: Workload, tpot_ms: float,
            eff: Efficiency | None = None) -> Prediction:
    m, hw, e = MODELS[model], HARDWARE[gpu], eff or Efficiency()
    ctx = wl.input_tokens + wl.output_tokens // 2      # mean during generation
    b = max_batch_at_tpot(m, hw, n_gpu, ctx, tpot_ms, e)
    if b == 0:
        return Prediction(n_gpu, 0, float("inf"), float("inf"), float("inf"), 0.0)
    step = decode_step_s(m, hw, n_gpu, b, ctx, e)
    return Prediction(
        n_gpu=n_gpu, batch=b, tpot_ms=step * 1e3,
        gpu_s_out=gpu_s_per_output_token(m, hw, n_gpu, b, ctx, e),
        gpu_s_in=gpu_s_per_input_token(m, hw, n_gpu, int(wl.uncached_tokens), e),
        tokens_per_s=b / step)


# ── calibration and the fidelity test ────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    """One measured configuration, from a frontier run."""
    n_gpu: int
    batch: float
    context: int
    gpu_s_out: float          # measured GPU-seconds per output token
    model: str = "Qwen/Qwen3.8-27B-FP8"
    gpu: str = "H100"


def calibrate_decode(obs: list[Observation]) -> float:
    """Fit the one number decode cost depends on.

    `gpu_s_out = n_gpu * bytes(batch, ctx) / (bw * n_gpu * f) / batch`, so f
    appears only as a divisor: each observation implies a value directly, and
    the fit is their mean. No optimiser, nothing to overfit.
    """
    fs = []
    for o in obs:
        m, hw = MODELS[o.model], HARDWARE[o.gpu]
        ideal = (decode_bytes_per_step(m, int(o.batch), o.context)
                 / (hw.hbm_bandwidth * o.n_gpu)) / o.batch * o.n_gpu
        fs.append(ideal / o.gpu_s_out)
    return sum(fs) / len(fs)


def validate_loo(obs: list[Observation]) -> dict:
    """Leave-one-out: calibrate on the rest, predict the held-out config.

    This is the fidelity number. Two parameters cannot memorise four GPU
    counts, so a low error here means the model has the physics right rather
    than having been fitted to the answer.
    """
    if len(obs) < 2:
        return {"available": False, "reason": "need >= 2 configurations"}
    rows = []
    for i, held in enumerate(obs):
        rest = obs[:i] + obs[i + 1:]
        f = calibrate_decode(rest)
        m, hw = MODELS[held.model], HARDWARE[held.gpu]
        pred = (m.active_params * WEIGHT_BYTES
                + int(held.batch) * held.context * m.kv_bytes_per_token(KV_BYTES))
        pred = pred / (hw.hbm_bandwidth * held.n_gpu * f) / held.batch * held.n_gpu
        err = (pred - held.gpu_s_out) / held.gpu_s_out
        rows.append({"n_gpu": held.n_gpu, "batch": round(held.batch, 1),
                     "calibrated_on": [o.n_gpu for o in rest],
                     "decode_bw": round(f, 4),
                     "predicted": pred, "measured": held.gpu_s_out,
                     "rel_error": round(err, 4)})
    errs = [abs(r["rel_error"]) for r in rows]
    return {"available": True, "rows": rows,
            "mean_abs_error": round(sum(errs) / len(errs), 4),
            "worst_abs_error": round(max(errs), 4),
            "decode_bw_all": round(calibrate_decode(obs), 4)}
