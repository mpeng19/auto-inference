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
    # Per sequence: growing KV *and* the fixed linear-attention state, which
    # is re-read every step just like KV. 48 of this model's 64 layers are
    # linear, contributing 155 MB/sequence that a KV-only count misses.
    per_seq = m.bytes_per_seq(context, KV_BYTES)
    return weights + batch * per_seq


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
    mem_cap = int(max(0, usable) // max(1.0, m.bytes_per_seq(context, KV_BYTES)))
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
        # Call the shared byte count rather than restating it: an inlined copy
        # here silently kept a KV-only formula when linear-attention state was
        # added, and the round-trip test caught the drift.
        pred = (decode_bytes_per_step(m, int(held.batch), held.context)
                / (hw.hbm_bandwidth * held.n_gpu * f) / held.batch * held.n_gpu)
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


@dataclass(frozen=True)
class PrefillObservation:
    n_gpu: int
    chunk_tokens: int         # tokens prefilled per scheduler chunk
    gpu_s_in: float           # measured GPU-seconds per uncached input token
    model: str = "Qwen/Qwen3.8-27B-FP8"
    gpu: str = "H100"


def calibrate_prefill(obs: list[PrefillObservation]) -> float:
    fs = []
    for o in obs:
        m, hw = MODELS[o.model], HARDWARE[o.gpu]
        ideal = (prefill_flops(m, o.chunk_tokens)
                 / (hw.peak_flops_dense * o.n_gpu)) / o.chunk_tokens * o.n_gpu
        fs.append(ideal / o.gpu_s_in)
    return sum(fs) / len(fs)


# ── inverting a competitor's published latency ───────────────────────────

def infer_operating_point(tokens_per_s: float, context: int,
                          eff: Efficiency | None = None,
                          model: str = "Qwen/Qwen3.8-27B-FP8",
                          gpu: str = "H100",
                          gpu_counts: tuple[int, ...] = (1, 2, 4, 8),
                          ) -> list[dict]:
    """What batch would produce this per-stream decode rate?

    This is the move that makes a *fair* comparison possible. A competitor's
    cost is unobservable, but OpenRouter publishes their throughput, and
    per-stream decode rate is 1/TPOT -- which our model ties to (GPUs, batch,
    context). Solving for batch gives their implied GPU-seconds per output
    token, so we can compare cost against cost rather than our cost against
    their price.

    Underdetermined on its own: 83 tok/s is consistent with 8 GPUs at a large
    batch or 4 at a small one. It returns every consistent (GPUs, batch) pair
    and the cost each implies, which brackets them rather than pinning them.

    Assumes the competitor runs the same model on the same hardware at the same
    efficiency we do. The first is true by construction; the others are not, so
    treat the output as a bracket, not a measurement.
    """
    m, hw, e = MODELS[model], HARDWARE[gpu], eff or Efficiency()
    step_s = 1.0 / tokens_per_s
    kv_per_seq = context * m.kv_bytes_per_token(KV_BYTES)
    out = []
    for n in gpu_counts:
        budget = step_s * hw.hbm_bandwidth * n * e.decode_bw
        spare = budget - m.active_params * WEIGHT_BYTES
        if spare <= 0:
            continue                       # cannot even stream the weights
        batch = spare / kv_per_seq
        if batch < 1:
            continue
        out.append({"n_gpu": n, "implied_batch": round(batch, 1),
                    "gpu_s_out": gpu_s_per_output_token(
                        m, hw, n, max(1, int(batch)), context, e),
                    "tpot_ms": round(step_s * 1e3, 2)})
    return out


# ── the model that actually predicts: constant step time ─────────────────
#
# The roofline model above scores 27% mean leave-one-out error across TP
# 1/2/4/8 -- against 31% for a zero-parameter null that just predicts the mean
# cost. Its two parameters buy essentially nothing, and it is worse in the
# tail (49% vs 41%). The cause: it assumes tensor parallelism is free. It is
# not -- measured decode efficiency falls from 0.60 at TP=2 to 0.29 at TP=8.
#
# What the data shows instead is that the time to advance one decode step is
# ~constant, across 8x the GPUs and 6x the batch:
#
#     GPUs  batch  batch/GPU  step ms   GPU-s/out token
#        1   15.6       15.6     23.0         1.473e-03
#        2   40.6       20.3     20.6         1.017e-03
#        4   78.1       19.5     20.3         1.042e-03
#        8   96.7       12.1     21.5         1.775e-03
#
#     mean 21.4ms, sd 1.0ms, spread 1.13x
#
# So cost per output token is STEP * n_gpu / batch, and everything reduces to
# *what batch a configuration can sustain*. One parameter, 5% mean LOO error.
#
# This is an empirical invariant, not a derivation -- 21.4ms has no first-
# principles justification, and roofline says TP=8 should manage 6.2ms. Treat
# it as calibrated-for-this-stack: it should be re-measured for a different
# model, GPU, or workload shape, and a change in it is itself a finding, since
# closing the gap to roofline is the largest available optimisation.


@dataclass(frozen=True)
class StepModel:
    """Cost per output token = step_s * n_gpu / batch."""
    step_s: float = 0.0214

    def gpu_s_out(self, n_gpu: int, batch: float) -> float:
        return self.step_s * n_gpu / batch

    def tokens_per_s(self, batch: float) -> float:
        return batch / self.step_s


def calibrate_step(obs: list[Observation]) -> StepModel:
    """Each observation implies a step time directly; the fit is their mean."""
    return StepModel(sum(o.gpu_s_out * o.batch / o.n_gpu for o in obs) / len(obs))


def validate_loo_step(obs: list[Observation]) -> dict:
    """The same held-out test applied to the constant-step model."""
    if len(obs) < 2:
        return {"available": False, "reason": "need >= 2 configurations"}
    rows = []
    for i, held in enumerate(obs):
        rest = obs[:i] + obs[i + 1:]
        pred = calibrate_step(rest).gpu_s_out(held.n_gpu, held.batch)
        rows.append({"n_gpu": held.n_gpu, "batch": round(held.batch, 1),
                     "predicted": pred, "measured": held.gpu_s_out,
                     "rel_error": round((pred - held.gpu_s_out) / held.gpu_s_out, 4)})
    errs = [abs(r["rel_error"]) for r in rows]
    m = calibrate_step(obs)
    return {"available": True, "rows": rows,
            "mean_abs_error": round(sum(errs) / len(errs), 4),
            "worst_abs_error": round(max(errs), 4),
            "step_ms": round(m.step_s * 1e3, 2)}


# Measured on the market workload (20,583 in / 2,076 out) at TP 1/2/4/8.
# `batch` is the decode mix's mean running batch from `BatchSampler`.
SWEEP_2026_08_31 = [
    Observation(n_gpu=1, batch=15.6, context=22_659, gpu_s_out=1.473e-03),
    Observation(n_gpu=2, batch=40.6, context=22_659, gpu_s_out=1.017e-03),
    Observation(n_gpu=4, batch=78.1, context=22_659, gpu_s_out=1.042e-03),
    Observation(n_gpu=8, batch=96.7, context=22_659, gpu_s_out=1.775e-03),
]


# ── predicting the batch, which is what everything reduces to ────────────
#
# `schedule_policy.py` admits requests until the KV pool is exhausted, charging
# each running request its prompt *plus* a reservation for future decode
# tokens: `min(max_new_tokens, CLIP_MAX_NEW_TOKENS=4096) * new_token_ratio`,
# where `new_token_ratio` scales with `--schedule-conservativeness`.
#
# That is why batch and GPU count were confounded (§3.3.1 of docs/design.md):
# the KV pool scales with GPU count, so the ceiling does too. Backing out an
# implied bytes-per-sequence from the sweep:
#
#     GPUs  usable GB  batch  implied GB/seq   modelled GB/seq
#        1       44.9   15.6            2.88              1.64
#        2      112.9   40.6            2.78              1.64
#        4      248.9   78.1            3.19              1.64
#        8      520.9   96.7            5.39              1.64
#
# TP 1/2/4 sit at a constant ~2.9 GB/sequence -- batch really is the memory
# ceiling there, and the 1.77x over the modelled figure is the scheduler's
# decode reservation plus page rounding. **TP=8 is the outlier**: it reaches
# only ~54% of the same ceiling, so at TP=8 something other than memory caps
# admission. Since cost falls with batch, that gap is the largest identified
# saving in the stack -- see docs/design.md §8.2.

RESERVATION_FACTOR = 1.77   # calibrated on TP 1/2/4; scheduler reserve + rounding


def predict_batch(model: str, gpu: str, n_gpu: int, context: int,
                  mem_fraction: float = 0.85,
                  reservation: float = RESERVATION_FACTOR,
                  max_running_requests: int = 256) -> int:
    """Largest batch the KV pool admits, per SGLang's admission rule."""
    m, hw = MODELS[model], HARDWARE[gpu]
    usable = hw.hbm_bytes * n_gpu * mem_fraction - m.active_params * WEIGHT_BYTES
    if usable <= 0:
        return 0
    per_seq = m.bytes_per_seq(context, KV_BYTES) * reservation
    return int(min(max_running_requests, max(0.0, usable) // per_seq))


def validate_batch(obs: list[Observation], **kw) -> dict:
    """Predicted vs measured batch. TP=8 is expected to miss high."""
    rows = []
    for o in obs:
        p = predict_batch(o.model, o.gpu, o.n_gpu, o.context, **kw)
        rows.append({"n_gpu": o.n_gpu, "predicted": p, "measured": o.batch,
                     "rel_error": round((p - o.batch) / o.batch, 3)})
    errs = [abs(r["rel_error"]) for r in rows]
    return {"rows": rows, "mean_abs_error": round(sum(errs) / len(errs), 3),
            "worst_abs_error": round(max(errs), 3)}


# Measurement noise floor: worst residual across accepted attribution runs was
# ~8% (fit 0.92-0.94), so an experiment whose predicted effect is smaller than
# this cannot resolve it however long it runs.
FIT_NOISE = 0.08


def effect_size(model: str, context: int, batch_lo: float, batch_hi: float,
                gpu: str = "H100") -> dict:
    """Will this experiment produce a signal? Check BEFORE spending.

    Cost per output token is `n * (W/batch + per_seq) / bandwidth`, so it varies
    with batch **only through the weight term**. When per-sequence KV dominates
    the weights, cost is nearly flat in batch and no batch sweep can measure
    anything.

    This exists because a planned sweep on the Qwen3-4B dev model — chosen to
    save money — would have produced 1.21x cost variation over a 2.2x batch
    range, inside the 8% fit noise. The 4B has 4 GB of weights against the
    27B's 23 GB, so the effect being tested is largely absent there. Cheap
    iteration on a dev model is fine for anything model-independent; this is
    not that. Run validation on the model being sold.
    """
    m = MODELS[model]
    W, ps = m.active_params * WEIGHT_BYTES, m.bytes_per_seq(context, KV_BYTES)
    lo, hi = W / batch_hi + ps, W / batch_lo + ps
    ratio = hi / lo
    return {"model": model, "cost_ratio": round(ratio, 3),
            "batch_ratio": round(batch_hi / batch_lo, 2),
            "weight_frac_at_lo_batch": round((W / batch_lo) / hi, 3),
            # Require the effect to be 4x the per-point noise. At 2x, the
            # rejected dev-model sweep (1.21x over 2.2x batch) would have
            # passed: 21% effect against +-8% on each endpoint is a
            # signal-to-noise of ~1.9, which is not enough to *fit a
            # functional form*, only to notice a difference. We are measuring
            # an exponent, not testing for a difference.
            "resolvable": bool(ratio - 1.0 > 4 * FIT_NOISE),
            "snr": round((ratio - 1.0) / (FIT_NOISE * 2 ** 0.5), 2),
            "noise_floor": FIT_NOISE}


# Serving physics is measured on ONE model. The GPU may vary — that is a
# parameter we deliberately sweep — but the model may not, because cost per
# token varies with batch only through the weight read and a 4B has 4 GB of
# weights against this model's 23 GB. Calibrating on anything else imports a
# different regime. The dev model is for harness validation only.
TARGET_MODEL = "Qwen/Qwen3.8-27B-FP8"


def assert_target_model(obs: list[Observation]) -> None:
    wrong = sorted({o.model for o in obs} - {TARGET_MODEL})
    if wrong:
        raise ValueError(
            f"calibration data must all be {TARGET_MODEL}; got {wrong}. "
            "The GPU may vary; the model may not.")


def discriminate(obs: list[Observation]) -> dict:
    """Constant-step vs roofline, on observations that share a GPU count.

    The two models disagree sharply about what happens as batch grows at fixed
    hardware. Writing `step = gpu_s_out * batch / n_gpu`:

      * constant-step says `step` does not move with batch;
      * roofline says `step = (W + batch*per_seq) / bandwidth`, which grows —
        at 1xH100 and 4k context it roughly doubles from batch 31 to 117.

    So the discriminating statistic is simply whether `step` is flat or rising.
    This is the test the TP sweep could not run, because there batch and GPU
    count moved together (docs/design.md §3.3.1c).

    Returns the observed step spread against each model's prediction. All
    observations must share `n_gpu`, `model` and `gpu`.
    """
    assert_target_model(obs)
    if len({(o.n_gpu, o.gpu) for o in obs}) != 1:
        raise ValueError("discriminate() needs one GPU count and type")
    if len(obs) < 2:
        return {"available": False, "reason": "need >= 2 batches"}

    o0 = obs[0]
    m, hw = MODELS[o0.model], HARDWARE[o0.gpu]
    rows = []
    for o in sorted(obs, key=lambda x: x.batch):
        step = o.gpu_s_out * o.batch / o.n_gpu
        roof = decode_bytes_per_step(m, int(o.batch), o.context) / (
            hw.hbm_bandwidth * o.n_gpu)
        rows.append({"batch": round(o.batch, 1), "step_ms": round(step * 1e3, 2),
                     "roofline_step_ms": round(roof * 1e3, 2),
                     "implied_bw_frac": round(roof / step, 3)})

    steps = [r["step_ms"] for r in rows]
    roofs = [r["roofline_step_ms"] for r in rows]
    obs_spread = max(steps) / min(steps)
    roof_spread = max(roofs) / min(roofs)
    # Which prediction is the observed spread closer to, in log space?
    import math
    d_const = abs(math.log(obs_spread))
    d_roof = abs(math.log(obs_spread) - math.log(roof_spread))
    return {"available": True, "rows": rows,
            "batch_range": round(max(o.batch for o in obs)
                                 / min(o.batch for o in obs), 2),
            "observed_step_spread": round(obs_spread, 3),
            "constant_step_predicts": 1.0,
            "roofline_predicts": round(roof_spread, 3),
            "verdict": ("constant-step" if d_const < d_roof else "roofline"),
            "margin": round(abs(d_const - d_roof), 3)}
