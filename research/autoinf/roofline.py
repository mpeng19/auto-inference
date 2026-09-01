"""Roofline capacity model -- what the hardware *should* do.

Moved out of the product on 2026-09-01. The settled method measures GPU
time and divides (HANDOFF SS6b); it never predicts from a roofline. Kept
because the *distance* between roofline and measurement is the
optimisation headroom, which is the finding in SS6c.
"""
from __future__ import annotations

from simulator.specs import ModelSpec, HardwareSpec, MODELS, HARDWARE


def request_cost(m: ModelSpec, input_tokens: int, output_tokens: int) -> dict:
    """FLOPs for one request, split by phase."""
    # Prefill attention grows with position: sum_{i<L} i ~ L^2/2.
    prefill_dense = input_tokens * m.dense_flops_per_token()
    prefill_attn = m.attn_flops_per_token(1) * input_tokens ** 2 / 2

    # Decode attends to a context growing from L to L+O.
    avg_ctx = input_tokens + output_tokens / 2
    decode_dense = output_tokens * m.dense_flops_per_token()
    decode_attn = output_tokens * m.attn_flops_per_token(avg_ctx)

    return {
        "prefill_tflops": (prefill_dense + prefill_attn) / 1e12,
        "decode_tflops": (decode_dense + decode_attn) / 1e12,
        "total_tflops": (prefill_dense + prefill_attn + decode_dense + decode_attn) / 1e12,
        "kv_bytes": m.kv_bytes_per_token() * (input_tokens + output_tokens),
    }


def capacity(m: ModelSpec, hw: HardwareSpec, input_tokens: int, output_tokens: int,
             batch: int = 128) -> dict:
    """Ceiling request rate, from whichever roofline binds.

    Prefill and decode contend for the same GPU, so the two ceilings are
    combined as a shared-resource fraction: each request needs some prefill
    seconds and some decode seconds, and the rate is 1 / (their sum).
    """
    c = request_cost(m, input_tokens, output_tokens)

    # Prefill: compute-bound.
    prefill_s = (c["prefill_tflops"] * 1e12) / hw.total_flops

    # Decode: memory-bound. A step costs bytes/bandwidth and yields one token
    # for each of `batch` sequences, so per-request decode seconds are
    # output_tokens * step_time / batch.
    step_bytes = m.decode_bytes_per_step(batch)
    step_s = step_bytes / hw.total_bandwidth
    decode_s = output_tokens * step_s / batch

    total_s = prefill_s + decode_s
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_assumed": batch,
        "experts_touched": round(m.experts_touched(batch), 1),
        "per_request_tflops": round(c["total_tflops"], 3),
        "prefill_tflops": round(c["prefill_tflops"], 3),
        "decode_tflops": round(c["decode_tflops"], 3),
        "decode_step_ms": round(step_s * 1000, 2),
        "decode_bytes_per_step_gb": round(step_bytes / 1e9, 1),
        "prefill_s_per_request": round(prefill_s, 4),
        "decode_s_per_request": round(decode_s, 4),
        "bound_by": "prefill/compute" if prefill_s > decode_s else "decode/bandwidth",
        "max_rps_roofline": round(1.0 / total_s, 2),
        "max_output_tok_s_roofline": round(output_tokens / total_s, 0),
    }
