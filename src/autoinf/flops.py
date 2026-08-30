"""Roofline capacity model: what the hardware *should* be able to do.

Sizing a load suite by guesswork is what produced the first calibration, which
was ~10x too low and measured an idle server. This derives the ceiling from
first principles instead, so a workload can be expressed as a fraction of
capacity rather than an arbitrary requests-per-second number.

The essential point about serving an MoE: **prefill and decode sit on opposite
sides of the roofline.**

  * Prefill is compute-bound. A long prompt is a big matmul; the limit is
    FLOP/s.
  * Decode is memory-bound. Each step reads weights to produce one token per
    sequence, so the limit is HBM bandwidth. And for an MoE this is worse than
    it looks: at any real batch size the tokens in a step collectively route to
    nearly *every* expert, so a step reads close to the whole model rather than
    just the active parameters. `active_params` predicts FLOPs; it badly
    under-predicts bytes.

Every number here is a ceiling, not a forecast. Real systems land well under it
(attention, KV traffic, imperfect overlap, prefill stealing decode steps). The
useful output is the ratio: measuring 50% of the bandwidth roofline is a
healthy server, 5% means something is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Architecture, from the model's own config.json."""
    name: str
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    moe_intermediate: int
    n_experts: int
    n_experts_active: int
    vocab_size: int
    bytes_per_param: float = 1.0        # FP8
    tie_embeddings: bool = False
    # A dense model is the degenerate MoE: one "expert", always active. Keeping
    # one code path means the roofline maths does not fork by architecture.
    dense: bool = False

    # ── parameter counts ─────────────────────────────────────────
    @property
    def attn_params_per_layer(self) -> int:
        d, hd = self.hidden_size, self.head_dim
        q = d * self.n_heads * hd
        k = d * self.n_kv_heads * hd
        v = d * self.n_kv_heads * hd
        o = self.n_heads * hd * d
        return q + k + v + o

    @property
    def expert_params(self) -> int:
        # SwiGLU: gate, up, down
        return 3 * self.hidden_size * self.moe_intermediate

    @property
    def moe_params_per_layer(self) -> int:
        if self.dense:
            return self.expert_params          # no router, single FFN
        router = self.hidden_size * self.n_experts
        return self.n_experts * self.expert_params + router

    @property
    def active_moe_params_per_layer(self) -> int:
        if self.dense:
            return self.expert_params
        return self.n_experts_active * self.expert_params + self.hidden_size * self.n_experts

    @property
    def total_params(self) -> int:
        body = self.n_layers * (self.attn_params_per_layer + self.moe_params_per_layer)
        heads = 1 if self.tie_embeddings else 2
        return body + heads * self.vocab_size * self.hidden_size

    @property
    def active_params(self) -> int:
        """Params touched by one token. Drives FLOPs, NOT bytes read."""
        body = self.n_layers * (self.attn_params_per_layer + self.active_moe_params_per_layer)
        return body + self.vocab_size * self.hidden_size       # lm_head only

    # ── per-token costs ──────────────────────────────────────────
    def dense_flops_per_token(self) -> float:
        """2 FLOPs per parameter per token (one multiply, one add)."""
        return 2.0 * self.active_params

    def attn_flops_per_token(self, context_len: int) -> float:
        """QK^T and AV, both linear in context length."""
        per_layer = 2 * 2 * self.n_heads * self.head_dim * context_len
        return per_layer * self.n_layers

    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> float:
        """KV cache written per token, per sequence. GQA makes this cheap."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * dtype_bytes

    def experts_touched(self, batch: int) -> float:
        """Expected distinct experts activated by `batch` tokens in one step.

        With uniform routing, P(expert unused) = (1 - k/E)^batch. This
        saturates fast: at batch 64 with 8-of-128 routing, ~127 of 128 experts
        are touched. That is why decode bytes track *total* params, not active.
        """
        if self.dense:
            return 1.0
        p_unused = (1.0 - self.n_experts_active / self.n_experts) ** batch
        return self.n_experts * (1.0 - p_unused)

    def decode_bytes_per_step(self, batch: int) -> float:
        """Weight bytes read for one decode step at a given batch size."""
        attn = self.n_layers * self.attn_params_per_layer
        experts = self.n_layers * self.experts_touched(batch) * self.expert_params
        head = self.vocab_size * self.hidden_size
        return (attn + experts + head) * self.bytes_per_param


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    n_gpus: int
    peak_flops_dense: float     # FLOP/s, at the serving dtype
    hbm_bandwidth: float        # bytes/s
    hbm_bytes: float            # per GPU

    @property
    def total_flops(self) -> float:
        return self.peak_flops_dense * self.n_gpus

    @property
    def total_bandwidth(self) -> float:
        return self.hbm_bandwidth * self.n_gpus


# ── known specs ──────────────────────────────────────────────────
QWEN3_30B_A3B = ModelSpec(
    name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    hidden_size=2048, n_layers=48, n_heads=32, n_kv_heads=4, head_dim=128,
    moe_intermediate=768, n_experts=128, n_experts_active=8, vocab_size=151936,
)

# Dense small models, for cheap iteration. They lose the MoE dynamics (expert
# routing, the all-experts-touched decode read) but keep every scheduling,
# batching, KV and prefix-cache behaviour, which is most of what the serving
# layer actually does. Qwen3-4B is the pick: 262k native context, so it can
# hold an agentic conversation, on a GPU that costs a fraction of an H100.
QWEN3_4B = ModelSpec(
    name="Qwen/Qwen3-4B-Instruct-2507-FP8",
    hidden_size=2560, n_layers=36, n_heads=32, n_kv_heads=8, head_dim=128,
    moe_intermediate=9728, n_experts=1, n_experts_active=1, vocab_size=151936,
    tie_embeddings=True, dense=True,
)

QWEN3_8B = ModelSpec(
    name="Qwen/Qwen3-8B-FP8",
    hidden_size=4096, n_layers=36, n_heads=32, n_kv_heads=8, head_dim=128,
    moe_intermediate=12288, n_experts=1, n_experts_active=1, vocab_size=151936,
    tie_embeddings=False, dense=True,
)

QWEN3_235B_A22B = ModelSpec(
    name="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    hidden_size=4096, n_layers=94, n_heads=64, n_kv_heads=4, head_dim=128,
    moe_intermediate=1536, n_experts=128, n_experts_active=8, vocab_size=151936,
)

# H100 SXM5: 989.4 TFLOP/s dense FP8 (1979 with 2:4 sparsity, which serving
# does not use), 3.35 TB/s HBM3.
H100 = HardwareSpec("H100 SXM5", 1, 989.4e12, 3.35e12, 80e9)
H100_8X = HardwareSpec("8x H100 SXM5", 8, 989.4e12, 3.35e12, 80e9)
# Cheaper accelerators for iteration. Note how much bandwidth they give up:
# decode is bandwidth-bound, so an L4 is ~11x slower at decode than an H100
# while costing only ~5x less. Cheaper per hour is not cheaper per token.
L40S = HardwareSpec("L40S", 1, 362e12, 864e9, 48e9)
A10G = HardwareSpec("A10G", 1, 125e12, 600e9, 24e9)
L4 = HardwareSpec("L4", 1, 121e12, 300e9, 24e9)

MODELS = {m.name: m for m in (QWEN3_4B, QWEN3_8B, QWEN3_30B_A3B, QWEN3_235B_A22B)}
HARDWARE = {"H100": H100, "L40S": L40S, "A10G": A10G, "L4": L4}

# Modal per-hour rates, for cost-per-token comparisons.
GPU_HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "L40S": 1.95,
              "A10G": 1.10, "L4": 0.80}


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
