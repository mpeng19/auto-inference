"""Architecture facts about the target model, and the card it runs on.

Two callers, neither of which prices anything:

  * `ServingConfig.validate()` reads `sglang_moe_check_intermediate` to catch
    a launch-time constraint before a GPU is rented.
  * `harness.tools.roofline` reads `active_params`, `bytes_per_seq` and
    `HardwareSpec.hbm_bandwidth` to report how far a measured decode step
    sits from the bandwidth floor. That ratio is a diagnostic; the price
    itself comes from the device timer, never from these numbers.

Every field is taken from the model's own `config.json`. Two earlier versions
of this spec were wrong in opposite directions and one of them killed a launch
128 s in, which is why the checks in `tests/test_specs.py` pin the parameter
count against the model's name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A dense transformer, from its config.json."""
    name: str
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    bytes_per_param: float = 1.0        # FP8
    tie_embeddings: bool = False
    # Layers that actually hold a growing KV cache. Hybrid models interleave
    # linear attention (constant state, no KV growth) with full attention, so
    # this can be far below n_layers -- Qwen3.8-27B has 16 of 64. Assuming all
    # layers cache is a 4x overestimate of memory, which changes what hardware
    # a workload needs.
    n_kv_layers: int | None = None

    # SGLang runs `check_quantized_moe_compatibility` on some models that have
    # no MoE at all, using its own fallback `moe_intermediate_size` rather than
    # anything in the config. Qwen3.8-27B-FP8 is dense -- its config has no
    # moe/expert field whatsoever -- yet SGLang refuses to load it at tp=8
    # unless `--ep-size` makes (512 / (tp/ep)) % 128 == 0. Set this to the
    # value SGLang uses so `ServingConfig.validate()` can catch the mismatch
    # locally instead of 128 GPU-seconds into a run.
    sglang_moe_check_intermediate: int | None = None

    # Gated-DeltaNet / Mamba-style layers keep a fixed-size recurrent state per
    # *sequence* instead of a cache that grows with context. That state is
    # invisible to `kv_bytes_per_token` -- which counts only the full-attention
    # layers -- yet it occupies memory and is re-read every decode step. On
    # Qwen3.8-27B it is 155 MB/sequence: 10% of per-sequence memory at the
    # marketplace's 20.6k context, 2% at 132k. Omitting it overstates how many
    # sequences fit, worst at short contexts, which is where the market lives.
    linear_num_value_heads: int | None = None
    linear_value_head_dim: int | None = None
    linear_key_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    linear_state_dtype_bytes: float = 4.0        # config: mamba_ssm_dtype float32

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
    def ffn_params_per_layer(self) -> int:
        # SwiGLU: gate, up, down
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def total_params(self) -> int:
        body = self.n_layers * (self.attn_params_per_layer + self.ffn_params_per_layer)
        heads = 1 if self.tie_embeddings else 2
        return body + heads * self.vocab_size * self.hidden_size

    @property
    def active_params(self) -> int:
        """Params touched by one token: the whole body plus the lm_head, since
        a dense model reads every weight every step. The input embedding is a
        lookup, not a read of the table."""
        body = self.n_layers * (self.attn_params_per_layer + self.ffn_params_per_layer)
        return body + self.vocab_size * self.hidden_size

    # ── per-sequence memory ──────────────────────────────────────
    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> float:
        """KV cache written per token, per sequence.

        Counts only full-attention layers: linear-attention layers keep a fixed
        -size state that does not grow with context.
        """
        layers = self.n_kv_layers if self.n_kv_layers is not None else self.n_layers
        return 2 * layers * self.n_kv_heads * self.head_dim * dtype_bytes

    @property
    def n_linear_layers(self) -> int:
        if self.n_kv_layers is None:
            return 0
        return self.n_layers - self.n_kv_layers

    @property
    def linear_state_bytes_per_seq(self) -> float:
        """Fixed per-sequence recurrent state, independent of context length."""
        if not self.linear_num_value_heads or not self.n_linear_layers:
            return 0.0
        recurrent = (self.linear_num_value_heads * (self.linear_value_head_dim or 0)
                     * (self.linear_key_head_dim or 0))
        conv = (self.linear_conv_kernel_dim or 0) * self.hidden_size
        return (recurrent + conv) * self.linear_state_dtype_bytes * self.n_linear_layers

    def bytes_per_seq(self, context: int, kv_dtype_bytes: float = 2.0) -> float:
        """All per-sequence memory: growing KV plus fixed linear state."""
        return (context * self.kv_bytes_per_token(kv_dtype_bytes)
                + self.linear_state_bytes_per_seq)


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    hbm_bandwidth: float        # bytes/s
    hbm_bytes: float


# THE TARGET. A **dense** hybrid-attention vision-language model. The config:
#
#   * 64 layers, but only **16 use full attention** (`layer_types` gives full
#     attention every 4th layer). Linear attention keeps a fixed-size state, so
#     only those 16 contribute growing KV. Counting all 64 overestimates 4x.
#   * **No MoE.** `has_moe: false`, `intermediate_size: 17408`, a plain dense
#     FFN. An earlier spec guessed 128 experts x 512 -- that is a 64B-parameter
#     model, more than twice the 27B on the tin, which is exactly what
#     `test_every_spec_matches_the_parameter_count_in_its_own_name` checks.
#     `--ep-size` is therefore meaningless here.
#   * `head_dim=256`, unusually large, which is why KV is still 64 KiB/token
#     despite only a quarter of the layers caching.
QWEN3_8_27B = ModelSpec(
    name="Qwen/Qwen3.8-27B-FP8",
    hidden_size=5120, n_layers=64, n_kv_layers=16,
    n_heads=24, n_kv_heads=4, head_dim=256,
    intermediate_size=17408,
    vocab_size=248320, tie_embeddings=False,
    sglang_moe_check_intermediate=512,
    linear_num_value_heads=48, linear_value_head_dim=128,
    linear_key_head_dim=128, linear_conv_kernel_dim=4,
)

# H100 SXM5: 3.35 TB/s HBM3, 80 GB.
H100 = HardwareSpec("H100 SXM5", 3.35e12, 80e9)

MODELS = {QWEN3_8_27B.name: QWEN3_8_27B}
HARDWARE = {"H100": H100}
