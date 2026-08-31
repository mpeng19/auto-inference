"""A spec must reproduce the parameter count in the model's own name.

Qwen3.8-27B was specced as 128 experts x moe_intermediate 512 because the
config was not checked -- that is a 64B-parameter model, more than twice the
27B on the tin, and it underestimated FFN FLOPs ~4.3x. The real config has
`has_moe: false` and `intermediate_size: 17408`.

Nothing else caught it: KV, memory and the measured GPU-seconds were all fine,
so the roofline model was quietly wrong for the one model we care about while
every other check passed. Counting parameters is the cheapest way to notice.
"""
import re

import pytest

from autoinf.flops import MODELS

# Vision towers and other non-transformer parameters are not modelled, so a
# spec is allowed to land under the advertised size, but not wildly over.
LOW, HIGH = 0.80, 1.10


def _advertised(name: str) -> tuple[float, float | None]:
    """('Qwen3-30B-A3B-...') -> (30e9, 3e9); dense -> (n, None)."""
    total = re.search(r"[-.](\d+(?:\.\d+)?)B", name)
    active = re.search(r"-A(\d+(?:\.\d+)?)B", name)
    assert total, f"cannot parse a size out of {name}"
    return float(total.group(1)) * 1e9, (
        float(active.group(1)) * 1e9 if active else None)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_total_params_match_the_name(name):
    m = MODELS[name]
    total, _ = _advertised(name)
    assert LOW * total <= m.total_params <= HIGH * total, (
        f"{name}: spec gives {m.total_params/1e9:.1f}B, name says "
        f"{total/1e9:.0f}B")


@pytest.mark.parametrize("name", sorted(MODELS))
def test_active_params_match_the_name(name):
    m = MODELS[name]
    total, active = _advertised(name)
    if active is None:                      # dense: active ~ total
        assert m.active_params >= LOW * total
    else:
        assert LOW * active <= m.active_params <= HIGH * active, (
            f"{name}: spec gives {m.active_params/1e9:.1f}B active, name says "
            f"{active/1e9:.0f}B")


def test_the_target_is_dense():
    """Confirmed against the authoritative config, not a summary of it.

    `Qwen/Qwen3.8-27B-FP8/config.json` has no moe/expert field anywhere:
    `text_config.intermediate_size` is 17408 and `layer_types` is 48
    linear_attention + 16 full_attention. (`weight_block_size` [128,128] is
    real; the `moe_intermediate_size=512` SGLang reports is its own fallback.)
    """
    m = MODELS["Qwen/Qwen3.8-27B-FP8"]
    assert m.dense and m.n_experts == 1
    assert m.moe_intermediate == 17408
    assert m.n_kv_layers == 16 and m.n_layers == 64


def test_dense_target_still_needs_ep_for_sglang():
    """Dense, yet SGLang refuses tp=8/ep=1 -- and that cost a run.

    The MoE block guard was gated on `n_experts > 1`, so correcting the spec to
    dense silently disabled it, and the next launch died 128s in on exactly the
    constraint the guard existed to catch.
    """
    from autoinf.config import ServingConfig

    def probs(ep):
        return ServingConfig(model="Qwen/Qwen3.8-27B-FP8", gpu="H100",
                             n_gpu=8, tp_size=8, ep_size=ep).validate()

    assert probs(0) and "512" in probs(0)[0]
    assert probs(1)
    assert not probs(2) and not probs(4) and not probs(8)


def test_the_old_moe_spec_would_fail():
    from dataclasses import replace
    bad = replace(MODELS["Qwen/Qwen3.8-27B-FP8"],
                  moe_intermediate=512, n_experts=128, n_experts_active=8,
                  dense=False)
    assert bad.total_params > 2 * 27e9


def test_ep_size_zero_is_omitted_not_passed_as_zero():
    """launch.py defaults --ep to 0; `--ep-size 0` is not a no-op to SGLang."""
    from autoinf.config import ServingConfig
    args = ServingConfig(model="m", gpu="H100", n_gpu=8, tp_size=8,
                         ep_size=0).to_sglang_args()
    assert "--ep-size" not in args
    args8 = ServingConfig(model="m", gpu="H100", n_gpu=8, tp_size=8,
                          ep_size=8).to_sglang_args()
    assert args8[args8.index("--ep-size") + 1] == "8"
