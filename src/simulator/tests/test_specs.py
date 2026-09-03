"""Model and hardware specs. Getting one wrong killed a launch 128 s in.

The spec is load-bearing twice over: it gates the parallelism config before
a container spawns, and `bytes_per_seq` is what `harness.tools.roofline`
compares a measured decode step against.
"""
import pytest

from simulator.specs import MODELS


def test_target_model_is_dense_with_hybrid_attention():
    """The config says has_moe: false and intermediate_size: 17408 -- a dense
    FFN with hybrid *attention*: full attention every 4th layer, so only 16
    of 64 layers hold growing KV."""
    m = MODELS["Qwen/Qwen3.8-27B-FP8"]
    assert m.intermediate_size == 17408
    assert m.n_kv_layers == 16 and m.n_layers == 64


def test_kv_per_token_is_64_kib_not_256():
    m = MODELS["Qwen/Qwen3.8-27B-FP8"]
    per_tok = m.n_kv_layers * 2 * m.n_kv_heads * m.head_dim * 2.0
    assert per_tok == 65536 == m.kv_bytes_per_token(2.0)


def test_per_sequence_state_at_market_context():
    """1.504 GB per 20.6k conversation, KV plus the linear-attention state."""
    m = MODELS["Qwen/Qwen3.8-27B-FP8"]
    assert m.bytes_per_seq(20583, 2.0) / 1e9 == pytest.approx(1.504, abs=0.005)


def test_every_spec_matches_the_parameter_count_in_its_own_name():
    """A spec claiming 128 experts x 512 on a '27B' implies 64B. Caught once."""
    import re
    for name, m in MODELS.items():
        got = re.search(r"(\d+(?:\.\d+)?)B", name.split("/")[-1])
        if not got:
            continue
        claimed = float(got.group(1)) * 1e9
        assert 0.5 * claimed <= m.total_params <= 2.0 * claimed, name
