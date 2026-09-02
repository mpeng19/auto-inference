"""A stack is identified by its content, because that is the cache key."""

import json

from simulator.stack import InferenceStack


def test_stock_is_stock():
    s = InferenceStack.stock()
    assert s.is_stock and s.describe() == "stock SGLang"


def test_digest_is_content_addressed(tmp_path):
    (tmp_path / "sglang" / "srt" / "managers").mkdir(parents=True)
    f = tmp_path / "sglang" / "srt" / "managers" / "x.py"
    f.write_text("A = 1\n")
    a = InferenceStack.from_dir(tmp_path)
    f.write_text("A = 2\n")
    b = InferenceStack.from_dir(tmp_path)
    assert a.digest != b.digest
    f.write_text("A = 1\n")
    assert InferenceStack.from_dir(tmp_path).digest == a.digest


def test_travels_by_value(tmp_path):
    """The whole stack must survive a round trip through the call, or a run
    cannot be reproduced from its record alone."""
    (tmp_path / "sglang" / "srt").mkdir(parents=True)
    (tmp_path / "sglang" / "srt" / "y.py").write_text("B = 3\n")
    s = InferenceStack.from_dir(tmp_path)
    r = InferenceStack.from_dict(s.as_dict())
    assert r.digest == s.digest and r.files == s.files


def test_reads_the_upstream_manifest(tmp_path):
    """Without it, drift goes undetected: a stack derived from an older sglang
    would silently revert upstream changes while still looking like a valid
    experiment."""
    (tmp_path / "sglang" / "srt" / "managers").mkdir(parents=True)
    (tmp_path / "sglang" / "srt" / "managers" / "schedule_policy.py").write_text("x = 1\n")
    (tmp_path / "UPSTREAM.json").write_text(json.dumps(
        {"srt/managers/schedule_policy.py":
         {"sglang_version": "0.5.18", "upstream_sha": "7fa154986e574cab"}}))
    s = InferenceStack.from_dir(tmp_path)
    assert s.upstream_sha["srt/managers/schedule_policy.py"] == "7fa154986e574cab"


def test_patches_are_keyed_to_their_target(tmp_path):
    (tmp_path / "sglang" / "srt").mkdir(parents=True)
    (tmp_path / "sglang" / "srt" / "z.py.patch").write_text("--- a\n+++ b\n")
    s = InferenceStack.from_dir(tmp_path)
    assert list(s.patches) == ["srt/z.py"]


def test_apply_starts_from_stock_in_a_reused_container(tmp_path):
    """The container is warm between calls. On 2026-09-02 every evaluation
    after the first ran on top of the previous stack's files, and a stock
    run in a warm container was not stock."""
    from simulator.stack import InferenceStack, _sha, restore_stock

    root = tmp_path / "sglang"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "a.py").write_text("stock a\n")
    (root / "srt" / "b.py").write_text("stock b\n")
    sha_a, sha_b = _sha(b"stock a\n"), _sha(b"stock b\n")

    a = InferenceStack(files={"srt/a.py": "edited a\n"}, upstream_sha={"srt/a.py": sha_a})
    prov = a.apply(root=root)
    assert prov["restored"] == [] and (root / "srt" / "a.py").read_text() == "edited a\n"

    # next call, same container, a different file: a.py must be stock again
    b = InferenceStack(files={"srt/b.py": "edited b\n"}, upstream_sha={"srt/b.py": sha_b})
    prov = b.apply(root=root)
    assert prov["restored"] == ["srt/a.py"]
    assert (root / "srt" / "a.py").read_text() == "stock a\n"
    assert (root / "srt" / "b.py").read_text() == "edited b\n"

    # the same file again is not "stale": it is compared with stock
    prov = a.apply(root=root)
    assert prov["stale"] == [] and prov["restored"] == ["srt/b.py"]

    # and stock means stock
    prov = InferenceStack.stock().apply(root=root)
    assert prov["restored"] == ["srt/a.py"]
    assert (root / "srt" / "a.py").read_text() == "stock a\n"
    assert restore_stock(root) == []


def test_serving_overrides_are_part_of_the_stack():
    """The same code with a different chunk size is a different experiment:
    hashed into the digest, carried in the record, applied at launch."""
    from simulator.config import ServingConfig
    from simulator.stack import InferenceStack

    a = InferenceStack(files={"srt/a.py": "x"})
    b = InferenceStack(files={"srt/a.py": "x"}, serving={"chunked_prefill_size": 16384})
    c = InferenceStack(serving={"mem_fraction_static": 0.9}, env={"SGLANG_X": "1"})
    assert a.digest != b.digest and not c.is_stock and not b.is_stock
    assert "chunked_prefill_size=16384" in b.describe()
    assert InferenceStack.from_dict(c.as_dict()) == c

    sc = ServingConfig().with_overrides(
        {"chunked_prefill_size": 16384, "extra_args": ["--flag", "v"], "ep_size": 0})
    assert sc.chunked_prefill_size == 16384 and sc.extra_args == ("--flag", "v")
    assert sc.ep_size is None and sc.model == ServingConfig().model
    assert "--chunked-prefill-size" in sc.to_sglang_args()
    import pytest
    with pytest.raises(ValueError, match="not allowed"):
        ServingConfig().with_overrides({"model": "other"})
    with pytest.raises(ValueError, match="unknown"):
        ServingConfig().with_overrides({"chunk_size": 1})


def test_serving_only_stack_applies_as_stock_code(tmp_path):
    from simulator.stack import InferenceStack

    root = tmp_path / "sglang"
    (root / "srt").mkdir(parents=True)
    (root / "srt" / "a.py").write_text("stock\n")
    prov = InferenceStack(serving={"chunked_prefill_size": 4096}).apply(root=root)
    assert prov["applied"] == [] and (root / "srt" / "a.py").read_text() == "stock\n"
