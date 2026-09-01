"""A stack is identified by its content, because that is the cache key."""

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


def test_reads_the_repo_overlays():
    s = InferenceStack.from_dir("overlays")
    assert not s.is_stock
    assert "srt/managers/schedule_policy.py" in s.files
    assert s.upstream_sha, "UPSTREAM.json must be read, or drift goes undetected"


def test_patches_are_keyed_to_their_target(tmp_path):
    (tmp_path / "sglang" / "srt").mkdir(parents=True)
    (tmp_path / "sglang" / "srt" / "z.py.patch").write_text("--- a\n+++ b\n")
    s = InferenceStack.from_dir(tmp_path)
    assert list(s.patches) == ["srt/z.py"]
