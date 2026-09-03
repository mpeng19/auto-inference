"""Hybrid search: the blend, the vector store, and the model's absence."""
import sqlite3
import sys
from typing import ClassVar

import numpy as np
import pytest

from harness import embeddings as emb
from harness.embeddings import (
    DEFAULT,
    EmbeddingStore,
    HashEmbedder,
    LocalEmbedder,
    default_embedder,
    hybrid_rank,
    hybrid_search,
    resolve,
)


class CountingEmbedder:
    """A HashEmbedder that records every `encode`, so a test can see what
    was embedded and what was read back from the table."""

    def __init__(self, name: str = "hash256"):
        self.inner = HashEmbedder(name=name)
        self.name = name
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return self.inner.encode(texts)


def test_hash_embedder_is_deterministic_unit_and_tolerant_of_inflection():
    e = HashEmbedder()
    a, b = e.encode(["quantise the kv cache", "quantise the kv cache"])
    assert a == b and len(a) == 256 and abs(np.linalg.norm(a) - 1) < 1e-5
    assert HashEmbedder().encode(["x y z"]) == e.encode(["x y z"])     # across instances
    q, v, u = (np.array(x) for x in e.encode(
        ["kv quantisation", "quantised kv", "speculative decoding"]))
    assert q @ v > 0.5                       # shared stem "quant" and "kv"
    assert abs(q @ u) < 1e-6                 # nothing shared
    assert e.encode([""])[0] == [0.0] * 256
    assert isinstance(e, emb.Embedder) and isinstance(CountingEmbedder(), emb.Embedder)


def test_hybrid_rank_is_pure_lexical_without_an_embedder():
    cands = [("a", "x"), ("b", "y"), ("c", "z")]
    got = hybrid_rank("q", cands, None, {"b": 2.0, "a": 1.0})
    assert got == [("b", 2.0), ("a", 1.0), ("c", 0.0)]      # raw scores, as given


def test_hybrid_rank_blends_min_max_cosine_with_min_max_lexical():
    class Stub:
        name = "stub"
        vecs: ClassVar = {"q": [1.0, 0.0], "lex": [0.0, 1.0], "sem": [1.0, 0.0], "none": [-1.0, 0.0]}

        def encode(self, texts):
            return [self.vecs[t] for t in texts]

    cands = [("lex", "lex"), ("sem", "sem"), ("none", "none")]
    # cosine: lex 0, sem 1, none -1 -> min-max 0.5, 1, 0; lexical: lex only -> 1, 0, 0
    got = dict(hybrid_rank("q", cands, Stub(), {"lex": 3.0}))
    assert got == {"lex": pytest.approx(0.75), "sem": pytest.approx(0.5), "none": 0.0}
    assert hybrid_rank("q", cands, Stub(), {"lex": 3.0}, alpha=1.0)[0][0] == "sem"
    assert hybrid_rank("q", cands, Stub(), {"lex": 3.0}, alpha=0.0)[0][0] == "lex"


def test_store_persists_vectors_and_reembeds_when_the_model_changes(tmp_path):
    c = sqlite3.connect(tmp_path / "x.db")
    e = CountingEmbedder("m1")
    s = EmbeddingStore(e)
    s.ensure(c)
    texts = {"a": "paged attention", "b": "kv quantisation"}
    v1 = s.vectors(c, texts)
    c.commit()
    assert set(v1) == {"a", "b"} and len(e.calls) == 1
    v2 = s.vectors(c, texts)
    assert len(e.calls) == 1 and np.array_equal(v1["a"], v2["a"])    # read back, not recomputed
    s.vectors(c, {**texts, "c": "new"})
    assert e.calls[-1] == ["new"]                                      # only the missing row
    e2 = CountingEmbedder("m1")
    assert EmbeddingStore(e2).vectors(c, texts).keys() == texts.keys() and e2.calls == []
    e3 = CountingEmbedder("m2")                                        # another model: all again
    EmbeddingStore(e3).vectors(c, texts)
    assert e3.calls == [list(texts.values())]
    rows = c.execute("SELECT model, COUNT(*) FROM embeddings GROUP BY model ORDER BY model")
    assert rows.fetchall() == [("m1", 3), ("m2", 2)]
    s.forget(c, ["a"])
    assert c.execute("SELECT COUNT(*) FROM embeddings WHERE id='a'").fetchone()[0] == 0


def test_a_failing_embedder_degrades_to_lexical_with_one_warning(caplog):
    class Broken:
        name = "broken"

        def encode(self, texts):
            raise RuntimeError("no weights")

    c = sqlite3.connect(":memory:")
    s = EmbeddingStore(Broken())
    s.ensure(c)
    assert s.active
    assert s.vectors(c, {"a": "x"}) == {} and not s.active
    assert hybrid_search(s, c, "x", {"a": "x"}, {"a": 1.0}, k=5) == [("a", 1.0)]
    assert "lexical from here" in caplog.text


def test_hybrid_search_admits_a_row_no_word_matches():
    c = sqlite3.connect(":memory:")
    s = EmbeddingStore(HashEmbedder())
    s.ensure(c)
    texts = {"a": "int4 kv cache quantisation", "b": "speculative decoding", "c": "paged attention"}
    got = hybrid_search(s, c, "quantising kv", texts, lexical={}, k=3)
    assert [i for i, _ in got] == ["a"]                  # under the floor: not a candidate
    got = hybrid_search(s, c, "quantising kv", texts, lexical={"c": 2.0}, k=3)
    assert [i for i, _ in got] == ["c", "a"]             # a lexical hit always is one


def test_default_embedder_is_none_without_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setenv("HARNESS_EMBEDDINGS", "on")       # past the pytest guard
    assert default_embedder() is None
    assert resolve(DEFAULT) is None and resolve(None) is None
    e = HashEmbedder()
    assert resolve(e) is e


def test_default_embedder_never_loads_under_pytest_and_never_raises(monkeypatch):
    monkeypatch.setattr(emb, "_package_available", lambda: True)
    monkeypatch.delenv("HARNESS_EMBEDDINGS", raising=False)
    assert default_embedder() is None                    # PYTEST_CURRENT_TEST is set
    monkeypatch.setenv("HARNESS_EMBEDDINGS", "off")
    assert default_embedder() is None
    monkeypatch.setenv("HARNESS_EMBEDDINGS", "on")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(emb, "model_cached", lambda model=None: False)
    assert default_embedder() is None                    # offline and no cached weights
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    got = default_embedder()                             # constructed, not loaded
    assert isinstance(got, LocalEmbedder) and got.name == "st:all-MiniLM-L6-v2"
    assert emb._MODELS == {}

    def boom():
        raise OSError("boom")

    monkeypatch.setattr(emb, "_package_available", boom)
    assert default_embedder() is None
