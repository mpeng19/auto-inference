"""Embeddings for hybrid search: lexical match plus a sentence model.

Every store in the harness (ideas, memory, skills) searched by words alone:
FTS5 BM25 for a query, Jaccard over word sets for "is this the same idea".
Words miss paraphrase -- "quantise the KV cache to four bits" and "int4 KV
with per-block scales" share nothing but "KV" -- and a fleet that cannot see
that hands two agents the same mechanism. This module adds the second
signal and the rule for blending the two.

    Embedder            protocol: `name` and `encode(texts) -> vectors`
    LocalEmbedder       sentence-transformers all-MiniLM-L6-v2, on CPU, lazy
    HashEmbedder        deterministic, dependency-free; tests and fallback
    default_embedder()  the local one when installed and usable, else None
    EmbeddingStore      the `embeddings` table beside each record table
    hybrid_rank         the blend: min-max cosine and min-max lexical, alpha
    hybrid_search       candidates from both signals, ranked by hybrid_rank

The model is optional: `uv sync --group embeddings` installs it, and without
it every path degrades to what it was (pure lexical). `default_embedder`
never raises and never downloads under pytest, so the suite stays offline.

**The blend.** Cosine over the candidates is min-max normalised to [0, 1],
BM25 (or Jaccard) over the same candidates likewise, and the score is
`alpha * cosine + (1 - alpha) * lexical` with alpha 0.5. Candidates are the
union of lexical hits and rows whose raw cosine clears `SEMANTIC_FLOOR`, so a
paraphrase with no shared words can still enter the ranking.

**Twins.** For "is this the same idea" (a claim's `avoid`, a fact's
contradiction check) the two signals are not blended but each held to its
own threshold: Jaccard at the caller's cutoff, cosine at `TWIN_COSINE`. A
record is a twin if either says so.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import pathlib
import sqlite3
from collections.abc import Iterable
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

LOCAL_MODEL = "all-MiniLM-L6-v2"
# Cosine below which a row is not a candidate on meaning alone. MiniLM puts
# unrelated engineering prose at 0.1-0.3 and same-topic prose at 0.4+; the
# hash embedder puts texts with no shared stem at exactly 0.
SEMANTIC_FLOOR = 0.3
# Cosine at which two texts are "the same idea": MiniLM paraphrases score
# 0.7-0.9, same-topic-different-mechanism 0.4-0.6.
TWIN_COSINE = 0.6

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings(
  id TEXT, model TEXT, vec BLOB, PRIMARY KEY (id, model));
"""


@runtime_checkable
class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class _Default:
    """Sentinel: "use `default_embedder()`", as distinct from `None` (lexical only)."""

    def __repr__(self) -> str:
        return "DEFAULT"


DEFAULT: Final = _Default()


def resolve(embedder: Embedder | _Default | None) -> Embedder | None:
    return default_embedder() if isinstance(embedder, _Default) else embedder


# ── embedders ────────────────────────────────────────────────────────────
_STOP = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "its", "of", "on", "or", "so", "than", "that", "the", "this", "to", "via", "with"])


class HashEmbedder:
    """Signed feature hashing of a bag of words into `dims` dimensions, L2
    normalised. Cosine between two texts is then (an unbiased estimate of)
    the cosine between their word bags -- lexical, but tolerant of
    inflection because every word longer than five characters also emits
    its five-character stem, so "quantise" and "quantisation" meet. No
    dependencies, deterministic across processes (blake2b, not `hash`),
    and fast enough to embed a bank in milliseconds; what tests use, and
    what a caller can pass when the model is not wanted."""

    def __init__(self, dims: int = 256, name: str = ""):
        self.dims = dims
        self.name = name or f"hash{dims}"

    @staticmethod
    def features(text: str) -> Iterable[str]:
        for w in "".join(c if c.isalnum() else " " for c in text.lower()).split():
            if len(w) < 2 or w in _STOP:
                continue
            yield w
            if len(w) > 5:
                yield "p:" + w[:5]

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.dims, dtype=np.float32)
            for f in self.features(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                v[h % self.dims] += 1.0 if h >> 63 else -1.0
            out.append(_unit(v).tolist())
        return out


_MODELS: dict[str, Any] = {}


class LocalEmbedder:
    """`sentence-transformers` on CPU. The model loads on first `encode`
    and is shared across instances in the process, so constructing a bank
    costs nothing and a fleet of ten agents loads MiniLM once."""

    def __init__(self, model: str = LOCAL_MODEL):
        self.model = model
        self.name = f"st:{model}"

    def _load(self) -> Any:
        m = _MODELS.get(self.model)
        if m is None:
            from sentence_transformers import SentenceTransformer

            m = _MODELS[self.model] = SentenceTransformer(self.model, device="cpu")
        return m

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False)
        return [[float(x) for x in row] for row in arr]


def _package_available() -> bool:
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):     # ValueError: sys.modules entry set to None
        return False


def model_cached(model: str = LOCAL_MODEL) -> bool:
    """Is the model already in the Hugging Face hub cache? The only way to
    know it is usable without opening a socket."""
    try:
        from huggingface_hub import constants

        root = pathlib.Path(constants.HF_HUB_CACHE)
    except Exception:
        home = os.environ.get("HF_HOME") or pathlib.Path.home() / ".cache" / "huggingface"
        root = pathlib.Path(home) / "hub"
    return (root / f"models--sentence-transformers--{model}").is_dir()


def default_embedder() -> Embedder | None:
    """The local model when it can be used, else None; never raises.

    `HARNESS_EMBEDDINGS=off` forces None; `=on` overrides the pytest guard.
    Under pytest the answer is None unless forced, so the suite is the same
    on a machine with the model installed as on one without, and never
    downloads. Outside pytest the model is returned when the package is
    importable and the weights are cached or may be fetched (no
    `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`); the fetch itself happens on
    the first `encode`, which the stores wrap so a failure degrades the
    search to lexical rather than failing a claim."""
    try:
        mode = os.environ.get("HARNESS_EMBEDDINGS", "").lower()
        if mode == "off":
            return None
        if "PYTEST_CURRENT_TEST" in os.environ and mode != "on":
            return None
        if not _package_available():
            return None
        offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        if offline and not model_cached():
            return None
        return LocalEmbedder()
    except Exception as e:              # a misconfigured install is not a fleet outage
        log.warning("no embedder: %s", e)
        return None


# ── vectors ──────────────────────────────────────────────────────────────
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi > lo:
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    return {k: (1.0 if v > 0 else 0.0) for k, v in scores.items()}


def cosine_scores(query: np.ndarray, vectors: dict[str, np.ndarray]) -> dict[str, float]:
    """Cosine of `query` against each unit vector. A few hundred rows is one
    matrix-vector product."""
    if not vectors:
        return {}
    ids = list(vectors)
    m = np.stack([vectors[i] for i in ids])
    return dict(zip(ids, (m @ query).astype(float).tolist(), strict=True))


class EmbeddingStore:
    """The `embeddings(id, model, vec)` table beside a record table.

    Filled on add and lazily for rows missing a vector under the current
    model, so switching embedders re-embeds without a migration and a bank
    written before this module existed embeds itself on first search. An
    embedder that fails (weights not downloadable, say) is dropped for the
    rest of the process with one warning; everything then runs lexical.

    Connection handling stays with the owner: methods take the connection
    and the owner commits."""

    def __init__(self, embedder: Embedder | None):
        self.embedder = embedder
        self._broken = False

    @property
    def active(self) -> bool:
        return self.embedder is not None and not self._broken

    @property
    def model(self) -> str:
        return self.embedder.name if self.embedder is not None else ""

    def ensure(self, c: sqlite3.Connection) -> None:
        c.executescript(SCHEMA)

    def forget(self, c: sqlite3.Connection, ids: Iterable[str]) -> None:
        c.executemany("DELETE FROM embeddings WHERE id=?", [(i,) for i in ids])

    def _encode(self, texts: list[str]) -> list[list[float]] | None:
        if not self.active:
            return None
        try:
            return self.embedder.encode(texts)      # type: ignore[union-attr]
        except Exception as e:
            log.warning("embedder %s failed (%s); search is lexical from here",
                        self.model, e)
            self._broken = True
            return None

    def query(self, text: str) -> np.ndarray | None:
        """Embed a query (not stored)."""
        vecs = self.query_many([text])
        return vecs[0] if vecs else None

    def query_many(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vecs = self._encode(texts)
        if vecs is None:
            return []
        return [_unit(np.asarray(v, dtype=np.float32)) for v in vecs]

    def vectors(self, c: sqlite3.Connection, texts: dict[str, str]) -> dict[str, np.ndarray]:
        """Unit vectors for `texts` (id -> text) under the current model,
        embedding and storing whichever are missing."""
        if not self.active or not texts:
            return {}
        have: dict[str, np.ndarray] = {}
        ids = list(texts)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            for r in c.execute(
                    f"SELECT id, vec FROM embeddings WHERE model=? AND id IN ({marks})",
                    (self.model, *chunk)):
                have[r[0]] = np.frombuffer(r[1], dtype=np.float32)
        missing = [i for i in ids if i not in have]
        if missing:
            vecs = self._encode([texts[i] for i in missing])
            if vecs is not None:
                rows = []
                for i, v in zip(missing, vecs, strict=True):
                    have[i] = _unit(np.asarray(v, dtype=np.float32))
                    rows.append((i, self.model, have[i].tobytes()))
                c.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?)", rows)
        return have


# ── ranking ──────────────────────────────────────────────────────────────
def hybrid_rank(query: str, candidates: list[tuple[str, str]], embedder: Embedder | None,
                lexical_scores: dict[str, float], alpha: float = 0.5, *,
                vectors: dict[str, np.ndarray] | None = None,
                query_vec: np.ndarray | None = None) -> list[tuple[str, float]]:
    """Rank `candidates` (id, text) by `alpha` * cosine + (1 - alpha) *
    lexical, each min-max normalised over the candidates. `lexical_scores`
    are higher-is-better (negate BM25 first); a missing id scores 0. With
    no embedder the result is the lexical scores as given, sorted, so the
    lexical-only path is exactly what it was before embeddings existed.

    `vectors` and `query_vec` are precomputed unit vectors (from an
    `EmbeddingStore`); whatever they do not cover is encoded here."""
    ids = [i for i, _ in candidates]
    order = {i: n for n, i in enumerate(ids)}
    lex_raw = {i: float(lexical_scores.get(i, 0.0)) for i in ids}
    if embedder is None or not ids:
        return sorted(lex_raw.items(), key=lambda t: (-t[1], order[t[0]]))

    vecs = dict(vectors or {})
    missing = [(i, t) for i, t in candidates if i not in vecs]
    to_encode = ([] if query_vec is not None else [query]) + [t for _, t in missing]
    if to_encode:
        enc = [_unit(np.asarray(v, dtype=np.float32)) for v in embedder.encode(to_encode)]
        if query_vec is None:
            query_vec, enc = enc[0], enc[1:]
        for (i, _), v in zip(missing, enc, strict=True):
            vecs[i] = v
    sem = _minmax(cosine_scores(query_vec, {i: vecs[i] for i in ids}))  # type: ignore[arg-type]
    lex = _minmax(lex_raw)
    blended = {i: alpha * sem[i] + (1.0 - alpha) * lex[i] for i in ids}
    return sorted(blended.items(), key=lambda t: (-t[1], order[t[0]]))


def hybrid_search(store: EmbeddingStore, c: sqlite3.Connection, query: str,
                  texts: dict[str, str], lexical: dict[str, float], k: int,
                  alpha: float = 0.5, floor: float = SEMANTIC_FLOOR,
                  width: int | None = None) -> list[tuple[str, float]]:
    """The whole hybrid read. `texts` is every row that may be returned (id
    -> what its vector embeds); `lexical` the lexical hits among them.
    Candidates are the lexical hits plus the `width` rows (default 4k, at
    least 40) nearest the query with cosine at or above `floor`; the top
    `k` by `hybrid_rank` come back. Without an active embedder this is
    the lexical hits sorted, unchanged."""
    width = width or max(4 * k, 40)
    lexical = {i: s for i, s in lexical.items() if i in texts}
    qv = store.query(query) if store.active else None
    if qv is None:
        ranked = hybrid_rank(query, [(i, texts[i]) for i in lexical], None, lexical)
        return ranked[:k]
    vectors = store.vectors(c, texts)
    cos = cosine_scores(qv, vectors)
    near = sorted((i for i, s in cos.items() if s >= floor), key=lambda i: -cos[i])[:width]
    cand_ids = list(lexical) + [i for i in near if i not in lexical]
    ranked = hybrid_rank(query, [(i, texts[i]) for i in cand_ids], store.embedder, lexical,
                         alpha, vectors=vectors, query_vec=qv)
    return ranked[:k]


def max_cosine(vec: np.ndarray | None, against: list[np.ndarray]) -> float:
    """Largest cosine of `vec` against a list; 0 when either side is empty."""
    if vec is None or not against:
        return 0.0
    return max(float(vec @ a) for a in against)
