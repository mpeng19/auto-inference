"""Reference MemoryService: SQLite, a typed edge table, FTS, and a brief.

Chosen because the write rate is ~20 rows an hour across the whole fleet, so
the interesting engineering is entirely in the read. SQLite gives transactions,
recursive CTEs for lineage, and FTS5 for free, in one file with no service to
operate. When something better exists it replaces this class and nothing else.

**How a read is scored.** Three signals, combined, because none alone works:

    text       FTS5 over hypothesis + summary, blended with cosine over a
               sentence embedding of the same text when an embedder is
               present (`harness.embeddings`; lexical alone otherwise)
    graph      distance from the asking agent's own lineage
    recency    a half-life, so a stale timeline decays out of the brief

Pure lexical retrieval is what `agent-db` measured as a **clean null**
out-of-sample. The graph term is the part that is supposed to beat it: an
experiment two edges from what you are about to try is relevant even when it
shares no vocabulary with your query. The embedding closes the same gap from
the other side: a hypothesis phrased differently from your intent still
surfaces, marked "near your intent" in the audit trail.

**Negative results are boosted, not filtered.** The expensive knowledge in a
research loop is what already failed and why. A retrieval tuned for wins would
let ten agents rediscover the same dead end.

**The brief is the product.** `recall` returns prose with the hits attached as
an audit trail, because the one condition that beat placebo was a synthesised
state of knowledge rather than a pile of retrieved facts.
"""
from __future__ import annotations

import json
import math
import pathlib
import sqlite3
import threading
import time
from dataclasses import asdict

from ..contracts.common import Provenance
from ..contracts.memory import Brief, Experiment, Finding, Hit, Recall, Relation
from ..embeddings import DEFAULT, Embedder, EmbeddingStore, hybrid_search, resolve

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS experiments(
  id TEXT PRIMARY KEY, agent_id TEXT, idea_id TEXT, timeline TEXT, ts REAL,
  hypothesis TEXT, rationale TEXT, stack_digest TEXT, eval_digest TEXT,
  verdict TEXT, metrics TEXT, baseline_metrics TEXT, summary TEXT,
  tags TEXT, provenance TEXT, trace_ref TEXT);
CREATE INDEX IF NOT EXISTS exp_idea ON experiments(idea_id);
CREATE INDEX IF NOT EXISTS exp_tl   ON experiments(timeline, ts);
CREATE INDEX IF NOT EXISTS exp_stack ON experiments(stack_digest);

-- Typed edges. The whole reason this is a graph and not a log.
CREATE TABLE IF NOT EXISTS relations(
  src TEXT, dst TEXT, kind TEXT, note TEXT, ts REAL,
  PRIMARY KEY (src, dst, kind));
CREATE INDEX IF NOT EXISTS rel_dst ON relations(dst);

CREATE TABLE IF NOT EXISTS findings(
  id TEXT PRIMARY KEY, claim TEXT, kind TEXT, evidence TEXT,
  confidence REAL, provenance TEXT, superseded_by TEXT DEFAULT '');

CREATE VIRTUAL TABLE IF NOT EXISTS exp_fts USING fts5(
  hypothesis, summary, content='experiments', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS exp_ai AFTER INSERT ON experiments BEGIN
  INSERT INTO exp_fts(rowid, hypothesis, summary)
  VALUES (new.rowid, new.hypothesis, new.summary); END;
CREATE TRIGGER IF NOT EXISTS exp_au AFTER UPDATE ON experiments BEGIN
  INSERT INTO exp_fts(exp_fts, rowid, hypothesis, summary)
  VALUES('delete', old.rowid, old.hypothesis, old.summary);
  INSERT INTO exp_fts(rowid, hypothesis, summary)
  VALUES (new.rowid, new.hypothesis, new.summary); END;

CREATE VIRTUAL TABLE IF NOT EXISTS fnd_fts USING fts5(
  claim, content='findings', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS fnd_ai AFTER INSERT ON findings BEGIN
  INSERT INTO fnd_fts(rowid, claim) VALUES (new.rowid, new.claim); END;
"""

HALF_LIFE_S = 7 * 24 * 3600      # a week: long enough for a slow sweep to matter
W_LEXICAL, W_GRAPH, W_RECENCY = 1.0, 1.2, 0.4
NEGATIVE_BOOST = 1.25


def _fts_query(text: str) -> str:
    """FTS5 MATCH from free text, tolerant of punctuation an agent will write."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in text).split()
             if len(w) > 2]
    return " OR ".join(sorted(set(words))[:24])


def _embed_text(hypothesis: str, summary: str) -> str:
    """What an experiment's vector embeds: the same two fields FTS indexes."""
    return f"{hypothesis} {summary}".strip()


# Rows considered for the semantic half of a read: the newest ones. A fleet
# writes ~20 an hour, so this is weeks of history in one matrix product.
SEMANTIC_ROWS = 2000


class SqliteMemory:
    """Reference implementation of `contracts.memory.MemoryService`."""

    def __init__(self, path: str | pathlib.Path = "memory.db",
                 embedder: Embedder | None = DEFAULT):
        """`embedder` defaults to `embeddings.default_embedder()`; pass
        `None` for a purely lexical read."""
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.embeddings = EmbeddingStore(resolve(embedder))
        self._conn().executescript(SCHEMA)
        self.embeddings.ensure(self._conn())
        self._conn().commit()

    @property
    def embedder(self) -> Embedder | None:
        return self.embeddings.embedder

    def _conn(self) -> sqlite3.Connection:
        """One connection per thread.

        Ten agent threads sharing a single connection raises "bad parameter or
        other API misuse" the moment two of them query at once -- observed on a
        live fleet, where it surfaced as an agent failing to propose an idea.
        A lock would also work; per-thread connections keep concurrent reads
        actually concurrent, which is the point of WAL.
        """
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=15)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=15000")
            self._local.c = c
        return c

    @property
    def _c(self) -> sqlite3.Connection:
        return self._conn()

    # ── write ────────────────────────────────────────────────────────────
    def record(self, exp: Experiment) -> str:
        self._c.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (exp.id, exp.agent_id, exp.idea_id, exp.timeline, exp.ts,
             exp.hypothesis, exp.rationale, exp.stack_digest, exp.eval_digest,
             exp.verdict, json.dumps(exp.metrics), json.dumps(exp.baseline_metrics),
             exp.summary, json.dumps(list(exp.tags)),
             json.dumps(asdict(exp.provenance)), exp.trace_ref))
        self.embeddings.forget(self._c, [exp.id])
        self.embeddings.vectors(self._c, {exp.id: _embed_text(exp.hypothesis, exp.summary)})
        self._c.commit()
        return exp.id

    def relate(self, rel: Relation) -> None:
        self._c.execute("INSERT OR REPLACE INTO relations VALUES (?,?,?,?,?)",
                        (rel.src, rel.dst, rel.kind, rel.note, rel.ts))
        self._c.commit()

    def assert_finding(self, f: Finding) -> str:
        self._c.execute("INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?)",
                        (f.id, f.claim, f.kind, json.dumps(list(f.evidence)),
                         f.confidence, json.dumps(asdict(f.provenance)),
                         f.superseded_by))
        self._c.commit()
        return f.id

    def supersede(self, finding_id: str, by: str) -> None:
        self._c.execute("UPDATE findings SET superseded_by=? WHERE id=?",
                        (by, finding_id))
        self._c.commit()

    # ── read ─────────────────────────────────────────────────────────────
    def _row_to_exp(self, r: sqlite3.Row) -> Experiment:
        return Experiment(
            id=r["id"], agent_id=r["agent_id"], idea_id=r["idea_id"],
            timeline=r["timeline"], ts=r["ts"], hypothesis=r["hypothesis"],
            rationale=r["rationale"], stack_digest=r["stack_digest"],
            eval_digest=r["eval_digest"], verdict=r["verdict"],
            metrics=json.loads(r["metrics"] or "{}"),
            baseline_metrics=json.loads(r["baseline_metrics"] or "{}"),
            summary=r["summary"], tags=tuple(json.loads(r["tags"] or "[]")),
            provenance=Provenance(**json.loads(r["provenance"] or "{}")),
            trace_ref=r["trace_ref"])

    def lineage(self, experiment_id: str, depth: int = 4) -> tuple[Experiment, ...]:
        """Ancestors and descendants, both directions, breadth-first."""
        sql = """
        WITH RECURSIVE walk(id, d) AS (
          SELECT ?, 0
          UNION
          SELECT r.dst, w.d + 1 FROM relations r JOIN walk w ON r.src = w.id
            WHERE w.d < ?
          UNION
          SELECT r.src, w.d + 1 FROM relations r JOIN walk w ON r.dst = w.id
            WHERE w.d < ?
        )
        SELECT e.*, MIN(w.d) AS dist FROM walk w JOIN experiments e ON e.id = w.id
        WHERE e.id != ? GROUP BY e.id ORDER BY dist, e.ts DESC
        """
        rows = self._c.execute(sql, (experiment_id, depth, depth, experiment_id))
        return tuple(self._row_to_exp(r) for r in rows)

    def _graph_distances(self, q: Recall, depth: int = 3) -> dict[str, int]:
        """How far each experiment is from what this agent has already done."""
        seeds = [r["id"] for r in self._c.execute(
            "SELECT id FROM experiments WHERE (idea_id=? AND ?!='') "
            "OR (agent_id=? AND ?!='') ORDER BY ts DESC LIMIT 20",
            (q.idea_id, q.idea_id, q.agent_id, q.agent_id))]
        dist: dict[str, int] = {s: 0 for s in seeds}
        frontier = list(seeds)
        for d in range(1, depth + 1):
            if not frontier:
                break
            marks = ",".join("?" * len(frontier))
            nxt = [r[0] for r in self._c.execute(
                f"SELECT dst FROM relations WHERE src IN ({marks}) "
                f"UNION SELECT src FROM relations WHERE dst IN ({marks})",
                frontier + frontier)]
            frontier = [n for n in nxt if n not in dist]
            for n in frontier:
                dist[n] = d
        return dist

    def recall(self, q: Recall) -> Brief:
        now = time.time()
        dist = self._graph_distances(q)
        cand: dict[str, tuple[Experiment, float, list[str]]] = {}

        width = max(q.k * 4, 40)
        match = _fts_query(q.intent)
        lexical: dict[str, float] = {}
        if match:
            rows = self._c.execute(
                "SELECT e.id, bm25(exp_fts) AS rank FROM exp_fts "
                "JOIN experiments e ON e.rowid = exp_fts.rowid "
                "WHERE exp_fts MATCH ? ORDER BY rank LIMIT ?", (match, width))
            for r in rows:
                # bm25 returns more-negative for better; map to (0, 1].
                lexical[r["id"]] = 1.0 / (1.0 + math.exp(min(6.0, max(-6.0, r["rank"]))))
        if self.embeddings.active:
            texts = {r["id"]: _embed_text(r["hypothesis"], r["summary"]) for r in self._c.execute(
                "SELECT id, hypothesis, summary FROM experiments ORDER BY ts DESC LIMIT ?",
                (SEMANTIC_ROWS,))}
        else:
            texts = dict.fromkeys(lexical, "")
        for exp_id, score in hybrid_search(self.embeddings, self._c, q.intent, texts,
                                           lexical, k=width):
            row = self._c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
            if row is None:
                continue
            why = "matches your intent" if exp_id in lexical else "near your intent"
            cand[exp_id] = (self._row_to_exp(row), W_LEXICAL * score, [why])
        self._c.commit()                 # vectors embedded lazily above

        # Graph neighbours of the agent's own work, whether or not they match.
        for exp_id, d in dist.items():
            row = self._c.execute("SELECT * FROM experiments WHERE id=?",
                                  (exp_id,)).fetchone()
            if row is None:
                continue
            e = self._row_to_exp(row)
            bump = W_GRAPH / (1.0 + d)
            if e.id in cand:
                prev = cand[e.id]
                cand[e.id] = (e, prev[1] + bump, [*prev[2], f"{d} edge(s) from your work"])
            else:
                cand[e.id] = (e, bump, [f"{d} edge(s) from your work"])

        hits: list[Hit] = []
        for e, score, why in cand.values():
            if q.since and e.ts < q.since:
                continue
            if e.verdict == "loss":
                if not q.include_negative:
                    continue
                score *= NEGATIVE_BOOST      # failures are the costly knowledge
            score += W_RECENCY * 0.5 ** ((now - e.ts) / HALF_LIFE_S)
            hits.append(Hit(experiment=e, score=round(score, 4),
                            why="; ".join(why),
                            lineage=tuple(sorted(dist)[:3])))
        hits.sort(key=lambda h: -h.score)
        hits = hits[:q.k]

        fnd = self._live_findings(match, limit=q.k)
        text, est = self._compose(q, hits, fnd)
        return Brief(text=text, hits=tuple(hits), findings=tuple(fnd),
                     open_questions=self._open_questions(hits), est_tokens=est)

    def _live_findings(self, match: str, limit: int) -> list[Finding]:
        if match:
            rows = self._c.execute(
                "SELECT f.* FROM fnd_fts JOIN findings f ON f.rowid = fnd_fts.rowid "
                "WHERE fnd_fts MATCH ? AND f.superseded_by='' "
                "ORDER BY bm25(fnd_fts) LIMIT ?", (match, limit))
        else:
            rows = self._c.execute(
                "SELECT * FROM findings WHERE superseded_by='' "
                "ORDER BY confidence DESC LIMIT ?", (limit,))
        return [Finding(id=r["id"], claim=r["claim"], kind=r["kind"],
                        evidence=tuple(json.loads(r["evidence"] or "[]")),
                        confidence=r["confidence"],
                        provenance=Provenance(**json.loads(r["provenance"] or "{}")),
                        superseded_by=r["superseded_by"]) for r in rows]

    @staticmethod
    def _open_questions(hits: list[Hit]) -> tuple[str, ...]:
        return tuple(h.experiment.hypothesis for h in hits
                     if h.experiment.verdict == "pending")[:5]

    def _compose(self, q: Recall, hits: list[Hit], fnd: list[Finding]) -> tuple[str, int]:
        """Prose, not a result list. Ordered so the costly knowledge comes first."""
        L = [f"State of knowledge for: {q.intent}", ""]
        losses = [h for h in hits if h.experiment.verdict == "loss"]
        wins = [h for h in hits if h.experiment.verdict == "win"]
        if fnd:
            L.append("Established:")
            L += [f"  - {f.claim}  (confidence {f.confidence:.2f},"
                  f" from {len(f.evidence)} experiment(s))" for f in fnd]
            L.append("")
        if losses:
            L.append("Already tried and did NOT work:")
            L += [f"  - {h.experiment.hypothesis} -> {h.experiment.summary}"
                  f"  [{h.experiment.id}, {h.why}]" for h in losses]
            L.append("")
        if wins:
            L.append("Known to work:")
            L += [f"  - {h.experiment.hypothesis} -> {h.experiment.summary}"
                  f"  [{h.experiment.id}]" for h in wins]
            L.append("")
        if not (fnd or hits):
            L.append("Nothing on record. You are first here -- say so in your "
                     "write-up, because a null result is worth recording.")
        text = "\n".join(L)
        # ~4 chars per token is close enough to enforce a budget with.
        while len(text) // 4 > q.max_tokens and len(L) > 3:
            L.pop()
            text = "\n".join(L)
        return text, len(text) // 4

    # ── maintenance ──────────────────────────────────────────────────────
    def prune_stale(self, current_stack: str = "", current_eval: str = "") -> int:
        n = 0
        for r in self._c.execute("SELECT * FROM findings WHERE superseded_by=''"):
            p = Provenance(**json.loads(r["provenance"] or "{}"))
            if p.is_stale(current_stack, current_eval):
                self.supersede(r["id"], by="stale:provenance")
                n += 1
        return n
