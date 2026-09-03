"""Reference SkillBankService: SQLite plus FTS, and a supersede chain.

Small by design: tens to a few hundred facts. The interesting part is the
contradiction step in `add`, which takes a judge so the bank itself never
has to decide what "contradicts" means.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import threading
from dataclasses import asdict

from ..contracts.skills import Fact, FactStatus, Judge

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS facts(
  id TEXT PRIMARY KEY, claim TEXT, topic TEXT, evidence TEXT, source TEXT,
  confidence REAL, tags TEXT, status TEXT, superseded_by TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS facts_topic ON facts(topic, status);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(id UNINDEXED, claim, topic, evidence, tags);
"""


def default_skills_path() -> pathlib.Path:
    home = pathlib.Path(os.environ.get("HARNESS_HOME") or pathlib.Path.home() / ".auto-inference")
    return home / "skills.db"


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()
            if len(w) > 3}


def lexical_judge(new: Fact, existing: tuple[Fact, ...]) -> tuple[str, ...]:
    """The fallback judge: same topic and high word overlap means the new
    fact is a restatement or a revision of the old one, so the old one
    yields. Crude, and says so; the manager passes a model instead."""
    a = _tokens(new.claim)
    out = []
    for f in existing:
        b = _tokens(f.claim)
        if a and b and len(a & b) / len(a | b) >= 0.5:
            out.append(f.id)
    return tuple(out)


class SqliteSkillBank:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=15000")
            self._local.conn = c
        return c

    def add(self, fact: Fact, judge: Judge | None = None) -> tuple[str, tuple[str, ...]]:
        with self._lock:
            same_topic = self.list(topic=fact.topic) if fact.topic else ()
            losers = tuple((judge or lexical_judge)(fact, same_topic)) if same_topic else ()
            d = asdict(fact)
            d["tags"] = json.dumps(list(fact.tags))
            with self._conn() as c:
                c.execute("INSERT OR REPLACE INTO facts VALUES "
                          "(:id,:claim,:topic,:evidence,:source,:confidence,:tags,"
                          ":status,:superseded_by,:created_at)", d)
                c.execute("DELETE FROM facts_fts WHERE id=?", (fact.id,))
                c.execute("INSERT INTO facts_fts(id,claim,topic,evidence,tags) VALUES (?,?,?,?,?)",
                          (fact.id, fact.claim, fact.topic, fact.evidence, " ".join(fact.tags)))
            for old in losers:
                if old != fact.id:
                    self.supersede(old, fact.id)
            return fact.id, tuple(x for x in losers if x != fact.id)

    def get(self, fact_id: str) -> Fact | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        return _row(r) if r else None

    def list(self, topic: str = "", status: FactStatus | None = "active") -> tuple[Fact, ...]:
        q, args = "SELECT * FROM facts", []
        cond = []
        if topic:
            cond.append("topic=?")
            args.append(topic)
        if status:
            cond.append("status=?")
            args.append(status)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY created_at"
        with self._conn() as c:
            return tuple(_row(r) for r in c.execute(q, args).fetchall())

    def search(self, text: str, k: int = 8) -> tuple[Fact, ...]:
        words = _tokens(text)
        if not words:
            return ()
        q = " OR ".join(f'"{w}"' for w in words)
        with self._conn() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts) LIMIT ?",
                (q, k * 2)).fetchall()]
        out = [f for f in (self.get(i) for i in ids) if f and f.status == "active"]
        return tuple(out[:k])

    def supersede(self, old_id: str, by: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE facts SET status='superseded', superseded_by=? WHERE id=?",
                      (by, old_id))

    def retract(self, fact_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE facts SET status='retracted' WHERE id=?", (fact_id,))

    def render(self, k: int = 40, query: str = "") -> str:
        facts = self.search(query, k) if query else self.list()[-k:]
        if not facts:
            return ""
        by_topic: dict[str, list[Fact]] = {}
        for f in facts:
            by_topic.setdefault(f.topic or "general", []).append(f)
        lines = ["---", "name: serving-facts",
                 "description: What earlier runs on this machine established about serving "
                 "this model on this hardware. Read before designing; cite the fact id when "
                 "you rely on one.", "---", "",
                 "# Serving facts, from earlier runs", "",
                 "Each fact carries the evidence it rests on and the writer's confidence. "
                 "A fact that later evidence contradicted is not here; its successor is.", ""]
        for topic, fs in sorted(by_topic.items()):
            lines.append(f"## {topic}")
            for f in fs:
                lines.append(f"- **{f.claim}**  ({f.id}, confidence {f.confidence:.1f}, from {f.source or '?'})")
                if f.evidence:
                    lines.append(f"  evidence: {f.evidence}")
            lines.append("")
        return "\n".join(lines)


def _row(r: sqlite3.Row) -> Fact:
    d = dict(r)
    d["tags"] = tuple(json.loads(d.get("tags") or "[]"))
    return Fact(**d)
