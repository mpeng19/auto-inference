from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY, pid INTEGER, tid TEXT, name TEXT, kind TEXT);
CREATE TABLE IF NOT EXISTS names(
  id INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS spans(
  id INTEGER PRIMARY KEY, name_id INTEGER, track_id INTEGER,
  ts REAL, dur REAL, cat TEXT, corr INTEGER);
CREATE INDEX IF NOT EXISTS spans_track_ts ON spans(track_id, ts);
CREATE INDEX IF NOT EXISTS spans_name_ts ON spans(name_id, ts);
CREATE TABLE IF NOT EXISTS steps(
  id INTEGER PRIMARY KEY, idx INTEGER, ts REAL, dur REAL);
"""


class TraceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._names: dict[str, int] = {r["name"]: r["id"] for r in self.conn.execute("SELECT * FROM names")}
        self._tracks: dict[tuple, int] = {}

    def name_id(self, name: str) -> int:
        nid = self._names.get(name)
        if nid is None:
            cur = self.conn.execute("INSERT INTO names(name) VALUES (?)", (name,))
            nid = cur.lastrowid
            self._names[name] = nid
        return nid

    def track_id(self, pid, tid, name: str = "", kind: str = "") -> int:
        key = (pid, str(tid))
        t = self._tracks.get(key)
        if t is None:
            cur = self.conn.execute("INSERT INTO tracks(pid, tid, name, kind) VALUES (?,?,?,?)",
                                    (pid, str(tid), name, kind))
            t = cur.lastrowid
            self._tracks[key] = t
        return t

    def set_track_label(self, pid, tid, name: str | None = None, kind: str | None = None) -> None:
        t = self.track_id(pid, tid)
        if name:
            self.conn.execute("UPDATE tracks SET name=? WHERE id=? AND (name='' OR name IS NULL)", (name, t))
        if kind:
            self.conn.execute("UPDATE tracks SET kind=? WHERE id=?", (kind, t))

    def add_spans(self, rows: list[tuple]) -> None:
        # rows: (name_id, track_id, ts, dur, cat, corr)
        self.conn.executemany("INSERT INTO spans(name_id, track_id, ts, dur, cat, corr) VALUES (?,?,?,?,?,?)", rows)

    def finalize(self, meta: dict) -> None:
        for k, v in meta.items():
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (k, json.dumps(v)))
        self.conn.commit()

    def meta(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self.conn.execute("SELECT * FROM meta")}
