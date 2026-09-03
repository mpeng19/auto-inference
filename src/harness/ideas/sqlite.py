"""Reference IdeaBankService: one SQLite file, FTS5 for search, and a claim
rule that hands each agent the record least like what is already in flight.

Why a bank at all: the first fleets were seeded with one-sentence hints and
produced one-line diffs -- fifteen scheduler constants, every one inside
noise, and two agents on the same constant. A record here is a mechanism with
its sources, targets and risks attached; the agent that claims it starts from
a design. And a claim is exclusive, so diversity is a property of the bank
rather than a hope about the agents.

**How a claim is chosen.** Among available records, the one whose text is
least similar (max Jaccard over word sets) to every text in `avoid` -- the
ideas live in the fleet and those already tried -- with a tie broken toward a
`scale` not in `live_scales`. Cheap, explainable, and good enough: the bank is
tens to hundreds of records, not millions. The reference fleet passes `avoid`
and not `live_scales`: an `Idea` does not carry the record's scale, so that
tie-break only fires for a caller that tracks it.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
from dataclasses import asdict

from ..contracts.ideas import BankStatus, IdeaRecord, Scale, content_id

SEEDS = pathlib.Path(__file__).parent / "seeds"

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS ideas(
  id TEXT PRIMARY KEY, title TEXT, mechanism TEXT, hypothesis TEXT,
  source TEXT, source_title TEXT, url TEXT, scale TEXT, targets TEXT,
  expected_gain TEXT, risks TEXT, prerequisites TEXT, tags TEXT,
  status TEXT, claimed_by TEXT, experiment_ids TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS ideas_status ON ideas(status);
CREATE VIRTUAL TABLE IF NOT EXISTS ideas_fts USING fts5(
  id UNINDEXED, title, mechanism, hypothesis, tags);
"""

_LIST_FIELDS = ("targets", "tags", "experiment_ids")
# Jaccard above which a record counts as "the same idea" as a live one.
# Two records about the same mechanism share most of their words (0.6-0.9);
# records that merely share the field's vocabulary sit at 0.1-0.3.
AVOID_CLOSENESS = 0.35


def default_bank_path() -> pathlib.Path:
    """Shared across sessions on purpose: the bank is knowledge, not run
    state. `HARNESS_HOME` moves it, as it moves the session store."""
    import os

    home = pathlib.Path(os.environ.get("HARNESS_HOME") or pathlib.Path.home() / ".auto-inference")
    return home / "ideas.db"


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()
            if len(w) > 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class SqliteIdeaBank:
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

    # ── writes ───────────────────────────────────────────────────────────
    def add(self, rec: IdeaRecord) -> str:
        """Insert or replace by id. A record with the same title under a
        different id (the random ids of banks filled before ids were
        content-addressed) is folded into this one: its status, claimant and
        experiment history carry over and the old row goes, so re-seeding an
        old bank is a migration rather than a duplication."""
        with self._lock, self._conn() as c:
            twin = c.execute("SELECT * FROM ideas WHERE lower(trim(title))=lower(trim(?)) AND id<>?",
                             (rec.title, rec.id)).fetchone()
            if twin is not None:
                old = _row(twin)
                rec = IdeaRecord(**{**asdict(rec), "status": old.status,
                                    "claimed_by": old.claimed_by,
                                    "experiment_ids": old.experiment_ids,
                                    "created_at": old.created_at})
                c.execute("DELETE FROM ideas WHERE id=?", (old.id,))
                c.execute("DELETE FROM ideas_fts WHERE id=?", (old.id,))
        d = asdict(rec)
        for k in _LIST_FIELDS:
            d[k] = json.dumps(list(d[k]))
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO ideas VALUES "
                "(:id,:title,:mechanism,:hypothesis,:source,:source_title,:url,"
                ":scale,:targets,:expected_gain,:risks,:prerequisites,:tags,"
                ":status,:claimed_by,:experiment_ids,:created_at)", d)
            c.execute("DELETE FROM ideas_fts WHERE id=?", (rec.id,))
            c.execute("INSERT INTO ideas_fts(id,title,mechanism,hypothesis,tags) "
                      "VALUES (?,?,?,?,?)",
                      (rec.id, rec.title, rec.mechanism, rec.hypothesis, " ".join(rec.tags)))
        return rec.id

    def release(self, rec_id: str, status: BankStatus = "available") -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE ideas SET status=?, claimed_by='' WHERE id=?",
                      (status, rec_id))

    def record_outcome(self, rec_id: str, experiment_id: str,
                       status: BankStatus = "tried") -> None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT experiment_ids FROM ideas WHERE id=?",
                            (rec_id,)).fetchone()
            if row is None:
                return
            ids = json.loads(row["experiment_ids"] or "[]")
            if experiment_id and experiment_id not in ids:
                ids.append(experiment_id)
            c.execute("UPDATE ideas SET experiment_ids=?, status=?, claimed_by='' "
                      "WHERE id=?", (json.dumps(ids), status, rec_id))

    # ── reads ────────────────────────────────────────────────────────────
    def get(self, rec_id: str) -> IdeaRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM ideas WHERE id=?", (rec_id,)).fetchone()
        return _row(row) if row else None

    def list(self, status: BankStatus | None = None,
             scale: Scale | None = None) -> tuple[IdeaRecord, ...]:
        q, args = "SELECT * FROM ideas", []
        cond = []
        if status:
            cond.append("status=?")
            args.append(status)
        if scale:
            cond.append("scale=?")
            args.append(scale)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY created_at, id"
        with self._conn() as c:
            return tuple(_row(r) for r in c.execute(q, args).fetchall())

    def search(self, text: str, k: int = 8) -> tuple[IdeaRecord, ...]:
        words = [w for w in _tokens(text)]
        if not words:
            return ()
        q = " OR ".join(f'"{w}"' for w in words)
        with self._conn() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM ideas_fts WHERE ideas_fts MATCH ? "
                "ORDER BY bm25(ideas_fts) LIMIT ?", (q, k)).fetchall()]
        return tuple(r for r in (self.get(i) for i in ids) if r is not None)

    def count(self, status: BankStatus | None = None) -> int:
        with self._conn() as c:
            if status:
                return c.execute("SELECT COUNT(*) FROM ideas WHERE status=?",
                                 (status,)).fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM ideas").fetchone()[0]

    def related(self, rec_id: str, k: int = 5) -> tuple[IdeaRecord, ...]:
        me = self.get(rec_id)
        if me is None:
            return ()
        mine = _tokens(me.text)
        scored = [(_jaccard(mine, _tokens(r.text)), r) for r in self.list() if r.id != rec_id]
        scored.sort(key=lambda t: (-t[0], t[1].created_at))
        return tuple(r for score, r in scored[:k] if score > 0.0)

    # ── the claim ────────────────────────────────────────────────────────
    def claim(self, agent_id: str, avoid: tuple[str, ...] = (),
              live_scales: tuple[str, ...] = (), seed: str = "") -> IdeaRecord | None:
        """Hand `agent_id` the available record least like `avoid`, or with
        a `seed`, the one most like the seed that is not close to `avoid`."""
        with self._lock:
            pool = self.list(status="available")
            if not pool:
                return None
            avoid_sets = [_tokens(t) for t in avoid if t]
            live = set(live_scales)

            def closest(r: IdeaRecord) -> float:
                return max((_jaccard(_tokens(r.text), a) for a in avoid_sets), default=0.0)

            if seed.strip():
                want = _tokens(seed)
                # "not close to avoid": a seed can steer, but never hand back
                # the idea already running or its twin. Records near what is
                # live are out; if that empties the pool, the seed is asking
                # for exactly what is live and the fallback is diversity.
                far = [r for r in pool if closest(r) < AVOID_CLOSENESS]
                if not far:
                    best = min(pool, key=lambda r: (closest(r), r.created_at))
                else:
                    best = max(far, key=lambda r: (_jaccard(want, _tokens(r.text)),
                                                   -closest(r), -r.created_at))
            else:
                best = min(pool, key=lambda r: (closest(r), r.scale in live, r.created_at))
            with self._conn() as c:
                c.execute("UPDATE ideas SET status='claimed', claimed_by=? WHERE id=?",
                          (agent_id, best.id))
            return self.get(best.id)

    # ── bulk ─────────────────────────────────────────────────────────────
    def seed(self, source: str = "book") -> int:
        """The packaged seed set (`seeds/<source>.jsonl`): the inference
        engineering book's 27 mechanisms, extracted once and committed, so a
        fresh machine has a bank before any model call."""
        p = SEEDS / f"{source}.jsonl"
        if not p.is_file():
            raise FileNotFoundError(f"no seed set {source!r}; have "
                                    + ", ".join(sorted(q.stem for q in SEEDS.glob("*.jsonl"))))
        return self.import_jsonl(p, source_default=source)

    def import_jsonl(self, path: str | pathlib.Path, source_default: str = "") -> int:
        """Load records written by an extractor. Unknown keys are ignored,
        missing ones default, so an extractor's schema can drift a little
        without the import refusing everything."""
        n = 0
        for line in pathlib.Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = record_from_dict(d, source_default=source_default)
            self.add(rec)
            n += 1
        return n


def record_from_dict(d: dict, source_default: str = "") -> IdeaRecord:
    fields = {f for f in IdeaRecord.__dataclass_fields__}
    clean = {k: v for k, v in d.items() if k in fields}
    for k in _LIST_FIELDS:
        v = clean.get(k)
        if isinstance(v, str):
            clean[k] = tuple(x.strip() for x in v.split(",") if x.strip())
        elif v is not None:
            clean[k] = tuple(str(x) for x in v)
    # A model asked for prose sometimes answers with a list; the column is
    # text either way.
    for k, v in list(clean.items()):
        if k not in _LIST_FIELDS and isinstance(v, (list, tuple)):
            clean[k] = "; ".join(str(x) for x in v)
        elif k not in _LIST_FIELDS and isinstance(v, dict):
            clean[k] = "; ".join(f"{a}: {b}" for a, b in v.items())
    if not clean.get("source") and source_default:
        clean["source"] = source_default
    if not clean.get("id"):
        clean["id"] = content_id(clean.get("title", ""), clean.get("mechanism", ""))
    scale = clean.get("scale", "kernel")
    if scale not in ("kernel", "architecture", "memory", "scheduler",
                     "parallelism", "numerics", "other"):
        clean["scale"] = "other"
    status = clean.get("status", "available")
    if status not in ("available", "claimed", "tried", "retired"):
        clean["status"] = "available"
    return IdeaRecord(**clean)


def _row(r: sqlite3.Row) -> IdeaRecord:
    d = dict(r)
    for k in _LIST_FIELDS:
        d[k] = tuple(json.loads(d.get(k) or "[]"))
    return IdeaRecord(**d)
