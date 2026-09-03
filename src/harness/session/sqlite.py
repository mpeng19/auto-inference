"""Reference SessionStore: one SQLite file, WAL mode, two processes.

WAL is the whole reason this works. A detached fleet writes a snapshot every
tick while a TUI reads at 4 Hz from another process, and in WAL mode readers
never block the writer and vice versa. Anything fancier would be solving a
problem that does not exist at this scale -- the fleet writes ~1 row/second and
a watcher reads a handful.

The snapshot is stored whole, as JSON, deliberately. A watcher wants *one
consistent picture* of the fleet, and reconstructing that from normalised
tables invites showing agent `a03` from one tick beside `a07` from the next.
Commands are a real table because they need durability and acknowledgement.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import threading
import time
from dataclasses import asdict

from ..contracts.session import AgentView, Command, SessionView, TokenUse

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, started_at REAL, updated_at REAL, phase TEXT,
  snapshot TEXT);
CREATE INDEX IF NOT EXISTS sess_started ON sessions(started_at DESC);

-- Attributed to an agent, always. "The fleet spent $200" is not actionable.
CREATE TABLE IF NOT EXISTS tokens(
  session_id TEXT, agent_id TEXT,
  input INT DEFAULT 0, output INT DEFAULT 0,
  cache_read INT DEFAULT 0, cache_write INT DEFAULT 0,
  PRIMARY KEY (session_id, agent_id));

CREATE TABLE IF NOT EXISTS commands(
  id TEXT PRIMARY KEY, session_id TEXT, kind TEXT, agent_id TEXT, value TEXT,
  issued_at REAL, applied_at REAL DEFAULT 0, result TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS cmd_pending ON commands(session_id, applied_at);
"""


def default_store_path() -> pathlib.Path:
    """Where a fleet and a TUI find each other with no configuration."""
    root = os.environ.get("HARNESS_HOME")
    base = pathlib.Path(root) if root else pathlib.Path.home() / ".auto-inference"
    base.mkdir(parents=True, exist_ok=True)
    return base / "sessions.db"


class SqliteSessionStore:
    """Reference implementation of `contracts.session.SessionStore`."""

    def __init__(self, path: str | pathlib.Path | None = None):
        self.path = pathlib.Path(path) if path else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)
        self._conn().commit()

    def _conn(self) -> sqlite3.Connection:
        """One connection per thread; see `memory.sqlite` for why."""
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

    # ── fleet side ───────────────────────────────────────────────────────
    def create(self, view: SessionView) -> str:
        self._c.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)",
            (view.session_id, view.started_at or time.time(), time.time(),
             view.phase, json.dumps(_dump(view))))
        self._c.commit()
        return view.session_id

    def publish(self, view: SessionView) -> None:
        self._c.execute(
            "UPDATE sessions SET updated_at=?, phase=?, snapshot=? WHERE id=?",
            (time.time(), view.phase, json.dumps(_dump(view)), view.session_id))
        self._c.commit()

    def add_tokens(self, session_id: str, agent_id: str, use: TokenUse) -> None:
        self._c.execute(
            "INSERT INTO tokens(session_id, agent_id, input, output, cache_read,"
            " cache_write) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(session_id, agent_id) DO UPDATE SET "
            " input=input+excluded.input, output=output+excluded.output,"
            " cache_read=cache_read+excluded.cache_read,"
            " cache_write=cache_write+excluded.cache_write",
            (session_id, agent_id, use.input, use.output, use.cache_read,
             use.cache_write))
        self._c.commit()

    def tokens(self, session_id: str) -> dict[str, TokenUse]:
        rows = self._c.execute("SELECT * FROM tokens WHERE session_id=?", (session_id,))
        return {r["agent_id"]: TokenUse(r["input"], r["output"], r["cache_read"],
                                        r["cache_write"]) for r in rows}

    def take_commands(self, session_id: str) -> tuple[Command, ...]:
        rows = self._c.execute(
            "SELECT * FROM commands WHERE session_id=? AND applied_at=0 "
            "ORDER BY issued_at", (session_id,))
        return tuple(_cmd(r) for r in rows)

    def acknowledge(self, command_id: str, result: str = "") -> None:
        self._c.execute("UPDATE commands SET applied_at=?, result=? WHERE id=?",
                        (time.time(), result, command_id))
        self._c.commit()

    # ── watcher side ─────────────────────────────────────────────────────
    def read(self, session_id: str = "") -> SessionView | None:
        if session_id:
            r = self._c.execute("SELECT snapshot FROM sessions WHERE id=?",
                                (session_id,)).fetchone()
        else:
            r = self._c.execute(
                "SELECT snapshot FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return _load(json.loads(r["snapshot"])) if r else None

    def sessions(self, limit: int = 20) -> tuple[SessionView, ...]:
        rows = self._c.execute(
            "SELECT snapshot FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,))
        return tuple(_load(json.loads(r["snapshot"])) for r in rows)

    def send(self, cmd: Command) -> str:
        """Issue a command to the most recent session. `send_to` names one."""
        self._c.execute(
            "INSERT OR REPLACE INTO commands VALUES (?,?,?,?,?,?,?,?)",
            (cmd.id, self._latest_session(), cmd.kind, cmd.agent_id, cmd.value,
             cmd.issued_at, cmd.applied_at, cmd.result))
        self._c.commit()
        return cmd.id

    def send_to(self, session_id: str, cmd: Command) -> str:
        self._c.execute(
            "INSERT OR REPLACE INTO commands VALUES (?,?,?,?,?,?,?,?)",
            (cmd.id, session_id, cmd.kind, cmd.agent_id, cmd.value,
             cmd.issued_at, cmd.applied_at, cmd.result))
        self._c.commit()
        return cmd.id

    def command_status(self, command_id: str) -> Command | None:
        r = self._c.execute("SELECT * FROM commands WHERE id=?",
                            (command_id,)).fetchone()
        return _cmd(r) if r else None

    def _latest_session(self) -> str:
        r = self._c.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        return r["id"] if r else ""


# ── (de)serialisation, kept dumb on purpose ──────────────────────────────

def _dump(v: SessionView) -> dict:
    return asdict(v)


def _load(d: dict) -> SessionView:
    agents = tuple(AgentView(**{**a, "tokens": TokenUse(**a.get("tokens", {}))})
                   for a in d.get("agents", []))
    d = {**d, "agents": agents, "tokens": TokenUse(**d.get("tokens", {}))}
    return SessionView(**d)


def _cmd(r: sqlite3.Row) -> Command:
    return Command(id=r["id"], kind=r["kind"], agent_id=r["agent_id"],
                   value=r["value"], issued_at=r["issued_at"],
                   applied_at=r["applied_at"], result=r["result"])
