"""How a fleet points its agents at GPU profiles.

`tracedb` is a separate service with its own CLI and MCP surface -- it knows
nothing about fleets, and a fleet does not need to know how it stores a trace.
This module is the seam: where a session's trace databases live, and how to
hand them to a Claude Code agent as first-class tools.

**MCP rather than the shell.** Agents already have a shell, so `tracedb ...`
would work. But a trace query is a tool an agent should reach for without being
told the syntax, and an MCP server gives it a typed tool list it can discover.
`claude --mcp-config` takes a JSON file, so wiring this is one generated file
per agent rather than a prompt full of command lines.
"""
from __future__ import annotations

import json
import pathlib
import sys

MCP_SERVER_NAME = "tracedb"


def profiles_dir(root: str | pathlib.Path) -> pathlib.Path:
    """Where a fleet keeps ingested traces. One SQLite file per capture."""
    d = pathlib.Path(root) / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def databases(root: str | pathlib.Path) -> list[pathlib.Path]:
    """Ingested trace databases for this fleet, newest first."""
    d = profiles_dir(root)
    return sorted(d.glob("*.sqlite"), key=lambda f: -f.stat().st_mtime)


def mcp_config(db: str | pathlib.Path) -> dict:
    """The `--mcp-config` payload that gives an agent the trace tools.

    Uses this interpreter rather than a bare `tracedb-mcp`, so it works from a
    venv that is not on the agent subprocess's PATH -- the same trap that made
    `preflight` silently skip its lint.
    """
    return {"mcpServers": {MCP_SERVER_NAME: {
        "command": sys.executable,
        "args": ["-m", "tracedb.mcp_server", "--db", str(db)],
    }}}


def write_mcp_config(workspace_root: str | pathlib.Path,
                     db: str | pathlib.Path) -> pathlib.Path:
    """Write the config beside the agent's workspace and return its path."""
    p = pathlib.Path(workspace_root) / "mcp.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mcp_config(db), indent=1))
    return p


def ingest(trace_path: str | pathlib.Path, root: str | pathlib.Path,
           name: str = "") -> dict:
    """Ingest a captured trace into this fleet's profile directory."""
    from tracedb.ingest import ingest as _ingest

    trace_path = pathlib.Path(trace_path)
    db = profiles_dir(root) / f"{name or trace_path.stem}.sqlite"
    return _ingest(trace_path, db)
