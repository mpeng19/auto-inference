"""Pointing agents at a GPU profile."""
import json

from harness import profile


def test_mcp_config_uses_this_interpreter(tmp_path):
    """A bare `tracedb-mcp` is not on an agent subprocess's PATH -- the same
    trap that made preflight silently skip its lint."""
    import sys

    cfg = profile.mcp_config(tmp_path / "x.sqlite")
    srv = cfg["mcpServers"]["tracedb"]
    assert srv["command"] == sys.executable
    assert srv["args"][:2] == ["-m", "tracedb.mcp_server"]


def test_write_mcp_config_lands_beside_the_workspace(tmp_path):
    p = profile.write_mcp_config(tmp_path / "a01", tmp_path / "x.sqlite")
    assert p.is_file()
    assert json.loads(p.read_text())["mcpServers"]["tracedb"]


def test_databases_are_listed_newest_first(tmp_path):
    d = profile.profiles_dir(tmp_path)
    for n in ("a", "b"):
        (d / f"{n}.sqlite").write_text("")
    import os
    import time
    os.utime(d / "b.sqlite", (time.time() + 10, time.time() + 10))
    assert [p.stem for p in profile.databases(tmp_path)] == ["b", "a"]


def test_ingest_lands_in_the_fleet_profile_dir(tmp_path):
    from tracedb.synth import generate

    generate(tmp_path / "t.json", steps=3)
    got = profile.ingest(tmp_path / "t.json", tmp_path, name="sweep-1")
    assert got["events"] > 0
    assert (profile.profiles_dir(tmp_path) / "sweep-1.sqlite").is_file()
