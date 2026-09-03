"""`harness delete` and the bank actions that fill and hand out ideas."""
import json
import os

import pytest

from harness.cli import main as cli_main
from harness.contracts.session import AgentView, Command, SessionView, TokenUse
from harness.session import SqliteSessionStore


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))


def _finished_run(tmp_path, phase="stopped", pid=0):
    root = tmp_path / "agents" / "s1"
    (root / "a00" / "runs" / "attempt-000").mkdir(parents=True)
    (root / "a00" / "runs" / "attempt-000" / "sweep.json").write_text("x" * 4096)
    (root / "fleet.json").write_text(json.dumps({"session_id": "s1"}))
    db = tmp_path / "s.db"
    store = SqliteSessionStore(db)
    v = SessionView(session_id="s1", phase=phase, pid=pid, root=str(root),
                    agents=(AgentView("a00", status="done"),))
    store.create(v)
    store.publish(v)
    store.add_tokens("s1", "a00", TokenUse(1, 1, 1, 1))
    store.send_to("s1", Command(kind="stop"))
    return db, store, root


def test_delete_wipes_a_finished_fleet_with_yes(tmp_path, capsys):
    db, store, root = _finished_run(tmp_path)
    assert cli_main(["--store", str(db), "delete", "--session", "s1", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "will remove" in out and "4 KB" in out and "1 token rows" in out
    assert "deleted" in out and "sessions 1" in out
    assert not root.exists()
    assert store.read("s1") is None and store.tokens("s1") == {}
    # the global --session works too, and a second delete finds nothing
    assert cli_main(["--store", str(db), "--session", "s1", "delete", "--yes"]) == 1
    assert "no session" in capsys.readouterr().err


def test_delete_refuses_a_running_fleet_whose_daemon_is_alive(tmp_path, capsys):
    db, store, root = _finished_run(tmp_path, phase="running", pid=os.getpid())
    assert cli_main(["--store", str(db), "delete", "--session", "s1", "--yes"]) == 1
    assert "alive" in capsys.readouterr().err
    assert root.exists() and store.read("s1") is not None
    # the same session with a dead daemon is finished in all but name
    store.publish(SessionView(session_id="s1", phase="running", pid=999_999_999,
                              root=str(root)))
    assert cli_main(["--store", str(db), "delete", "--session", "s1", "--yes"]) == 0
    assert "daemon is gone" in capsys.readouterr().out
    assert not root.exists()


def test_delete_refuses_a_pid_file_that_is_alive(tmp_path, capsys):
    db, _store, root = _finished_run(tmp_path)
    (root / "daemon.pid").write_text(str(os.getpid()))
    assert cli_main(["--store", str(db), "delete", "--session", "s1", "--yes"]) == 1
    assert "refusing" in capsys.readouterr().err
    assert root.exists()


def test_delete_needs_yes_or_a_terminal(tmp_path, capsys, monkeypatch):
    import sys

    db, _store, root = _finished_run(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli_main(["--store", str(db), "delete", "--session", "s1"]) == 1
    assert "--yes" in capsys.readouterr().err
    assert root.exists()
    # interactive: typing the wrong thing does nothing, the id deletes
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "nope")
    assert cli_main(["--store", str(db), "delete", "--session", "s1"]) == 1
    assert root.exists()
    monkeypatch.setattr("builtins.input", lambda prompt="": "s1")
    assert cli_main(["--store", str(db), "delete", "--session", "s1"]) == 0
    assert not root.exists()


def test_delete_without_a_session_row_still_removes_the_directory(tmp_path, capsys):
    """An old run whose store rows are already gone (or were in another
    store) is still a directory to be rid of."""
    root = tmp_path / "agents" / "old"
    root.mkdir(parents=True)
    (root / "memory.db").write_text("x")
    db = tmp_path / "s.db"
    assert cli_main(["--store", str(db), "delete", "--session", "old",
                     "--root", str(root), "--yes"]) == 0
    assert not root.exists()
    assert cli_main(["--store", str(db), "delete", "--session", "old", "--yes"]) == 1


def test_ideas_seed_claim_related(tmp_path, capsys):
    bank = str(tmp_path / "ideas.db")
    assert cli_main(["--store", str(tmp_path / "s.db"), "ideas", "seed", "--bank", bank]) == 0
    out = capsys.readouterr().out
    assert out.startswith("seeded ") and int(out.split()[1]) > 0
    # a claim hands out one record and marks it; --seed steers it
    assert cli_main(["--store", str(tmp_path / "s.db"), "ideas", "claim", "--bank", bank,
                     "--agent", "me", "--seed", "kv cache quantization"]) == 0
    line = capsys.readouterr().out.strip()
    rec_id, title = line.split("  ", 1)
    assert rec_id and title
    from harness.ideas import SqliteIdeaBank
    b = SqliteIdeaBank(bank)
    assert b.get(rec_id).claimed_by == "me" and b.get(rec_id).status == "claimed"
    assert cli_main(["--store", str(tmp_path / "s.db"), "ideas", "related", rec_id,
                     "--bank", bank, "-k", "3"]) == 0
    rows = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert 1 <= len(rows) <= 3
    assert all(ln.split()[0] != rec_id and b.get(ln.split()[0]) is not None for ln in rows)
    assert rows[0].split()[2] in ("available", "claimed", "tried", "retired")
    # an unknown seed set is a clean error, not a traceback
    with pytest.raises(FileNotFoundError):
        cli_main(["--store", str(tmp_path / "s.db"), "ideas", "seed", "nope", "--bank", bank])
