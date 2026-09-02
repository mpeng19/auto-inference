"""The control plane: two processes, one store, durable commands."""
import threading

import pytest

from harness.contracts.session import (
    AgentView,
    Command,
    SessionStore,
    SessionView,
    TokenUse,
)
from harness.session import SqliteSessionStore


@pytest.fixture
def store(tmp_path):
    return SqliteSessionStore(tmp_path / "s.db")


def test_satisfies_the_contract(store):
    assert isinstance(store, SessionStore)


def test_a_snapshot_survives_the_round_trip(store):
    v = SessionView(session_id="s1", phase="running", started_at=1.0, pid=42,
                    agents=(AgentView("a01", status="evaluating",
                                      activity="sweep #2",
                                      tokens=TokenUse(1, 2, 3, 4)),))
    store.create(v)
    store.publish(v)
    back = store.read()
    assert back.session_id == "s1" and back.pid == 42
    assert back.agents[0].activity == "sweep #2"
    assert back.agents[0].tokens.cache_read == 3


def test_commands_are_durable_until_acknowledged(store):
    store.create(SessionView(session_id="s1"))
    cid = store.send_to("s1", Command(kind="pause", agent_id="a01"))
    assert [c.kind for c in store.take_commands("s1")] == ["pause"]
    assert store.take_commands("s1"), "still pending until acknowledged"
    store.acknowledge(cid, "paused")
    assert store.take_commands("s1") == ()
    assert store.command_status(cid).result == "paused"


def test_tokens_are_attributed_per_agent(store):
    """'The fleet spent $200' is not actionable; 'a03 spent $80' is."""
    store.create(SessionView(session_id="s1"))
    store.add_tokens("s1", "a01", TokenUse(10, 5, 100, 2))
    store.add_tokens("s1", "a01", TokenUse(1, 1, 1, 1))
    store.add_tokens("s1", "a02", TokenUse(7, 0, 0, 0))
    t = store.tokens("s1")
    assert t["a01"].input == 11 and t["a01"].cache_read == 101
    assert t["a02"].input == 7


def test_two_threads_do_not_corrupt_the_store(store):
    """A fleet writes while a TUI reads, from different threads and processes.

    A single shared connection raised "bad parameter or other API misuse" on a
    live fleet, where it surfaced as an agent failing to propose an idea.
    """
    store.create(SessionView(session_id="s1"))
    errs = []

    def work(i):
        try:
            for j in range(50):
                store.add_tokens("s1", f"a{i}", TokenUse(1, 1, 1, 1))
                store.publish(SessionView(session_id="s1", phase="running"))
                store.read("s1")
                store.send_to("s1", Command(kind="note", value=f"{i}-{j}"))
        except Exception as e:
            errs.append(repr(e))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs[:3]
    assert sum(v.total for v in store.tokens("s1").values()) == 8 * 50 * 4


def test_reading_with_no_session_returns_none(store):
    assert store.read() is None
