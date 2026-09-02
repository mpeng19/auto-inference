"""The dashboard renders and its keys reach the fleet.

Driven headlessly through textual's pilot, so this is a real mount-and-press,
not a check that the module imports.
"""
import pytest

from harness.contracts.session import AgentView, SessionView, TokenUse
from harness.session import SqliteSessionStore
from harness.tui.app import FleetApp, _bar, _money, _tokens


@pytest.fixture
def store(tmp_path):
    s = SqliteSessionStore(tmp_path / "s.db")
    v = SessionView(
        session_id="demo", phase="running", started_at=1.0, pid=1,
        target_agents=3, cost_usd=12.5, budget_usd=200.0,
        tokens=TokenUse(1000, 500, 480000, 9000),
        evals_running=2, evals_capacity=3, evals_queued=1, evals_completed=7,
        evals_deduped=2, gpu_utilisation=0.81,
        agents=(
            AgentView("a00", status="evaluating", idea_title="lfu eviction",
                      idea_hypothesis="LFU keeps hot prefixes resident",
                      activity="attempt 2: full sweep running", attempt=2,
                      best_delta_pct=-4.1, cost_usd=8.2, queued_s=91.0,
                      tokens=TokenUse(400, 200, 300000, 4000)),
            AgentView("a01", status="thinking", idea_title="chunk sizing",
                      activity="writing a diff", cost_usd=2.1,
                      tokens=TokenUse(300, 150, 120000, 3000)),
            AgentView("a02", status="paused", idea_title="batch admission",
                      activity="paused by operator", cost_usd=2.2,
                      tokens=TokenUse(300, 150, 60000, 2000)),
        ))
    s.create(v)
    s.publish(v)
    return s


def test_formatters_are_readable():
    assert _tokens(206937) == "207k" and _tokens(1_500_000) == "1.5M"
    assert _money(12.234) == "$12.23"
    assert _bar(50, 100, 10) == "[#####.....]"


async def test_it_renders_every_agent(store):
    app = FleetApp(store, "demo")
    async with app.run_test(size=(120, 26)) as pilot:
        await pilot.pause()
        await pilot.pause()
        table = app.query_one("#table")
        assert table.row_count == 3
        assert "demo" in app.summary_text and "running" in app.summary_text
        # 2 of 3: a paused agent is not a live one, which is the distinction
        # an operator is looking at the number for.
        assert "agents 2/3" in app.summary_text
        assert "$12.50" in app.summary_text and "490k" in app.summary_text
        # the detail pane follows the cursor
        assert "lfu eviction" in app.detail_text


async def test_keys_issue_commands_the_fleet_will_see(store):
    """The TUI cannot call the fleet -- it is another process. It writes rows."""
    app = FleetApp(store, "demo")
    async with app.run_test(size=(120, 26)) as pilot:
        await pilot.pause()
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause()
        cmds = store.take_commands("demo")
        assert [(c.kind, c.agent_id) for c in cmds] == [("pause", "a00")]


async def test_scaling_keys_target_the_session_not_an_agent(store):
    app = FleetApp(store, "demo")
    async with app.run_test(size=(120, 26)) as pilot:
        await pilot.pause()
        await pilot.pause()
        await pilot.press("plus")
        await pilot.pause()
        cmds = [c for c in store.take_commands("demo") if c.kind == "scale"]
        assert cmds and cmds[0].value == "4" and cmds[0].agent_id == ""


async def test_it_survives_an_empty_store(tmp_path):
    empty = SqliteSessionStore(tmp_path / "empty.db")
    app = FleetApp(empty, "")
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert "no session" in app.summary_text.lower()
