"""The timeline is the trace, for people."""
from harness.context import JsonlContext
from harness.contracts import TraceMeta, Turn
from harness.timeline import (
    build,
    render_html,
    render_markdown,
    render_text,
    write_markdown,
)


def _trace(root, agent, title, with_phases=True):
    ctx = JsonlContext(root / "traces", session_id="s")
    t = ctx.open(TraceMeta(agent_id=agent, idea_id="i", model="proposer"))
    ctx.append(t, Turn(kind="prompt", content=title))
    ctx.append(t, Turn(kind="thought", name="recall", content="nothing",
                       data={"phase": "recall", "elapsed_s": 1} if with_phases else {}))
    ctx.append(t, Turn(kind="thought", name="propose", content="wrote it",
                       data={"phase": "propose", "elapsed_s": 600, "denials": 0} if with_phases else {}))
    ctx.append(t, Turn(kind="eval_submit", name="d1", data={"tier": "screen", "diff": "+x"}))
    ctx.append(t, Turn(kind="thought", name="study", content="notes",
                       data={"phase": "study", "elapsed_s": 300} if with_phases else {}))
    ctx.append(t, Turn(kind="eval_result", name="d1",
                       data={"tier": "screen", "bill_per_1k": 16.9, "cost_usd": 1.0,
                             "quality": [{"suite": "gsm8k", "accuracy": 0.7}],
                             **({"phase": "wait", "elapsed_s": 900} if with_phases else {})}))
    ctx.close(t, outcome="no_progress", cost_usd=1.0)
    return t


def test_phases_durations_and_results_come_through(tmp_path):
    _trace(tmp_path, "a00", "fused decode attention")
    _trace(tmp_path, "a01", "int8 kv", with_phases=False)
    runs = build(tmp_path)
    assert [r.agent for r in runs] == ["a00", "a01"]
    a = runs[0]
    assert a.title == "fused decode attention" and a.outcome == "no_progress"
    ph = a.by_phase()
    assert round(ph["edit"]) == 600 and round(ph["study"]) == 300 and round(ph["wait"]) == 900
    assert a.results == ["screen $16.90/1k, quality 70%"]
    b = runs[1]                                   # pre-phase trace: gaps, not stamps
    assert set(b.by_phase()) >= {"edit", "study", "wait"}
    text = render_text(tmp_path)
    assert "a00" in text and "edit 10m" in text and "-> screen $16.90/1k" in text
    md = render_markdown(tmp_path)
    assert "| edit | study | wait |" in md and "fused decode attention" in md
    p = write_markdown(tmp_path)
    assert p.name == "timeline.md" and "## a01" in p.read_text()
    page = render_html(tmp_path)
    assert "<svg" in page and "fused decode attention" in page and "#f58518" in page
    assert render_text(tmp_path / "empty") == "no traces yet"
