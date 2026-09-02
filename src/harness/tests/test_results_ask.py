"""The results view and the ask agent read what the run wrote."""
import json

from harness.ask import Asker, build_context
from harness.context import JsonlContext
from harness.contracts import Experiment, Turn
from harness.memory import SqliteMemory
from harness.results import diff_for, leaderboard, summary_text


def _run(tmp_path):
    root = tmp_path / "agents" / "x"
    root.mkdir(parents=True)
    m = SqliteMemory(root / "memory.db")
    ctx = JsonlContext(root / "traces", session_id="x")
    base = {"bill_per_1k": 12.23}
    from harness.contracts import TraceMeta
    trace = ctx.open(TraceMeta(agent_id="a00", idea_id="i1", model="proposer"))
    ctx.append(trace, Turn(kind="eval_submit", name="digest-win",
                           data={"tier": "full", "diff": "--- stock/srt/a.py\n+++ candidate\n+fast\n"}))
    m.record(Experiment(id="exp_win", agent_id="a00", idea_id="i1", verdict="win",
                        hypothesis="fused decode attention", summary="bill $10.50/1k, -14.1%",
                        stack_digest="digest-win", trace_ref=trace,
                        metrics={"bill_per_1k": 10.5, "rank_bill": 3, "rank_of": 12,
                                 "share_per_node": 0.0051, "n_star": 12,
                                 "quality": [{"suite": "gsm8k", "accuracy": 0.69}]},
                        baseline_metrics=base))
    m.record(Experiment(id="exp_neutral", agent_id="a01", idea_id="i2", verdict="neutral",
                        hypothesis="widen lpm cutoff", summary="bill $12.30/1k, +0.6%",
                        stack_digest="digest-n", metrics={"bill_per_1k": 12.3},
                        baseline_metrics=base))
    m.record(Experiment(id="exp_bad", agent_id="a02", idea_id="i3", verdict="invalid",
                        hypothesis="broken kernel", summary="", metrics={},
                        baseline_metrics=base))
    (root / "fleet.json").write_text(json.dumps({"session_id": "x", "mode": "build",
                                                 "baseline": base}))
    (root / "tools").mkdir()
    (root / "tools" / "README.md").write_text("# Shared tools\n\n## bench_attn\n")
    return root


def test_leaderboard_is_best_first_with_rank_and_share(tmp_path):
    root = _run(tmp_path)
    rows = leaderboard(root)
    assert [r.experiment_id for r in rows] == ["exp_win", "exp_neutral", "exp_bad"]
    best = rows[0]
    assert best.delta_pct == -14.15 and best.rank == "3/12" and best.n_star == 12
    assert abs(best.share_pct - 0.51) < 1e-9 and best.quality == "gsm8k 69%"
    assert rows[2].delta_pct is None
    assert "+fast" in diff_for(root, best) and diff_for(root, rows[1]) == ""
    text = summary_text(root)
    assert "3 experiments" in text and "-14." in text and "3/12" in text
    assert leaderboard(tmp_path / "nowhere") == []


def test_ask_sends_the_run_as_cached_context_and_keeps_history(tmp_path):
    root = _run(tmp_path)
    calls = []

    class Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class Msg:
        stop_reason = "end_turn"

        def __init__(self, text):
            self.content = [Block(text)]
            self.usage = type("U", (), {"input_tokens": 900, "output_tokens": 40,
                                        "cache_read_input_tokens": 800})()

    class Stream:
        def __init__(self, kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return Msg(f"answer {len(self.kw['messages'])}")

    class Messages:
        def stream(self, **kw):
            calls.append(kw)
            return Stream(kw)

    client = type("C", (), {"beta": type("B", (), {"messages": Messages()})()})()
    asker = Asker(root, client=client)
    assert asker.ask("which diff won?") == "answer 1"
    assert asker.ask("and why?") == "answer 3"
    kw = calls[-1]
    assert kw["model"] == "claude-fable-5-1" and kw["fallbacks"] == "default"
    system = kw["system"][0]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert "exp_win" in system["text"] and "+fast" in system["text"]
    assert "bench_attn" in system["text"] and '"mode": "build"' in system["text"]
    assert [m["role"] for m in kw["messages"]] == ["user", "assistant", "user"]
    assert asker.last_usage["cache_read"] == 800
    assert "Leaderboard" in build_context(root)
