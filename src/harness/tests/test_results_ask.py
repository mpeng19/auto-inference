"""The results view and the ask agent read what the run wrote."""
import json
import pathlib

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
            calls.append({**kw, "messages": list(kw["messages"])})   # a snapshot
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


# ── Modal dollars the fleet never counted ──────────────────────────────────

def _workbench(agent: pathlib.Path, n: int, cost: float, ok: bool = True, mtime: float | None = None,
               script: str = '"""Probe the decode kernel at bs 12.\n\nMore detail.\n"""\nimport torch\n',
               extra: dict | None = None) -> pathlib.Path:
    d = agent / f"workbench-{n}"
    d.mkdir(parents=True)
    (d / "script.py").write_text(script)
    rec = {"ok": ok, "exit_code": 0 if ok else 1, "cost_usd": cost, "elapsed_s": 60.0,
           "stdout": "\n".join(f"line {i}" for i in range(12)),
           "stderr": "" if ok else "Traceback\n  boom\nRuntimeError: bad\nmore\nextra"}
    rec.update(extra or {})
    (d / "result.json").write_text(json.dumps(rec))
    if mtime is not None:
        import os
        os.utime(d / "result.json", (mtime, mtime))
    return d


def test_unreported_tool_spend_is_what_the_ledger_never_named(tmp_path):
    """Two workbench results, one named by a ledger line: the other is on
    disk only. The ledger line's own dollars were drained by the daemon
    (spend.seen at the end of the file), so they are in `reported`."""
    from harness.results import agent_modal_spend, unreported_tool_spend

    agent = tmp_path / "a00"
    _workbench(agent, 1, 0.5)
    _workbench(agent, 2, 0.7)
    (agent / "spend.jsonl").write_text(json.dumps(
        {"ts": 100.0, "tool": "gpu-run", "cost_usd": 0.5, "where": "workbench-1"}) + "\n")
    (agent / "spend.seen").write_text(str((agent / "spend.jsonl").stat().st_size))
    assert unreported_tool_spend(agent) == 0.7
    assert agent_modal_spend(agent, 1.0) == 1.7
    # an absolute `where` matches the same directory
    (agent / "spend.jsonl").write_text(json.dumps(
        {"ts": 100.0, "tool": "gpu-run", "cost_usd": 0.5,
         "where": str(agent / "workbench-1")}) + "\n")
    (agent / "spend.seen").write_text(str((agent / "spend.jsonl").stat().st_size))
    assert unreported_tool_spend(agent) == 0.7
    assert unreported_tool_spend(tmp_path / "nowhere") == 0.0


def test_undrained_ledger_lines_and_where_less_equivalence_lines_count_once(tmp_path):
    """A daemon that predates the ledger never drains it, so its lines are
    unreported too; an equivalence line names no directory but its cost is
    the sum of its runs', which it claims newest-first -- never a $9 gpu-run
    from before the ledger existed."""
    from harness.results import spend_summary, unreported_tool_spend

    agent = tmp_path / "a01"
    _workbench(agent, 1, 9.0, mtime=1000.0)                 # pre-ledger gpu-run
    _workbench(agent, 2, 0.05, mtime=2000.0)                # equivalence: reference
    _workbench(agent, 3, 0.03, mtime=2100.0)                # equivalence: candidate
    _workbench(agent, 4, 0.4, mtime=2500.0)                 # gpu-run, named
    lines = [{"ts": 2101.0, "tool": "equivalence", "cost_usd": 0.08, "where": ""},
             {"ts": 2501.0, "tool": "gpu-run", "cost_usd": 0.4, "where": str(agent / "workbench-4")}]
    (agent / "spend.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    # no spend.seen: nothing drained, every ledger dollar is unreported
    assert abs(unreported_tool_spend(agent) - (9.0 + 0.08 + 0.4)) < 1e-6
    s = spend_summary(agent)
    assert s["by_tool"]["equivalence"] == {"n": 1, "usd": 0.08} and s["on_disk"] == 9.48
    # the daemon drains the first line only
    first = len(json.dumps(lines[0]) + "\n")
    (agent / "spend.seen").write_text(str(first))
    assert abs(unreported_tool_spend(agent) - (9.0 + 0.4)) < 1e-6


def test_snapshot_text_and_cli_show_modal_dollars(tmp_path, capsys, monkeypatch):
    """`harness status`, `harness sessions` and the snapshot text the ask
    context carries all show reported + unreported."""
    from harness.cli import main as cli_main
    from harness.contracts.session import AgentView, SessionView
    from harness.results import snapshot_text
    from harness.session import SqliteSessionStore

    root = tmp_path / "agents" / "demo"
    _workbench(root / "a00", 1, 0.5)
    _workbench(root / "a00", 2, 0.7)
    (root / "a00" / "spend.jsonl").write_text(json.dumps(
        {"ts": 1.0, "tool": "gpu-run", "cost_usd": 0.5, "where": "workbench-1"}) + "\n")
    (root / "a00" / "spend.seen").write_text(str((root / "a00" / "spend.jsonl").stat().st_size))
    db = tmp_path / "s.db"
    store = SqliteSessionStore(db)
    v = SessionView(session_id="demo", phase="stopped", started_at=1.0, root=str(root),
                    target_agents=1, cost_usd=3.0, budget_usd=50.0,
                    agents=(AgentView("a00", status="done", cost_usd=1.0),))
    store.create(v)
    store.publish(v)
    text = snapshot_text(v)
    assert "Modal spend $3.70 of $50" in text and "spend $1.70" in text
    assert cli_main(["--store", str(db), "status"]) == 0
    out = capsys.readouterr().out
    assert "$3.70 of $50" in out and "1.70" in out.split("a00")[1]
    assert cli_main(["--store", str(db), "sessions"]) == 0
    assert "$    3.70" in capsys.readouterr().out
    # a session with no directory behind it is the daemon's word alone
    monkeypatch.chdir(tmp_path)
    store.publish(SessionView(session_id="demo", phase="stopped", started_at=1.0, root="",
                              target_agents=1, cost_usd=3.0, budget_usd=50.0,
                              agents=(AgentView("a00", status="done", cost_usd=1.0),)))
    assert cli_main(["--store", str(db), "sessions"]) == 0
    assert "$    3.70" in capsys.readouterr().out          # falls back to agents/<session>


# ── the ask context carries the agent's whole tool history ────────────────

def test_context_carries_workbench_profiles_attempts_and_the_design_note(tmp_path):
    root = _run(tmp_path)
    agent = root / "a00"
    _workbench(agent, 0, 0.5)
    _workbench(agent, 1, 0.25, ok=False,
               script="import torch\nprint('no docstring here')\n")
    _workbench(agent, 2, 0.4, extra={"result": {"top1_agreement": 0.991, "mean_abs_dlogprob": 0.012,
                                                "max_abs_dlogprob": 0.3, "n": 2000},
                                     "regressed": False},
               script="MODE = 'candidate'\nPROMPTSET_DIGEST = 'x'\n")
    _workbench(agent, 3, 0.1, extra={"ncu": {"decode_attn": {"gpu__time_duration.sum": 12.5}}},
               script="NCU_JSON\n")
    (agent / "spend.jsonl").write_text(json.dumps(
        {"ts": 1.0, "tool": "gpu-run", "cost_usd": 0.5, "where": "workbench-0"}) + "\n")
    (agent / "spend.seen").write_text(str((agent / "spend.jsonl").stat().st_size))   # drained
    prof = root / "profiles"
    prof.mkdir()
    (prof / "digest-win.sqlite").write_bytes(b"")
    (prof / "a00-other.sqlite").write_bytes(b"")
    rep = agent / "runs" / "attempt-000"
    rep.mkdir(parents=True)
    (rep / "report.txt").write_text(
        "  Qwen   1xH100   stack: [digest-win] 1 modified file(s)\n\n"
        "    8   14   0.30\n\nN* = 12 users   batch 11.9\n"
        "  whole bill       $9.41 per 1k requests   rank 3/12\n"
        "  quality gsm8k: 70.0%  (+4.0 pts)\n"
        "caveat: every level passed\n")
    design = agent / "candidate" / "sglang"
    design.mkdir(parents=True)
    (design / "DESIGN.md").write_text("# Fused decode\n\n## 1. Premise\nthe kernel is byte-bound\n"
                                      + "\n".join(f"line {i}" for i in range(100)))
    ctx = build_context(root)
    # every workbench run, labelled, with what it set out to do and how it went
    assert "a00/workbench-0  [gpu-run]  ok exit 0  $0.50" in ctx
    assert "purpose: Probe the decode kernel at bs 12." in ctx
    assert "a00/workbench-1  [gpu-run]  FAILED exit 1" in ctx
    assert "purpose: import torch / print('no docstring here')" in ctx
    assert "stderr (head):\n  Traceback\n  boom\n  RuntimeError: bad\n  more" in ctx
    assert "extra" not in ctx.split("stderr (head):")[1].split("###")[0]
    assert "line 11" in ctx and "line 3\n" not in ctx.split("workbench-0")[1].split("###")[0]
    assert "[equivalence]" in ctx and '"top1_agreement": 0.991' in ctx and '"regressed": false' in ctx
    assert "[ncu]" in ctx and "gpu__time_duration.sum" in ctx
    # profiles, with the stack each belongs to
    assert "digest-win.sqlite: stack digest-win  exp_win (a00, win)" in ctx
    assert "a00-other.sqlite: stack other  no experiment recorded" in ctx
    # the attempt's verdict lines, the ledger, the design note
    assert "N* = 12 users" in ctx and "quality gsm8k: 70.0%" in ctx and "caveat: every level passed" in ctx
    assert "8   14   0.30" not in ctx
    assert "ledger gpu-run: 1 calls, $0.50" in ctx
    assert "not in the fleet's own cost figure: $0.75" in ctx
    assert "Design note: a00" in ctx and "    # Fused decode" in ctx and "    line 55" in ctx
    assert "line 70" not in ctx.split("Design note: a00")[1]
    assert "Timeline" in ctx


def test_context_fits_its_budget_by_dropping_the_oldest_runs_first(tmp_path):
    from harness.ask import WORKBENCH_KEEP

    root = _run(tmp_path)
    agent = root / "a00"
    for i in range(30):
        _workbench(agent, i, 0.1, script=f'"""Run number {i:03d}."""\n')
    full = build_context(root, budget=10_000_000)
    assert "Run number 000." in full and "Run number 029." in full
    small = build_context(root, budget=len(full) - 2000)
    assert len(small) <= len(full) - 2000
    assert "Run number 029." in small and "Run number 000." not in small
    assert "older omitted" in small and "[truncated]" not in small
    # below what dropping runs can reach, the big sections give up the
    # excess in proportion; small ones are left whole
    tiny = build_context(root, budget=1800)
    assert len(tiny) <= 1800 and "[truncated]" in tiny
    assert tiny.startswith(f"## Run root\n{root}\n\n## Fleet config")
    assert tiny.count("### a00/workbench-") <= WORKBENCH_KEEP
