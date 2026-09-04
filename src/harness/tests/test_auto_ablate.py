"""A replicated win is ablated on the spot, so it can be publishable.

`results.publishable` asks for an ablation and nobody ran one: every win in
build-4 sat at `no-ablation`. Now the agent declares its kill switch in
`ablation.env`, and the loop prices the replicated win once more with it
set (`tools.ablate`, screen tier), records the verdict in the trace and the
attempt, and never lets the ablation fail the idea.
"""
import json
import pathlib
from dataclasses import dataclass, field

import pytest

from harness import EvalBroker, IterativeAgent, Workspace, tools
from harness import results as rs
from harness.contracts import AgentBudget, Idea

from .test_workspace import FakeStock

P = "srt/managers/schedule_policy.py"


@dataclass
class WinningProposer:
    env_text: str | None = "SGLANG_DISABLE_X=1\n"

    def edit(self, ws, idea, brief, attempt, history):
        ws.edit(P, "CHUNK = 32768\n\n\nclass SchedulePolicy:\n    pass\n")
        if self.env_text is not None:
            (ws.candidates / "ablation.env").write_text(self.env_text)
        return "a faster chunk size"


@dataclass
class QualityRunner:
    """Full sweeps that win, replicate a little worse, and score GSM8K."""
    n: int = 0
    calls: list = field(default_factory=list)

    def __call__(self, req):
        self.n += 1
        self.calls.append(req.tier)
        bill = 10.0 if self.n == 1 else 10.5
        return True, {"bill_per_1k": bill, "n_star": 12, "cost_usd": 1.0,
                      "quality": [{"suite": "gsm8k", "accuracy": 0.70}]}, ""


@dataclass
class StubAblate:
    """Stands in for `tools.ablate`: writes the record the real one would."""
    explains: bool = True
    fail: bool = False
    calls: list = field(default_factory=list)

    def __call__(self, workspace, env, tier="screen", baseline=None, source=None):
        self.calls.append({"workspace": str(workspace), "env": dict(env), "tier": tier,
                           "baseline": baseline})
        if self.fail:
            raise RuntimeError("modal is down")
        ws = Workspace(workspace, source=source)
        digest = ws.stack().digest
        out = pathlib.Path(workspace) / "ablations" / "0"
        out.mkdir(parents=True)
        rec = {"ok": True, "tier": tier, "env": dict(env), "stack_digest": digest,
               "baseline_bill_per_1k": baseline,
               "as_is": {"ok": True, "bill_per_1k": 11.0, "n_star": 12},
               "disabled": {"ok": True, "bill_per_1k": 12.2, "n_star": 12},
               "explains": self.explains, "explained_pct": 95.0, "delta_pct": -9.8,
               "cost_usd": 2.0, "ts": 1.0, "dir": str(out),
               "verdict": "switching the mechanism off returns the price to baseline"}
        (out / "ablation.json").write_text(json.dumps(rec))
        return rec


def _run(tmp_path, stock_dir, memory, context, prop, budget=None):
    runner = QualityRunner()
    broker = EvalBroker(runner, capacity=2)
    ws = Workspace(tmp_path / "a00", agent_id="a00", source=FakeStock(stock_dir))
    agent = IterativeAgent(agent_id="a00", workspace=ws, memory=memory, context=context,
                           proposer=prop, evals=broker,
                           baseline={"bill_per_1k": 12.23, "quality": {"gsm8k": 0.69},
                                     "screen": {"bill_per_1k": 14.0}})
    idea = Idea(title="chunk", hypothesis="a bigger chunk lowers the bill", targets=(P,))
    try:
        out = agent.run(idea, budget or AgentBudget(max_attempts=2, screen_first=False))
    finally:
        broker.shutdown()
    return out, runner, ws


def _turns(context, out, name):
    ref = out.attempts[-1].trace_ref
    return [t for t in context.read(ref) if t.name == name]


def test_a_replicated_win_is_ablated_and_becomes_publishable(tmp_path, stock_dir, memory,
                                                             context, monkeypatch):
    stub = StubAblate()
    monkeypatch.setattr(tools, "ablate", stub)
    out, runner, ws = _run(tmp_path, stock_dir, memory, context, WinningProposer())
    assert out.stop == "won" and runner.calls == ["full", "full"]
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["env"] == {"SGLANG_DISABLE_X": "1"} and call["tier"] == "screen"
    assert call["workspace"] == str(ws.root) and call["baseline"] == 14.0
    # The verdict travels with the attempt and the trace.
    abl = out.best.metrics["ablation"]
    assert abl["explains"] is True and abl["as_is"] == 11.0 and abl["disabled"] == 12.2
    assert abl["stack_digest"] == out.best.stack_digest
    turn = _turns(context, out, "ablate")
    assert len(turn) == 1 and turn[0].kind == "tool_call"
    assert "returns the price to baseline" in turn[0].content
    assert turn[0].data["cost_usd"] == 2.0
    assert out.cost_usd == pytest.approx(2.0 + 2.0)      # two sweeps, one ablation
    # What `harness results` reads: the record is found for this stack and
    # the win is publishable.
    ev = rs.evidence_for(tmp_path, "a00", out.best.stack_digest, baseline=12.23,
                         replicated=True, verdict="win", metrics=out.best.metrics)
    assert ev["ablation"]["explains"] is True and ev["gates"] == "held"
    assert rs.publishable("win", ev) == (True, "yes")
    board = rs.leaderboard(tmp_path)
    assert board and board[0].publishable and board[0].pub == "yes"


def test_no_kill_switch_means_no_ablation_and_not_publishable(tmp_path, stock_dir, memory,
                                                             context, monkeypatch):
    stub = StubAblate()
    monkeypatch.setattr(tools, "ablate", stub)
    out, _, _ = _run(tmp_path, stock_dir, memory, context, WinningProposer(env_text=None))
    assert out.stop == "won" and stub.calls == []
    assert "ablation" not in out.best.metrics
    note = _turns(context, out, "ablate")
    assert len(note) == 1 and note[0].kind == "thought" and "ablation.env" in note[0].content
    assert rs.leaderboard(tmp_path)[0].pub == "no-ablation"


def test_auto_ablate_can_be_switched_off(tmp_path, stock_dir, memory, context, monkeypatch):
    stub = StubAblate()
    monkeypatch.setattr(tools, "ablate", stub)
    out, _, _ = _run(tmp_path, stock_dir, memory, context, WinningProposer(),
                     budget=AgentBudget(max_attempts=2, screen_first=False, auto_ablate=False))
    assert out.stop == "won" and stub.calls == []
    assert _turns(context, out, "ablate") == []


def test_an_ablation_that_fails_never_fails_the_idea(tmp_path, stock_dir, memory, context,
                                                     monkeypatch):
    stub = StubAblate(fail=True)
    monkeypatch.setattr(tools, "ablate", stub)
    out, _, _ = _run(tmp_path, stock_dir, memory, context, WinningProposer())
    assert out.stop == "won" and len(stub.calls) == 1
    assert "ablation" not in out.best.metrics
    err = _turns(context, out, "ablate")
    assert len(err) == 1 and err[0].kind == "error" and "modal is down" in err[0].content
    assert out.cost_usd == pytest.approx(2.0)


def test_the_kill_switch_file_is_parsed_like_an_env_file(tmp_path, stock_dir):
    ws = Workspace(tmp_path / "a00", agent_id="a00", source=FakeStock(stock_dir))
    agent = IterativeAgent(agent_id="a00", workspace=ws, memory=None, context=None,
                           proposer=None, evals=None)
    assert agent._kill_switch() == {}
    (ws.candidates / "ablation.env").write_text(
        "# the switch\nSGLANG_DISABLE_X=1\n\nOTHER='two words'\nnot a pair\n =nokey\n")
    assert agent._kill_switch() == {"SGLANG_DISABLE_X": "1", "OTHER": "two words"}


def test_the_edit_prompt_asks_for_the_kill_switch():
    from harness.agent.claude_code import _EDIT_PROMPT

    assert "ablation.env" in _EDIT_PROMPT
    assert "cannot be published" in _EDIT_PROMPT
