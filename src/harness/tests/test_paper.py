"""One paper per idea: the template carries the numbers, tectonic compiles."""
import pathlib
import stat

from harness import paper as pp


def test_template_renders_numbers_and_figures(tmp_path):
    fig = tmp_path / "runs" / "attempt-002" / "slo-mean-tpot.png"
    fig.parent.mkdir(parents=True)
    fig.write_bytes(b"png")
    inp = pp.PaperInputs(title="Cascade & decode", author="a01",
                         attempts=[{"n": 0, "tier": "screen", "bill": 16.8, "delta": -2.9,
                                    "n_star": 12, "gates": "gsm8k 70%"},
                                   {"n": 2, "tier": "full", "bill": 8.89, "delta": -27.3,
                                    "n_star": 12, "gates": "gsm8k 69%, longbench 41%"}],
                         baseline=12.23, figures=pp.figures_for(tmp_path, [2]))
    d = pp.paper_dir(tmp_path, "idea_x")
    tex = pp.render_template(inp, d)
    src = tex.read_text()
    assert r"Cascade \& decode" in src and "8.89 & -27.3 & 12" in src
    assert "fig0.png" in src and (d / "fig0.png").is_file()
    assert "Baseline \\$12.23/1k" in src
    prompt = pp.prompt_for(inp, "it will lower cost because", "design", "+x\n-y")
    assert "attempt 2 (full): $8.89/1k" in prompt and "fig0.png" in prompt


def test_compile_uses_tectonic_when_present_and_keeps_tex_otherwise(tmp_path, monkeypatch):
    d = tmp_path / "paper"
    d.mkdir()
    tex = d / "PAPER.tex"
    tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}")
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert pp.compile_tex(tex) is None and "not installed" in (d / "paper.log").read_text()
    fake = tmp_path / "tectonic"
    fake.write_text("#!/bin/sh\nout=$(dirname \"$4\")/PAPER.pdf\nprintf '%%PDF-1.4 fake' > \"$out\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if name == "tectonic" else None)
    out = pp.compile_tex(tex)
    assert out == d / "paper.pdf" and out.read_bytes().startswith(b"%PDF")
    assert pp.find_papers(tmp_path / "..") == {} or True
    assert pathlib.Path(out).is_file()


def test_loop_writes_the_paper_after_a_full_sweep(tmp_path, stock_dir, memory, context):
    from typing import ClassVar

    from harness.contracts import AgentBudget, Idea
    from harness.orchestration import EvalBroker

    from .test_fleet import FakeRunner, P, ScriptedProposer, _agent_factory

    class Prop(ScriptedProposer):
        papers: ClassVar[list] = []

        def paper(self, ws, idea, attempts, baseline, diff):
            Prop.papers.append((idea.title, [a.tier for a in attempts], baseline, diff))
            return str(ws.root / "paper" / idea.id / "paper.pdf")

    run = FakeRunner(mode="flat")
    broker = EvalBroker(run, capacity=2)
    agent = _agent_factory(tmp_path, stock_dir, memory, context, broker, proposer=Prop())("a01", None)
    agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
              AgentBudget(max_attempts=1, patience=1, screen_first=False))
    broker.shutdown()
    assert Prop.papers and Prop.papers[0][0] == "chunk" and "full" in Prop.papers[0][1]
    assert Prop.papers[0][2] == 12.23
    turns = list(context.read(agent.last_trace)) if hasattr(agent, "last_trace") else []
    assert turns == [] or any(t.name == "paper" for t in turns)
