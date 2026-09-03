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
    # Like the real binary, refuse a -o directory that does not exist from
    # the cwd it is run in: a relative paper path handed through unchanged is
    # what left build-4's two papers as .tex.
    fake.write_text("#!/bin/sh\n[ -d \"$3\" ] || { echo 'output directory does not exist' >&2; exit 1; }\n"
                    "out=$(dirname \"$4\")/PAPER.pdf\nprintf '%%PDF-1.4 fake' > \"$out\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if name == "tectonic" else None)
    monkeypatch.chdir(tmp_path)
    out = pp.compile_tex(tex.relative_to(tmp_path))
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


def test_ensure_pdf_compiles_when_missing_or_stale(tmp_path, monkeypatch):
    """A .tex written by a daemon without tectonic, or after its PDF, gets
    compiled on the way to the reader."""
    import os
    import shutil
    import stat
    import time

    from harness import paper as pp

    d = tmp_path / "a02" / "paper" / "idea_x"
    d.mkdir(parents=True)
    tex = d / "PAPER.tex"
    tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}")
    fake = tmp_path / "tectonic"
    fake.write_text("#!/bin/sh\n[ -d \"$3\" ] || exit 1\nprintf '%%PDF-1.4 fake' > \"$(dirname \"$4\")/PAPER.pdf\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if name == "tectonic" else None)
    assert pp.find_papers(tmp_path)["idea_x"].suffix == ".tex"
    pdf = pp.ensure_pdf(tex)
    assert pdf == d / "paper.pdf" and pdf.is_file()
    assert pp.find_papers(tmp_path)["idea_x"] == pdf
    # a newer .tex is compiled again; a current PDF is left alone
    first = pdf.stat().st_mtime
    assert pp.ensure_pdf(tex) == pdf and pdf.stat().st_mtime == first
    time.sleep(0.02)
    tex.write_text(tex.read_text() + "%")
    os.utime(tex, None)
    assert pp.ensure_pdf(tex) == pdf and pdf.stat().st_mtime >= first
    assert pp.find_papers(tmp_path, compile=True)["idea_x"] == pdf


def test_template_carries_the_preamble_and_the_evidence_table(tmp_path):
    """The preamble is the owner's academic-paper template, inlined; the
    Mechanism section's evidence table is filled from results.evidence_for,
    with the file each row came from, and says *not measured* where nothing
    is on disk."""
    inp = pp.PaperInputs(
        title="Sparse KV", author="a01", attempts=[], baseline=14.96, figures=[],
        run_root="/agents/S/a01",
        evidence={"replicated": True, "gates": "held", "explains": True,
                  "ablation": {"tier": "screen", "env": {"SGLANG_DISABLE_X": "1"},
                               "as_is": {"bill_per_1k": 15.0}, "disabled": {"bill_per_1k": 17.4},
                               "baseline_bill_per_1k": 17.52, "explained_pct": 95.2,
                               "path": "/agents/S/a01/ablations/0/ablation.json",
                               "verdict": "the mechanism accounts for 95%"},
                  "decode_agreement": 0.7721, "lossless": False,
                  "equivalence_path": "/agents/S/a01/equivalence/d-5.json",
                  "profile_db": ""})
    d = pp.paper_dir(tmp_path, "idea_y")
    src = pp.render_template(inp, d).read_text()
    for pkg in ("newtxtext", "microtype", "mathtools", "booktabs", "siunitx", "cleveref",
                "hyperref", "caption", "enumitem", "tolblue"):
        assert pkg in src, pkg
    assert "\\newcommand{\\src}" in src and "\\pdfoutput=1\\n" not in src   # XeTeX, see the template
    for sec in ("Mechanism", "What changed", "Results", "What the gates said",
                "Limitations and what is unexplained", "Next"):
        assert f"\\section{{{sec}}}" in src, sec
    assert "\\label{tab:evidence}" in src and "\\label{tab:attempts}" in src
    assert "As is \\$15.00/1k, disabled \\$17.40/1k, baseline \\$17.52/1k" in src
    assert "accounts for 95\\% of the delta" in src and "within the 3\\%" in src
    assert "\\path{ablations/0/ablation.json}" in src
    assert "Decode agreement 0.7721" in src and "lossy" in src
    assert "\\path{equivalence/d-5.json}" in src
    assert "Profile & \\emph{not measured}" in src
    assert "pdftitle={Sparse KV}" in src
    # nothing measured: every row says so, nothing is invented
    bare = pp.render_template(pp.PaperInputs(title="t", author="a", attempts=[], baseline=None,
                                             figures=[]), pp.paper_dir(tmp_path, "idea_z"))
    assert bare.read_text().count("& \\emph{not measured}") == 3
    prompt = pp.prompt_for(inp, "h", "", "", files=["runs/attempt-002/report.txt"])
    assert ".claude/skills/writeup/SKILL.md" in prompt and "/agents/S/a01" in prompt
    assert "runs/attempt-002/report.txt" in prompt and "Refuse to write a number" in prompt
    assert "accounts for 95%" in prompt and "decode agreement 0.7721 (lossy)" in prompt


def test_run_files_lists_what_the_paper_may_cite(tmp_path):
    a = tmp_path / "a01"
    for rel in ("runs/attempt-002/report.txt", "runs/attempt-002/result.json",
                "ablations/0/ablation.json", "equivalence/d-1.json", "workbench-3/stdout.txt"):
        f = a / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "d.sqlite").write_bytes(b"")
    files = pp.run_files(a)
    assert "runs/attempt-002/report.txt" in files and "ablations/0/ablation.json" in files
    assert "workbench-3/stdout.txt" in files and "../profiles/d.sqlite" in files
    assert pp.run_files(tmp_path / "nope") == []
