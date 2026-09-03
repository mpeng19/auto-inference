"""One paper per idea: the write-up as a two-column PDF.

A trace is the record; a paper is the argument. When an idea has reached a
full sweep the agent writes `PAPER.tex` -- abstract, mechanism, what it
changed, the priced results with the run's own figures, what the gates said,
what it would try next -- and the harness compiles it with `tectonic`
(self-contained LaTeX, one binary) into `<agent>/paper/<idea id>/paper.pdf`.
Without tectonic the `.tex` is kept and the results view says so.

The template is fixed so the papers read alike across agents and nights; the
agent fills sections, it does not design a document.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
from dataclasses import dataclass

TEMPLATE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setlength{\parskip}{2pt}
\title{%(title)s}
\author{%(author)s}
\date{%(date)s}
\begin{document}
\maketitle
\begin{abstract}
%% Four sentences: the mechanism, what was changed, the priced result against
%% baseline with the gates' verdicts, and the one-line conclusion.
\end{abstract}

\section{Mechanism}
%% Why this should lower cost per output token, in the terms the price model
%% uses: the decode step is fixed + per-sequence KV read; which term moves.

\section{What changed}
%% Files, the launch line if changed (serving.json), and the design choices
%% that mattered. Cite the diff stat table.

\section{Results}
%% The attempts table, then the figures. Numbers from the run, not rounded to
%% look better.
\begin{table}[h]\centering\small
\begin{tabular}{llrrrl}
\toprule
attempt & tier & \$/1k & $\Delta$\%% & N* & gates \\
\midrule
%(attempt_rows)s
\bottomrule
\end{tabular}
\caption{Priced attempts. Baseline \$%(baseline)s/1k.}
\end{table}
%(figure_blocks)s

\section{What the gates said}
%% GSM8K, LongBench F1, token equivalence (agreement, |dlogprob|), and
%% whether the path under test was actually exercised by them.

\section{Next}
%% The single next experiment this result argues for, and what it would cost.
\end{document}
"""

_PROMPT = """Write the paper for the idea you just finished, as `PAPER.tex` in this
directory. Start from `PAPER.tex` as it is: it is a template with the title,
the attempts table and the figures already filled; replace each `%%` comment
with prose and leave the structure alone. Two columns, at most two pages.
Plain LaTeX only (the packages in the preamble), no new packages, no
\\input. Every number must come from the data below; if something was not
measured, say so rather than estimate.

Idea: {title}
Hypothesis: {hypothesis}
Design note (if any):
{design}

Attempts:
{attempts}

Best diff (truncated):
```
{diff}
```

Figures available (already \\includegraphics'd in the template): {figures}

When done, reply with one sentence: the paper's conclusion."""


@dataclass(frozen=True)
class PaperInputs:
    title: str
    author: str
    attempts: list[dict]         # {n, tier, bill, delta, n_star, gates}
    baseline: float | None
    figures: list[pathlib.Path]


def paper_dir(agent_root: str | pathlib.Path, idea_id: str) -> pathlib.Path:
    d = pathlib.Path(agent_root) / "paper" / idea_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_for(agent_root: str | pathlib.Path, attempt_ns: list[int]) -> list[pathlib.Path]:
    """The sweep plots of the given attempts, best first: the frontier plots
    and the price-vs-share curve. Copied into the paper directory by the
    caller so the .tex is self-contained."""
    out = []
    root = pathlib.Path(agent_root)
    for n in attempt_ns:
        d = root / "runs" / f"attempt-{n:03d}"
        for name in ("slo-mean-tpot.png", "price-vs-share.png", "slo-p90-ttft.png"):
            p = d / name
            if p.is_file():
                out.append(p)
    return out[:6]


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def render_template(inp: PaperInputs, out_dir: pathlib.Path) -> pathlib.Path:
    """Write the filled template. Figures are copied beside it."""
    rows = []
    for a in inp.attempts:
        bill = "-" if a.get("bill") is None else f"{a['bill']:.2f}"
        delta = "-" if a.get("delta") is None else f"{a['delta']:+.1f}"
        rows.append(f"{a.get('n', '?')} & {a.get('tier', '')} & {bill} & {delta} & "
                    f"{a.get('n_star', '-')} & {_tex_escape(str(a.get('gates', '')))} \\\\")
    blocks = []
    for i, fig in enumerate(inp.figures):
        dst = out_dir / f"fig{i}{fig.suffix}"
        shutil.copy2(fig, dst)
        blocks.append(f"\\begin{{figure}}[h]\\centering\\includegraphics[width=\\linewidth]"
                      f"{{{dst.name}}}\\caption{{{_tex_escape(fig.parent.name + ': ' + fig.stem)}}}"
                      f"\\end{{figure}}")
    import datetime as dt

    text = TEMPLATE % {
        "title": _tex_escape(inp.title), "author": _tex_escape(inp.author),
        "date": dt.date.today().isoformat(),
        "attempt_rows": "\n".join(rows) or "-- & -- & -- & -- & -- & -- \\\\",
        "baseline": "-" if inp.baseline is None else f"{inp.baseline:.2f}",
        "figure_blocks": "\n".join(blocks),
    }
    p = out_dir / "PAPER.tex"
    p.write_text(text)
    return p


def prompt_for(inp: PaperInputs, hypothesis: str, design: str, diff: str) -> str:
    return _PROMPT.format(
        title=inp.title, hypothesis=hypothesis,
        design=design or "(none)",
        attempts="\n".join(f"  attempt {a.get('n')} ({a.get('tier')}): ${a.get('bill')}/1k, "
                           f"{a.get('delta')}% vs baseline, N*={a.get('n_star')}, gates: {a.get('gates')}"
                           for a in inp.attempts) or "  (none priced)",
        diff=diff[:8000] or "(no diff recorded)",
        figures=", ".join(f"fig{i}{f.suffix} ({f.parent.name}/{f.stem})"
                          for i, f in enumerate(inp.figures)) or "none")


def compile_tex(tex: str | pathlib.Path, timeout_s: float = 240.0) -> pathlib.Path | None:
    """PDF beside the .tex, or None when tectonic is missing or fails.
    Failure is logged next to the source as `paper.log`, never raised."""
    tex = pathlib.Path(tex)
    binary = shutil.which("tectonic")
    if binary is None:
        (tex.parent / "paper.log").write_text("tectonic not installed; .tex kept\n")
        return None
    try:
        r = subprocess.run([binary, "--keep-logs", "-o", str(tex.parent), str(tex)],
                           capture_output=True, text=True, timeout=timeout_s,
                           cwd=str(tex.parent))
    except (subprocess.TimeoutExpired, OSError) as e:
        (tex.parent / "paper.log").write_text(f"{type(e).__name__}: {e}\n")
        return None
    (tex.parent / "paper.log").write_text((r.stdout or "") + (r.stderr or ""))
    pdf = tex.with_suffix(".pdf")
    if r.returncode != 0 or not pdf.is_file():
        return None
    out = tex.parent / "paper.pdf"
    if pdf != out:
        shutil.move(str(pdf), str(out))
    return out


def find_papers(root: str | pathlib.Path) -> dict[str, pathlib.Path]:
    """idea id -> paper.pdf (or PAPER.tex when uncompiled) under a fleet root."""
    out: dict[str, pathlib.Path] = {}
    for d in pathlib.Path(root).glob("a*/paper/*"):
        pdf = d / "paper.pdf"
        tex = d / "PAPER.tex"
        if pdf.is_file():
            out[d.name] = pdf
        elif tex.is_file():
            out[d.name] = tex
    return out
