"""One paper per idea: the write-up as a two-column PDF.

A trace is the record; a paper is the argument. When an idea has reached a
full sweep the agent writes `PAPER.tex` -- abstract, mechanism with its
evidence table, what it changed, the priced results with the run's own
figures, what the gates said, what is unexplained, what it would try next --
and the harness compiles it with `tectonic` (self-contained LaTeX, one
binary) into `<agent>/paper/<idea id>/paper.pdf`. Without tectonic the `.tex`
is kept and the results view says so.

The template is fixed so the papers read alike across agents and nights; the
agent fills sections, it does not design a document. Its preamble is the
owner's academic-paper template (the `latex-document-skill`'s
`assets/templates/academic-paper.tex`), copied here rather than imported: the
repo has to render a paper without that skill installed. What is kept from
it: Times text and math (`newtx`), `microtype`, `mathtools`, the
`caption`/`subcaption` and `booktabs`/`array`/`multirow` table conventions,
`enumitem` compact lists, the Tol colour palette, `siunitx`, `hyperref` with
coloured links and PDF metadata, and `cleveref` with short names. Dropped:
theorems, algorithms, `authblk`, `natbib` and one-and-a-half spacing -- a
two-page result note has no use for them.

The evidence table in the Mechanism section is filled by `render_template`
from `results.evidence_for`: the ablation, the profile and the equivalence
record, each with the file it came from. The prose is asked to cite files
the same way (`\\src{}`); the `writeup` skill says why.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass, field

TEMPLATE = r"""\documentclass[10pt,twocolumn]{article}

%% Preamble after `academic-paper.tex` in the owner's latex-document-skill,
%% trimmed to a two-page, two-column result note. Do not edit it. (No
%% \pdfoutput=1: tectonic is XeTeX, and that line makes hyperref look for
%% pdfTeX's \pdfmajorversion.)
%%---------------------------------------------------------------------------
%% ENCODING AND FONTS
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{newtxmath}
%% PAGE LAYOUT AND TYPOGRAPHY
\usepackage[margin=0.75in]{geometry}
\usepackage{microtype}
\setlength{\parskip}{2pt}
%% MATH (no amssymb: newtxmath supplies the AMS symbols, and loading both
%% is "\Bbbk already defined" under tectonic)
\usepackage{mathtools}
%% GRAPHICS AND FIGURES
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf,format=hang]{caption}
\usepackage{subcaption}
%% TABLES
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
%% LISTS
\usepackage{enumitem}
\setlist{nosep}
%% COLORS (colorblind-safe Tol palette)
\usepackage{xcolor}
\definecolor{linkblue}{RGB}{0,51,153}
\definecolor{tolblue}{RGB}{0,114,178}
\definecolor{tolorange}{RGB}{230,159,0}
\definecolor{tolgreen}{RGB}{0,158,115}
%% UNITS AND NUMBERS
\usepackage{siunitx}
\sisetup{detect-all}
%% HYPERLINKS (load near end)
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=linkblue,
    citecolor=linkblue,
    urlcolor=linkblue,
    pdfauthor={%(author)s},
    pdftitle={%(title)s},
    pdfsubject={Inference serving cost},
    bookmarks=true,
    bookmarksopen=true,
}
%% SMART CROSS-REFERENCES (must load after hyperref)
\usepackage{cleveref}
\crefname{figure}{Fig.}{Figs.}
\crefname{table}{Table}{Tables}
\crefname{section}{Sec.}{Secs.}
%% A citation to a file in the run directory. Every number carries one:
%% \src{runs/attempt-002/report.txt}. \path handles underscores.
\newcommand{\src}[1]{{\footnotesize\,\path{#1}}}
%%---------------------------------------------------------------------------

\title{\textbf{%(title)s}}
\author{%(author)s}
\date{%(date)s}

\begin{document}
\maketitle

\begin{abstract}
\noindent
%% Four sentences: the mechanism, what was changed, the priced result against
%% baseline with the gates' verdicts, and the one-line conclusion -- including
%% whether the result meets the publishable bar (replicated win, explained by
%% an ablation) or what is missing.
\end{abstract}

\section{Mechanism}
\label{sec:mechanism}
%% 1. The hypothesis: why this should lower cost per output token, in the
%%    terms the price model uses -- the decode step is fixed + per-sequence
%%    KV read; which term moves, by how many bytes or launches.
%% 2. The measurement that tests it: \Cref{tab:evidence}. The ablation
%%    (price with the kill switch on vs as is vs baseline), the profile
%%    before/after (tracedb: did the kernel you targeted get shorter AND the
%%    step), the ncu counters (did DRAM%% move toward the roofline).
%% 3. The number: the share of the delta the mechanism accounts for, and
%%    whether the disabled stack sits within the 3%% noise floor of baseline.
%% An evidence row that says "not measured" is a gap: say so in \Cref{sec:limits}.
\begin{table}[tbp]
\centering
\small
\caption{Evidence for the mechanism. Each row is a file in the run
directory; a row reading \emph{not measured} is an unexplained part of the
delta.}
\label{tab:evidence}
\begin{tabular}{@{}p{0.17\linewidth}p{0.47\linewidth}p{0.28\linewidth}@{}}
\toprule
\textbf{Evidence} & \textbf{Measurement} & \textbf{Source} \\
\midrule
%(evidence_rows)s
\bottomrule
\end{tabular}
\end{table}

\section{What changed}
\label{sec:change}
%% Files, the launch line if changed (serving.json), and the design choices
%% that mattered. What the kill switch (the ablation's env) turns off.

\section{Results}
\label{sec:results}
%% \Cref{tab:attempts}, then the figures. Numbers from the run, copied, not
%% rounded to look better; a screen is judged against stock at screen tier.
%% Name the replicate and which of the two runs the harness kept (the worse).
\begin{table}[tbp]
\centering
\small
\caption{Priced attempts. Baseline \$%(baseline)s/1k at full tier.}
\label{tab:attempts}
\begin{tabular}{@{}llrrrl@{}}
\toprule
\textbf{attempt} & \textbf{tier} & \textbf{\$/1k} & \textbf{$\Delta$\%%} & \textbf{N*} & \textbf{gates} \\
\midrule
%(attempt_rows)s
\bottomrule
\end{tabular}
\end{table}
%(figure_blocks)s

\section{What the gates said}
\label{sec:gates}
%% GSM8K, LongBench F1, MMLU; token equivalence (top-1 agreement, |dlogprob|,
%% decode agreement and whether that is lossless or lossy); and whether the
%% path under test was actually exercised by them. Lossy is allowed when the
%% accuracy suites hold; say which it was.

\section{Limitations and what is unexplained}
\label{sec:limits}
%% What the ablation did not cover, what was not measured, where the numbers
%% disagree, and what part of the delta has no mechanism behind it.

\section{Next}
\label{sec:next}
%% The single next experiment this result argues for, and what it would cost.
\end{document}
"""

_PROMPT = """Write the paper for the idea you just finished, as `PAPER.tex` in this
directory. Read `.claude/skills/writeup/SKILL.md` first: it says what a
write-up is here, what the publishable bar is, and what not to do.

Start from `PAPER.tex` as it is: a template with the title, the evidence
table, the attempts table and the figures already filled. Replace each `%%`
comment with prose and leave the preamble, the structure and the pre-filled
table rows alone (you may add a row to the evidence table when a file
supports it). Two columns, at most two pages. No new packages, no \\input.

**Every claim cites a file.** The run directory is
  {run_root}
Each number in the prose is followed by \\src{{<path relative to the run
directory>}} naming the file it was read from: a report.txt, result.json,
ablation.json, an equivalence record, a workbench stdout.txt, a profile
database. Refuse to write a number you cannot cite -- write "not measured"
and name the tool that would measure it. Copy numbers; do not round them to
look better. If no ablation exists, the mechanism section ends "not ablated;
the delta is unexplained". Files present in the run directory:
{files}

Idea: {title}
Hypothesis: {hypothesis}
Design note (if any; what was intended, not what happened):
{design}

Attempts:
{attempts}

Evidence (what results.py found on disk; the table is filled from this):
{evidence}

Best diff (truncated):
```
{diff}
```

Figures available (already \\includegraphics'd in the template): {figures}

When done, reply with two sentences: the paper's conclusion, and which of
the four publishable conditions (win, replicated, gates held, ablation
explains it) hold and which are missing."""


@dataclass(frozen=True)
class PaperInputs:
    title: str
    author: str
    attempts: list[dict]         # {n, tier, bill, delta, n_star, gates}
    baseline: float | None
    figures: list[pathlib.Path]
    # `results.evidence_for` for the best stack: ablation, decode agreement,
    # profile, replicate, gates. Empty when nothing is on disk.
    evidence: dict = field(default_factory=dict)
    # The agent directory, so evidence paths are cited relative to it.
    run_root: str = ""


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


def run_files(agent_root: str | pathlib.Path, limit: int = 60) -> list[str]:
    """The citeable files under an agent directory, relative: reports,
    results, ablations, equivalence records, workbench output, profiles.
    What the paper prompt lists so the model cites what exists."""
    root = pathlib.Path(agent_root)
    if not root.is_dir():
        return []
    pats = ("runs/*/report.txt", "runs/*/result.json", "runs/*/stack.json",
            "ablations/*/ablation.json", "equivalence/*.json",
            "workbench-*/stdout.txt", "workbench-*/script.py",
            "../profiles/*.sqlite", "DESIGN.md", "candidate/sglang/DESIGN.md")
    out: list[str] = []
    for pat in pats:
        for f in sorted(root.glob(pat)):
            if f.is_file():
                out.append(os.path.relpath(f, root))
    return out[:limit]


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def _rel(path: str, run_root: str) -> str:
    if not path:
        return ""
    if run_root:
        try:
            return os.path.relpath(path, run_root)
        except ValueError:
            return path
    return path


def evidence_rows(ev: dict, run_root: str = "") -> str:
    """The evidence table's rows: ablation, profile, equivalence. A row with
    nothing behind it says *not measured*, never a placeholder number."""
    ev = ev or {}
    rows = []

    def row(kind: str, measurement: str, source: str) -> None:
        src = f"\\path{{{source}}}" if source else "--"
        rows.append(f"{kind} & {measurement} & {src} \\\\")

    abl = ev.get("ablation")
    if abl:
        on = (abl.get("as_is") or {}).get("bill_per_1k")
        off = (abl.get("disabled") or {}).get("bill_per_1k")
        env = ", ".join(f"{k}={v}" for k, v in sorted((abl.get("env") or {}).items()))
        exp = abl.get("explained_pct")
        base = abl.get("baseline_bill_per_1k")
        parts = [f"{abl.get('tier', '?')} tier, kill switch {_tex_escape(env)}."]
        if on is not None and off is not None:
            parts.append(f"As is \\${on:.2f}/1k, disabled \\${off:.2f}/1k"
                         + (f", baseline \\${base:.2f}/1k" if isinstance(base, (int, float)) else "")
                         + ".")
        if exp is not None:
            parts.append(f"The mechanism accounts for {exp:.0f}\\% of the delta; the disabled "
                         f"stack is {'within' if ev.get('explains') else 'outside'} the 3\\% "
                         "noise floor of baseline.")
        elif not (on is not None and off is not None):
            parts.append("A sweep did not price: " + _tex_escape(
                (abl.get("as_is") or {}).get("reason") or (abl.get("disabled") or {}).get("reason") or "?"))
        row("Ablation", " ".join(parts), _rel(abl.get("path") or "", run_root))
    else:
        row("Ablation", "\\emph{not measured}: no ablation was run, so the delta has no "
                        "mechanism behind it (\\texttt{harness tool ablate --env KEY=VAL}).", "")

    prof = ev.get("profile_db")
    if prof:
        row("Profile", "Full-sweep GPU trace (tracedb). Fill from \\texttt{trace\\_ops\\_grouped} "
                       "on stock's and this profile: the targeted kernel's \\si{\\micro\\second} "
                       "per step before and after, and the step itself.",
            _rel(prof, run_root))
    else:
        row("Profile", "\\emph{not measured}: no profile was captured for this stack "
                       "(screen tier, or profiling off).", "")

    dec = ev.get("decode_agreement")
    if dec is not None:
        label = "lossless" if ev.get("lossless") else "lossy (floor for a correct kernel 0.84)"
        row("Equivalence", f"Decode agreement {dec:.4f} against stock's greedy generation: {label}. "
                           f"Accuracy gates: {_tex_escape(str(ev.get('gates', 'not scored')))}.",
            _rel(ev.get("equivalence_path") or "", run_root))
    else:
        row("Equivalence", "\\emph{not measured}: token equivalence was not run. Accuracy gates: "
                           f"{_tex_escape(str(ev.get('gates', 'not scored')))}.", "")
    return "\n".join(rows)


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
        blocks.append(f"\\begin{{figure}}[tbp]\\centering\\includegraphics[width=\\linewidth]"
                      f"{{{dst.name}}}\\caption{{{_tex_escape(fig.parent.name + ': ' + fig.stem)}}}"
                      f"\\label{{fig:{i}}}\\end{{figure}}")
    import datetime as dt

    text = TEMPLATE % {
        "title": _tex_escape(inp.title), "author": _tex_escape(inp.author),
        "date": dt.date.today().isoformat(),
        "attempt_rows": "\n".join(rows) or "-- & -- & -- & -- & -- & -- \\\\",
        "baseline": "-" if inp.baseline is None else f"{inp.baseline:.2f}",
        "figure_blocks": "\n".join(blocks),
        "evidence_rows": evidence_rows(inp.evidence, inp.run_root),
    }
    p = out_dir / "PAPER.tex"
    p.write_text(text)
    return p


def prompt_for(inp: PaperInputs, hypothesis: str, design: str, diff: str,
               files: list[str] | None = None) -> str:
    ev = inp.evidence or {}
    abl = ev.get("ablation") or {}
    ev_lines = [
        f"  replicated: {'yes' if ev.get('replicated') else 'no'}",
        f"  accuracy gates: {ev.get('gates', 'not scored')}",
        ("  ablation: " + (abl.get("verdict") or "present") if abl
         else "  ablation: none -- the delta is unexplained"),
        ("  equivalence: decode agreement "
         f"{ev['decode_agreement']:.4f} ({'lossless' if ev.get('lossless') else 'lossy'})"
         if ev.get("decode_agreement") is not None else "  equivalence: not run"),
        f"  profile: {ev.get('profile_db') or 'none captured'}",
    ]
    if files is None:
        files = run_files(inp.run_root) if inp.run_root else []
    return _PROMPT.format(
        title=inp.title, hypothesis=hypothesis,
        design=design or "(none)",
        run_root=inp.run_root or "(unknown; cite paths relative to the agent directory)",
        files="\n".join(f"  {f}" for f in files) or "  (none found)",
        attempts="\n".join(f"  attempt {a.get('n')} ({a.get('tier')}): ${a.get('bill')}/1k, "
                           f"{a.get('delta')}% vs baseline, N*={a.get('n_star')}, gates: {a.get('gates')}"
                           for a in inp.attempts) or "  (none priced)",
        evidence="\n".join(ev_lines),
        diff=diff[:8000] or "(no diff recorded)",
        figures=", ".join(f"fig{i}{f.suffix} ({f.parent.name}/{f.stem})"
                          for i, f in enumerate(inp.figures)) or "none")


def compile_tex(tex: str | pathlib.Path, timeout_s: float = 240.0) -> pathlib.Path | None:
    """PDF beside the .tex, or None when tectonic is missing or fails.
    Failure is logged next to the source as `paper.log`, never raised."""
    # Absolute: tectonic runs with the paper directory as cwd, and a relative
    # `-o agents/<s>/a02/paper/<idea>` resolved from inside that directory
    # is exactly the "output directory does not exist" build-4 logged twice.
    tex = pathlib.Path(tex).resolve()
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


def ensure_pdf(tex: str | pathlib.Path) -> pathlib.Path | None:
    """`paper.pdf` for a `PAPER.tex`, compiled now if missing or older than
    the .tex. None when tectonic is absent or the compile fails; the .tex
    is still there and `paper.log` says why."""
    tex = pathlib.Path(tex)
    pdf = tex.parent / "paper.pdf"
    if pdf.is_file() and pdf.stat().st_mtime >= tex.stat().st_mtime:
        return pdf
    return compile_tex(tex)


def find_papers(root: str | pathlib.Path, compile: bool = False) -> dict[str, pathlib.Path]:
    """idea id -> paper.pdf (or PAPER.tex when uncompiled) under a fleet root.
    With `compile`, every .tex without a current PDF is compiled first, so
    the caller gets PDFs wherever tectonic can make one."""
    out: dict[str, pathlib.Path] = {}
    for d in pathlib.Path(root).glob("a*/paper/*"):
        pdf = d / "paper.pdf"
        tex = d / "PAPER.tex"
        if compile and tex.is_file():
            ensure_pdf(tex)                 # leaves paper.pdf beside the .tex
        if pdf.is_file():
            out[d.name] = pdf
        elif tex.is_file():
            out[d.name] = tex
    return out
