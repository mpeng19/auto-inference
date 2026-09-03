---
name: writeup
description: How to write the paper for an idea that reached a full sweep. Every claim traces to a file in the run directory; the mechanism section is a hypothesis plus the measurement that tested it; a win is publishable only when an ablation explains it. Read before writing or editing PAPER.tex or DESIGN.md's results section.
---

# writeup: the paper is an argument over files, not a summary of an idea

The harness compiles `PAPER.tex` into a two-page, two-column PDF for every
idea that reached a full sweep. The template is filled with the numbers
(attempts table, evidence table, figures) before you see it; you write the
prose. This skill says what that prose is.

## What a write-up is here

**Every claim traces to a measurement in the run directory, and names the
file.** The run directory is the agent directory (`<agent>/`); its files are
the only admissible evidence:

| claim about | file |
|---|---|
| a price, N\*, rank, share | `runs/attempt-NNN/report.txt`, `runs/attempt-NNN/result.json` |
| a replicate | `runs/attempt-NNN-rep1/report.txt` |
| the mechanism's share of the delta | `ablations/<n>/ablation.json` (`harness tool ablate`) |
| where decode time went, before/after | `../profiles/<digest>.sqlite` via the tracedb tools; cite the query |
| hardware counters (DRAM%, occupancy) | `workbench-<n>/stdout.txt` from `harness tool ncu` |
| a micro-benchmark or correctness check | `workbench-<n>/stdout.txt`, `workbench-<n>/script.py` |
| decode agreement, top-1, lossless/lossy | `equivalence/<digest>-<ts>.json` |
| accuracy gates | `runs/attempt-NNN/result.json` (`quality`) |
| what changed | the diff in the trace (`eval_submit`), `runs/attempt-NNN/stack.json` |

Write the citation as `\src{runs/attempt-002/report.txt}` after the number.
A number with no file behind it is not written; say **not measured** and
name the tool that would measure it.

**The mechanism section is three things, in order:**

1. the **hypothesis** -- why this should lower cost per output token, in the
   price model's terms (the decode step is a fixed weight read plus a
   per-sequence KV read; which term moves, by how many bytes or launches);
2. the **measurement that tests it** -- the ablation (`ablation.json`: the
   price with the mechanism's kill switch on, against as-is and baseline),
   the profile before and after (`trace_ops_grouped` on stock's and your
   sqlite: did the kernel you targeted get shorter *and* did the step), the
   `ncu` counters (did DRAM% move toward the roofline);
3. the **number** -- the share of the delta the mechanism accounts for, from
   `ablation.json`'s `explained_pct`, and whether the disabled stack sits
   within the 3% noise floor of baseline (`explains`).

**Negative results and unexplained deltas are reported as such.** A stack
that is faster with no ablation is "faster, unexplained". An ablation whose
disabled price is still well under baseline means something else in the
diff is doing the work: say that, and say what. A replicate that came back
worse is the number to report (the harness keeps the worse run). A lossy
equivalence result is not a defect to hide: state the decode agreement and
that the accuracy suites held (or did not).

**Numbers are copied from the reports, never rounded to look better.**
`$8.89/1k` is not "about $8.9"; `-27.3%` is not "nearly 30%"; a GSM8K score
of 0.69 on 100 items is "69% (n=100)". If two runs disagree, give both.

## The publishable bar

A result is **publishable** (the `pub` column in `harness results`) when all
four hold:

1. the verdict is a **win**: at least 3% under stock's bill at the same tier;
2. it was **replicated**: measured twice at full tier, and the worse run is
   the one reported;
3. the **accuracy gates held**: GSM8K, LongBench and MMLU were scored and did
   not reject (a rejected stack is never a win);
4. an **ablation explains it**: with the mechanism's kill switch on, the
   price returns to within 3% of baseline, so the mechanism -- not something
   else in the diff -- is the delta.

Lossless is a label, not a requirement: a lossy kernel with the gates held is
allowed. Say which it was.

The paper's last sentence, and your reply, says which of the four hold and
what is missing.

## What NOT to do

- **A prose dump of the design note.** DESIGN.md said what you intended; the
  paper says what happened. Sentences from the design note that were not
  tested belong in "Limitations and what is unexplained", marked as untested.
- **Restating the idea as if it were the result.** "The kernel reads 4x fewer
  bytes" is a claim about the design; the result is the measured step time
  and the price, with a file behind each.
- **Claiming a mechanism the ablation did not confirm.** If there is no
  `ablation.json`, the mechanism section ends with "not ablated; the delta is
  unexplained". If `explains` is false, say what the disabled stack priced at
  and that the mechanism is not the whole story.
- **Blaming the baseline.** Stock's price at each tier is measured the same
  way yours is, on the same grid, the same day. A screen is judged against
  stock at screen tier; if your number looks bad next to the wrong baseline,
  cite the right one, do not argue with it.
- **Estimating.** No extrapolation from a screen to a full sweep, from one
  batch size to N\*, from a micro-benchmark speedup to a price.
- **New packages or `\input`** in PAPER.tex, or changes to the preamble and
  the pre-filled tables. The template compiles; keep it that way.
