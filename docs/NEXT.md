# Next: a small real run

Written 2026-09-02. Scope: what to run first and what to check. This is a
next-steps note, not a running log — findings go in docstrings and
`docs/methodology.md`.

## The honest state

The tune-mode loop (nights 1-5, 2026-09-02) runs end to end: fifty-odd
evaluations, honest verdicts, no wins. It produced one-line scheduler tweaks
because that is what it asked for. The build-mode loop -- idea bank, GPU
workbench, token-level gate, build prompt, manager -- is written and tested
but **no build-mode agent has run yet**, and the workbench has run one smoke
script. Treat the first build run as a systems test that may produce a kernel.

## Step 1 -- baselines (once per grid; done for 2026-09-02)

```bash
mkdir -p runs/baseline runs/baseline-screen
uv run simulate run --root runs/baseline --levels 4,8,12,16,24 --seconds 120 --n-gpu 1
uv run simulate run --root runs/baseline-screen --levels 8,12 --seconds 60 --n-gpu 1
```

Read the interpolated frontier line as well as N*. The stock numbers on this
grid are $12.23/1k full (stock on its good days; see below), $17.30 screen,
GSM8K 0.69.

## Step 2 -- the equivalence reference and its noise floor (once per model)

```bash
mkdir -p runs/equiv-ref runs/equiv-noise
uv run simulate equivalence --root runs/equiv-ref       # stock: writes the cached reference
uv run simulate equivalence --root runs/equiv-noise     # stock again: the noise floor
```

The second run's agreement and mean |dlogprob| are what a candidate must
stay inside; the thresholds in `measure/equivalence.py` (0.97, 0.05) are
provisional until this has run.

## Step 3 -- fill the bank (once; grows over time)

```bash
uv run harness ideas import docs/ideas/book.jsonl --source book   # 27 records, committed
uv run harness ideas arxiv -k 15 --model opus                       # ~30 min of model calls
uv run harness ideas list
```

Records carry mechanism, targets, expected gain and risks. `harness ideas
show <id>` before believing one.

## Step 4 -- three build-mode agents, overnight

```bash
uv run harness --session build-1 start \
  --agents 3 --evals 2 --model opus --mode build --bank --manager \
  --budget 200 --agent-budget 70 --max-attempts 4 \
  --root agents/build-1 \
  --baseline '{"bill_per_1k": 12.23, "quality": {"gsm8k": 0.69}, "screen": {"bill_per_1k": 17.30}}'
```

No `--seed`: agents claim from the bank, least-similar first, one mechanism
each. Opus, because a kernel is not a knob. Four attempts, because an attempt
is now hours: design note, workbench correctness and micro-benchmark,
equivalence, then the sweep. The manager reviews every third outcome and
stashes a tool under `agents/build-1/tools/` only when it can name the hours
it saves; agents see the index in their prompt.

**Leave the lid open.** The daemon runs under `caffeinate`, which does not
survive clamshell sleep; the status line prints `host slept` if it happens.

## In the morning

```bash
uv run harness --session build-1 status
uv run harness traces list --root agents/build-1
uv run harness traces show <id> --kind eval_submit --full     # the diffs
uv run harness ideas list --status tried
ls agents/build-1/tools/                                     # what the manager stashed
```

Judge: did each agent write a DESIGN.md and run the workbench before the
sweep (`tool_call` turns, `denials` = 0 in the call stats)? Did any kernel
pass equivalence? Did the manager stash anything, and was it worth it? A
price move is a result only if it is outside the interpolated frontier's
noise and replicated.

## What the first night found (2026-09-02)

- **Warm containers carried the previous stack's files.** `apply()` backed
  up each file it overwrote and nothing ever restored it. Modal reuses a
  container between back-to-back calls, so every evaluation after the first
  in a container ran on top of the last one's diff, a stock run in a warm
  container was not stock, and a stack touching a file the previous one had
  touched was refused as "stale". Fixed: `apply()` restores stock first and
  records `restored`. **Every fleet number from night-1 to night-3 is
  suspect**; the two full baselines ran alone in fresh containers and stand.
  night-4 is the first fleet whose measurements are attributable.

- **The baseline must be stock on its good days.** With N=12 on the SLO
  line, stock prices $12.2 when it holds and $15.0 when it does not, and it
  held in four of five sweeps on 2026-09-02. A single stock run is a draw
  from that; the $14.96 run was the unlucky one, and against it a no-op diff
  is a replicated 18% win. Until N* is interpolated, the fleet baseline is
  the **best** stock measurement on the grid ($12.23, `docs/examples/
  baseline-1xh100`), and the screen baseline the best stock-equivalent
  screen ($17.30 of 17.30-17.52). A win beats that or it is not a win.
- **The frontier is quantised to the grid, and that manufactures wins.**
  A no-op diff scored 18% below baseline: N=12 held 20 ms mean TPOT in its
  sweep (19.x ms) and missed in stock's (22.0 ms), so N* moved from 8 to 12
  and every other number matched stock to the cent. August's $12.23 and
  September's $14.96 for stock are the same flip. The harness now replicates
  a claimed win and keeps the worse run, which halves the false-win rate but
  does not remove it. The real fix is in the simulator: interpolate N* where
  the fitted SLO curve crosses the limit and price there, instead of taking
  the last passing level. `simulate rescore` could then re-judge every sweep
  from the night without a GPU. Decide this before believing any win.
- **A screen is not a small full sweep.** Stock prices ~15% higher at screen
  tier; screens must be judged against stock measured the same way.
- **GSM8K is noisy at n=50** (62% and 70% on stock, same items); at n=100 it
  has held 69-70%. The gate is 10 points.
- **Spend must land per attempt**, or the dashboard and the budget are blind
  for an entire idea.

## The laptop is part of the fleet

The daemon and every agent run on the machine that is logged in to Claude
Code. When it sleeps they freeze, their timeouts freeze with them, and the
sweeps they submitted keep billing on Modal. On 2026-09-02 a closed lid at
08:39 held three agents for five hours: night-5 averaged one evaluation an
hour overnight and six an hour once the lid was open. `harness start` now
runs the daemon under `caffeinate -i -s`, which prevents idle and AC sleep
but **not clamshell sleep** -- leave the lid open, or attach an external
display. The status line and TUI print `host slept ~N min` when it happens.

## Known gaps, worth fixing before scaling to ten

- **No profile is captured by default.** `--profile-level` exists and is
  untested. Until it runs, agents reason about where decode time goes instead
  of looking (`src/tracedb/`).
- **`bench_serving` cross-check has never run.** One invocation at N* would
  tell us whether our load generator agrees with SGLang's own.
- **Seeding costs a full `claude -p` call** (~30–60 s). `--seed-model` exists
  to make that cheaper and is untuned.
- **Memory is unproven at fleet scale.** `agent-db` measured retrieved facts as
  a *clean null* out-of-sample; the brief is the condition that beat placebo.
  Whether it helps here is an open question, and the placebo arm is not built.
