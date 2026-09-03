# Next: a small real run

Written 2026-09-02, updated 2026-09-03. Scope: what to run first and what to
check. This is a next-steps note, not a running log — findings go in
docstrings and `docs/methodology.md`.

## The honest state

The tune-mode loop (nights 1-5, 2026-09-02) ran end to end: fifty-odd
evaluations, honest verdicts, no wins. It produced one-line scheduler tweaks
because that is what it asked for.

Build mode has now run twice. `build-1` recorded no experiments: a workspace
fault drained six bank records in a minute, which is why an unmeasured error
now returns its idea to the bank. Its manager still stashed three tools.
`build-2` recorded nine experiments, stashed one more tool, and produced one
replicated **-27.3%**; read the section below before believing it. The loop
works. Its numbers are claims.

## What build-2 found (2026-09-02)

- **One replicated win, and the gate that cannot see it.** `a01`, stack
  `b0027aa57534`: "read a P-greedy-selected nested subset of KV pages instead
  of the full KV cache". Screen $16.76/1k (N\*=12), then two full sweeps at
  **$8.887** and **$8.879**/1k, both N\*=16, interpolated N\* 17.98 and 18.69.
  The worse is kept: **-27.3%** against $12.23, rank 3 of 12, one node 0.58% of
  the market. ~$8.3 of GPU across the three. Two sweeps agreeing to 0.1%, and
  an interpolated frontier well past the last grid point, means this is not the
  grid flip that manufactured the earlier 18%.

  What is *not* established is that it is the same model. Reading a subset of
  KV pages is an approximation, and GSM8K scored 68% and 67% against a 69%
  baseline — inside a 10-point gate that cannot resolve a numerics change at
  n=100. No `harness tool equivalence` score for this digest is in the run.
  **Run equivalence against `b0027aa57534` before this leaves the repository**;
  a 27% price cut that answers differently is a different product, not a win.

- **The manager writes tools but has not yet written a fact.** All twelve
  facts in the skill bank are `source=human`; nothing is attributed to a
  session, so the review's fact path has never landed one. It is the half of
  the manager that carries across runs, so check `harness skills list` for a
  fact sourced to the session before assuming it works.

- **`serving.json`'s `env` reaches the sweep, not the workbench.** The runner
  applies `stack.env` to the served process; `workbench` builds the script's
  environment from the container's own. So a change behind an environment flag
  is measured with the flag off: an agent's env-gated numerics change scored
  top-1 agreement exactly 1.0000 with |dlogprob| exactly 0.0000, which is
  impossible for a change that re-rounds every weight and is the tell that the
  candidate ran stock. It cost that agent a workbench run and an equivalence
  run to find. The fix belongs in `simulator/runner`; until it lands, the build
  prompt and `harness.tools` say to export the variable inside the script, or
  to make the change default-on with a kill switch.

## Step 1 -- baselines (once per grid and runner; done for runner v3 on 2026-09-03)

```bash
mkdir -p runs/baseline runs/baseline-screen
uv run simulate run --root runs/baseline        --levels 4,8,12,16,24 --seconds 120 --n-gpu 1
uv run simulate run --root runs/baseline-screen --levels 8,12         --seconds 60  --n-gpu 1
```

Two runs, because a screen is not a small full sweep, and both again
whenever the runner's measurement changes. Runner v3 ends each level at its
deadline and waits for the server to go idle before the next, which made a
full sweep ~25 minutes and moved stock's numbers down (drained levels had
been charging the tail of every reply to the price):

| stock, runner v3 | N* | $/1k | interpolated |
|---|---|---|---|
| full, 5 x 120 s, run 1 | 8 | 9.77 | N*~11.7, $8.54 |
| full, 5 x 120 s, run 2 | 12 | 8.32 | N*~12.2, $8.23 |
| screen, 2 x 60 s | 12 | 9.03 | |
| quality | gsm8k 0.64-0.70, longbench 0.53, mmlu 0.64 | | |

Old numbers ($12.23 / $14.96 full, $17.30 screen) are from drained levels
and are not comparable with anything measured now. Two things follow. The
fleet baseline is stock on its good day, $8.32. And at that price stock is
**rank 1 of 12** on the OpenRouter board: the drained tail had been adding
~30% to every price and pushing stock to rank 9. A win now has to beat a
stack that already undercuts every provider.

## Step 2 -- the equivalence reference (built on first use)

```bash
uv run simulate equivalence --root runs/equiv-ref --mkdir      # stock; ~5 min, ~$0.40
```

The reference is long-context (LongBench, ~15k tokens) so anything gated on
sequence length is scored while it is running. Stock against itself is
1.0000 / 0.0000 exactly. Note that `--stack` must point at a *saved* stack: a
run directory (it now holds `stack.json`) or a mirrored tree. Pointing it at
an agent's `candidate/sglang` after the agent moved on measures stock, which
is how build-2's -27% win went unverified and then unrecoverable.

## Step 3 -- fill the bank (once; grows over time)

```bash
uv run harness ideas import docs/ideas/book.jsonl --source book
uv run harness ideas arxiv -k 15 --model opus
uv run harness ideas list
```

## Step 4 -- five build-mode agents, until the money runs out

```bash
uv run harness --session build-4 start \
  --agents 5 --evals 3 --model opus --mode build --bank --manager \
  --budget 300 --agent-budget 80 --max-attempts 6 \
  --root agents/build-4 \
  --baseline '{"bill_per_1k": 8.32, "quality": {"gsm8k": 0.66, "longbench": 0.53, "mmlu": 0.64}, "screen": {"bill_per_1k": 9.03}}'
```

Five agents on three GPUs: with screens at ~10 minutes and full sweeps at
~25, three slots keep five opus agents fed. `--budget` is the Modal cap;
`--agent-budget` stops one agent monopolising it; six attempts because an
attempt is hours. Leave the lid open.

## In the morning

```bash
uv run harness --session build-3 status
uv run harness --session build-3 results --diff              # experiments, best first
uv run harness traces list --root agents/build-3
uv run harness traces show <id> --kind eval_submit --full     # the diffs
uv run harness ideas list --status tried
uv run harness skills list                                   # facts the manager wrote
ls agents/build-3/tools/ agents/build-3/profiles/            # stashed tools; ingested profiles
uv run harness --session build-3 timeline                    # who did what, when
uv run harness --session build-3 calls -v                    # where the agent-hours went
uv run harness --session build-3 paper                       # the write-ups
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
  does not remove it. The simulator now also **interpolates**: every report
  and every experiment's metrics carry where the binding metric crosses its
  limit between the last passing and first failing level, with the bill
  interpolated to match. It does not yet *price* there -- the priced point is
  still the last passing level -- so the interpolated line is a second opinion
  on a win rather than the verdict. Read both; build-2's -27.3% is the first
  claim where they agree. `simulate rescore` re-judges a stored sweep without
  a GPU if that rule changes.
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

- **No profile has ever been captured.** `--profile-level` now defaults to 12
  and the ingest and MCP wiring are tested offline, but both build runs were
  launched from daemons started before that landed, and no sweep record
  carries a `profiles` key. Until one does, agents reason about where decode
  time goes instead of looking (`src/tracedb/`). First check next run:
  `ls agents/<session>/profiles/`, and grep the daemon log for
  `profile ingest skipped`.
- **`bench_serving` cross-check has never run.** One invocation at N* would
  tell us whether our load generator agrees with SGLang's own.
- **Seeding costs a full `claude -p` call** (~30–60 s). `--seed-model` exists
  to make that cheaper and is untuned.
- **Memory is unproven at fleet scale.** `agent-db` measured retrieved facts as
  a *clean null* out-of-sample; the brief is the condition that beat placebo.
  Whether it helps here is an open question, and the placebo arm is not built.
