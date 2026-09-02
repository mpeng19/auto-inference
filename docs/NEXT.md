# Next: a small real run

Written 2026-09-02. Scope: what to run first and what to check. This is a
next-steps note, not a running log — findings go in docstrings and
`docs/methodology.md`.

## The honest state

Everything below has been exercised end to end **except the GPU path**. The
simulator's analysis reproduces the 1xH100 baseline exactly from a stored
sweep; the fleet, TUI, control plane, memory, traces and tools have all run
against real Claude Code agents. But since `frontier` was rewritten as
`sweep()`, **nothing has actually rented a GPU**. Specifically unverified:

- `runner.sweep()` — rewritten, deployed, never executed
- the quality gate against a live server (`/v1/completions`, GSM8K scoring)
- profile capture (`/start_profile`) and `simulate profile`
- `InferenceStack.apply()` on a real container with a real diff
- the whole chain: agent workspace -> stack -> Modal -> price -> memory

So the first run is a **systems test that happens to produce a number**, not a
result. Treat any price it prints as unconfirmed until a second run agrees.

## Step 1 — a baseline sweep (~45 min, ~$3)

Agents need two numbers before they can be scored: the baseline bill and the
baseline accuracy. Both come from one stock run. Do this first and alone, so a
failure is attributable.

```bash
mkdir -p runs/baseline runs/baseline-screen
uv run simulate run --root runs/baseline --levels 4,8,12,16,24 --seconds 120 --n-gpu 1
# stock at the fleet's *screen* tier, on the same grid a screen uses
uv run simulate run --root runs/baseline-screen --levels 8,12 --seconds 60 --n-gpu 1
```

Two runs, because a screen is not a small full sweep: its price carries
warm-up that 120 s levels amortise, and stock priced ~15% higher at screen
tier on 2026-09-02. A screen is compared with stock measured the same way or
it can never be promoted.

Check, in order:

1. It completed. If it died, read `runs/.../sweep.json` `failure` and the
   `server_log_tail` — the launch traps are model-spec and parallelism ones
   (`docs/methodology.md` §6).
2. `report.txt` has `quality gsm8k: NN%`. **If quality is missing or errored,
   stop** — the gate is the only thing standing between a speed win and a
   worse model, and a fleet without it is worse than no fleet.
3. `N*` is bracketed, i.e. some level passes and some fails. If every level
   passes, the sweep never found the frontier and the price is an upper bound;
   raise the top level and re-run.
4. Sanity: `busy_frac` near 1.0 (`::sanity`), and `eff-in`/`out` within a
   factor of two of $0.029/M and $5.60/M. Wildly different means something
   structural changed, not that we got faster.

Record the three numbers. They are the fleet's `--baseline`, and `harness start`
refuses to run without all of them: each missing one is a way the fleet runs
all night and learns nothing.

## Step 2 — three agents, overnight (~8h, cap $60)

Three, not ten: the point is to find out whether the loop runs at all, and
three is enough to exercise seeding diversity, dedup and the eval queue while
keeping the bill legible.

```bash
uv run harness --session night-1 start \
  --agents 3 --evals 2 --model sonnet \
  --budget 60 --agent-budget 20 --max-attempts 3 \
  --root agents/night-1 \
  --baseline '{"bill_per_1k": <full>, "quality": {"gsm8k": <acc>}, "screen": {"bill_per_1k": <screen>}}' \
  --seed "raise the decode batch the SLO permits" \
  --seed "reduce per-sequence KV bytes read per decode step" \
  --seed "improve prefix cache hit rate under load"
```

The seeds are not arbitrary. `docs/methodology.md` §8.3 says the per-sequence
KV term runs at 0.22–0.28 of memory bandwidth and **does not amortise with
batch** — it is the one term a TPOT SLO converts directly into money. The
second seed aims at it; the other two are controls.

Watch it with `harness tui`. Leave it.

### Cost arithmetic, so the cap is a decision and not a hope

A full sweep is ~45 min of 1xH100 at Modal retail ($3.95/hr) ≈ **$2.90**; a
screen is ~13 min ≈ **$0.85**. With `--evals 2` the fleet can burn about
$7.70/hour flat out, so $60 is roughly an 8-hour night. `--budget` is the hard
stop; `--agent-budget 20` stops one agent monopolising it.

## In the morning

```bash
harness status                       # or: harness --session night-1 status
harness traces list --root agents/night-1
harness traces show <id> --kind eval_submit --full   # the diffs they wrote
harness tool recall "decode batch" --root agents/night-1
```

Judge the *system*, not the science. Three attempts each is nowhere near enough
to find a real improvement, and a win on one sweep is noise until replicated.
What matters:

- Did every agent get an idea, write a diff, and get it priced?
- Did the eval queue stay busy (`gpu_utilisation`), or did agents stall?
- Did `cost_usd` track something plausible, and did the budget bind?
- Did anything reach memory, and can a second agent read it back?
- Did any diff fail `preflight` or the quality gate, and was that correct?

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
