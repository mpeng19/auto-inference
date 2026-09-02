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
mkdir -p runs/baseline-2026-09-02
uv run simulate run --root runs/baseline-2026-09-02 \
  --levels 4,8,12,16,24 --seconds 120 --n-gpu 1
```

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

Record the two numbers. They are the fleet's `--baseline`; the quality map is what makes the gate fire at all.

## Step 2 — three agents, overnight (~8h, cap $60)

Three, not ten: the point is to find out whether the loop runs at all, and
three is enough to exercise seeding diversity, dedup and the eval queue while
keeping the bill legible.

```bash
uv run harness --session night-1 start \
  --agents 3 --evals 2 --model sonnet \
  --budget 60 --agent-budget 20 --max-attempts 3 \
  --root agents/night-1 \
  --baseline '{"bill_per_1k": <from step 1>, "quality": {"gsm8k": <from step 1>}}' \
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
