# auto-inference

Price an LLM serving stack the way a marketplace would.

Give it an **inference stack** — stock SGLang, or stock plus a set of diffs to
`srt/` — and it answers one question: *what effective price can this stack
serve marketplace traffic at, and how much of that market can it hold?*

```python
from simulator import Simulator, InferenceStack

sim = Simulator(root_dir="runs/my-diff",
                stack=InferenceStack.from_dir("my-diff/"))   # or .stock()
result = await sim.eval()
print(result.summary())
```

```
N* = 12 users   batch 5.0   hit 0.748   7.34 GPU-s per market request
  effective input  $0.0294/M
  output           $5.6021/M
  whole bill       $12.23 per 1k requests   rank 9/12
  one node serves  0.42% of the market  (5,885 req/day)
  on effective input alone: rank 1/12  -- the metric OpenRouter sorts on,
  not what a buyer pays
```

Every artifact lands in `root_dir`: the raw sweep, the priced curve, a report,
and one figure per question — a `slo-<bound>.png` for each SLO bound (aggregate
throughput on y, the constraint on x, one point per concurrency, the forbidden
region shaded), plus `price-vs-share.png` and `price-vs-demand.png`.

## Setup, from a fresh clone

```bash
uv sync                    # dependencies
uv run modal token new     # your own Modal account -- nothing here is shared
make deploy                # push the runner
make test                  # 112 tests, no GPU, no network
```

That is the whole setup. No Hugging Face token is needed: the default
checkpoint is Apache-2.0 and ungated. No account-specific names are baked in —
the Modal volumes are created on first use in whichever workspace you are
logged into. Override any of it if you want to:

```bash
export SIMULATOR_HF_SECRET=huggingface     # only for a gated model
export SIMULATOR_APP_NAME=my-app
export SIMULATOR_MARKET_DATA=/path/to/snapshot.json
```

To price a **different model**, set `model=` and pull that model's market data
with `make market` — the snapshot supplies the demand denominator and the
provider board, and both are model-specific.

Start with **`docs/example.ipynb`**, which runs the whole thing end to end
against a stored sweep in a few seconds. `docs/examples/baseline-1xh100/` is
what a real run leaves behind.

## The method

1. **Sweep offered load** on real coding-agent traffic rescaled to the
   marketplace's own token mix (20,583 in / 2,076 out), until the SLOs stop
   holding. The last level that held is `N*`. Every evaluation needs its own
   sweep: a diff moves `N*`, and pricing it at the baseline's would understate
   every latency win.
2. **Read phase-split GPU time** at `N*` from SGLang's CUDA-event device timer
   (`forward_execution_seconds_total`, labelled `extend` and `decode`).
3. **Divide.** `eff_in = extend / ALL input tokens`, `out = decode / output
   tokens`. No regression. Splitting input into cached and uncached is needed
   only to re-blend at someone else's hit rate, and caching well *is* serving
   well — so we price at our own. **Cache hit rate is an outcome, never a
   control.**
4. **Score both ways.** Effective input price is what OpenRouter sorts on; the
   whole bill is what a buyer pays. On the current baseline those give rank 1
   and rank 9, so the report always carries both.

## The auto-research harness

`src/harness/` is a fleet of agents that propose diffs to SGLang's `srt/` and
price them with the simulator. Four services, each defined by a Protocol in
`harness.contracts` and each replaceable without the others noticing:

| service | question it answers |
|---|---|
| `MemoryService` | has anyone tried this, and what happened? |
| `ContextService` | how exactly did they do it? |
| `EvalService` | the queue in front of the GPUs |
| `AgentService` | one idea, iterated, with divergence and retry policy |
| `OrchestrationService` | N of those at once, kept diverse and in budget |

An agent's whole interface to the code is a `Workspace`:

```python
ws = Workspace("agents/a01")
ws.replace("srt/managers/schedule_policy.py", old, new)   # refuses if ambiguous
ok, why = ws.check()                                      # parses? actually changed?
result = await Simulator(root_dir=ws.run_dir(), stack=ws.stack()).eval()
```

Stock SGLang is fetched from the pinned wheel by URL and cached across the
fleet — 0.5.18 publishes only manylinux wheels and no sdist, so `pip download`
on a Mac cannot get it, and an agent needs the source, not an install.

**On isolation:** the modified SGLang only ever executes inside a fresh Modal
container, so the isolation that matters is already paid for. Agents write text
and wait on an API. The resource they genuinely contend over is **GPU
concurrency and money** — every attempt rents an H100 for 25–60 minutes — so
per-agent directories are enough.

**On keeping agents busy.** That scarcity is a *queue*, not a lock. An agent
submits and keeps working; it blocks only on `collect`, and only after it has
run out of useful things to do:

```python
ticket = evals.submit(EvalRequest(stack=ws.stack(), tier="screen"))
proposer.study(...)          # read other agents' traces, refine the hypothesis
rec = evals.collect(ticket.id)
```

Three mechanisms keep utilisation high, in descending order of what they buy:

1. **Tiers.** A screening run is a fraction of a full sweep and most candidates
   die in it. Capacity is reserved for screens so a backlog of confirmations
   cannot starve the tier that does the filtering.
2. **Dedup by content.** A stack digest is a content hash, so identical
   proposals share one GPU run. Writing the tests surfaced how common this is:
   ten agents seeded from one baseline produced **twenty proposals and one
   run**. Only successes are cached — a failure is a property of the moment,
   and memoising one turns an infra retry into an infinite loop.
3. **Fair ordering.** FIFO within a priority band, so a fast agent cannot barge
   repeatedly and starve a slow one.

`AgentOutcome.idle_s` and `QueueStats.utilisation` report whether it is
working, and tests assert that every `study` call happens with an evaluation
actually in flight.

## Layout

```
src/simulator/          the product
  api.py                Simulator, EvalResult  <- start here
  stack.py              InferenceStack: diffs to sglang, carried by value
  slo.py                bounds that name their own order statistic
  costs.py              $/GPU-hour by provider
  config.py specs.py    serving config and model/hardware specs
  workload/             request loading and sanitisation (TraceLab -> market mix)
  measure/              load generation, latency, server counters, canaries
  price/                GPU-seconds -> price -> market share
  runner/               where a sweep executes (Modal, one entrypoint)
  artifacts/            report and figures written into root_dir
  conftest.py           shared fixtures for every test below
  */tests/              tests live beside the service they test
src/harness/            the auto-research harness
  contracts/            the four service Protocols; no logic lives here
  memory/               SqliteMemory: experiment graph, FTS, briefs
  context/              JsonlContext: append-only agent traces
  agent/                Workspace (the diff API), the iterate-on-one-idea loop
  orchestration/        Fleet: N agents, one gate on GPU spend
monitor/                Modal spend monitoring
docs/methodology.md     how the method was arrived at, and every negative result
docs/example.ipynb      minimal end-to-end notebook
docs/examples/          what a finished run leaves behind
```

## Commands

```
make test                          # 112 unit tests, no GPU, no network
make lint                          # ruff

# tests sit beside the code, and every test directory is a package, so any
# of these resolve from any working directory:
pytest src/simulator/price                 # one service
pytest src/simulator/tests/test_slo.py     # one file
pytest --pyargs simulator                  # an installed copy, no repo needed
make deploy                        # push the runner
make run     ROOT=runs/baseline    # full evaluation
make submit  ROOT=runs/baseline    # start it and walk away
make collect ROOT=runs/baseline
make rescore ROOT=runs/baseline ARGS='--slo ttft:p99:1000,tpot:mean:20'
```

`rescore` is worth knowing about: every level stores its full percentile set
and raw counters, so changing the SLO, the cost basis or the utilisation
assumption costs nothing. Which order statistic the frontier is judged at is a
choice we have changed three times.

## What is assumed, and what is measured

Measured: GPU-seconds per token class, latency percentiles, cache hit rate,
running batch, market prices and volumes.

Assumed, and reported with every result because they cannot be measured from
inside the harness: the **GPU rate** and the **utilisation**. Both are fields on
`Simulator`, and `sim.assumptions()` returns the complete set.

Rates live in `simulator/costs.py`, keyed by provider — `gpu_provider=None`
gives the agreed $3.00/hr H100 default, `gpu_provider="nebius-committed"` gives
$2.50. Serverless retail (Modal's $3.95) is in the catalog but refused as a
serving basis unless you pass `allow_retail_rate=True`: it is what we pay to
run experiments, and pricing a serving business against it would flatter every
competitor by ~30%.

Utilisation is the single largest lever — at 25% every price doubles.
