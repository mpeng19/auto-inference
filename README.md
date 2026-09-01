# auto-inference

Price an LLM serving stack the way a marketplace would.

Give it an **inference stack** — stock SGLang, or stock plus a set of diffs to
`srt/` — and it answers one question: *what effective price can this stack
serve marketplace traffic at, and how much of that market can it hold?*

```python
from simulator import Simulator, InferenceStack

sim = Simulator(root_dir="runs/my-diff",
                stack=InferenceStack.from_dir("overlays"))
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
and two figures — where the SLO stops us, and what that costs against market
share.

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

## Layout

```
src/simulator/          the product
  api.py                Simulator, EvalResult  <- start here
  stack.py              InferenceStack: diffs to sglang, carried by value
  slo.py                bounds that name their own order statistic
  config.py specs.py    serving config and model/hardware specs
  workload/             request loading and sanitisation (TraceLab -> market mix)
  measure/              load generation, latency, server counters, canaries
  price/                GPU-seconds -> price -> market share
  runner/               where a sweep executes (Modal, one entrypoint)
  artifacts/            report and figures written into root_dir
research/               how we got here: retired code, negative results, docs
ops/                    spend monitoring
docs/HANDOFF.md         the running log; read §6b-§6d for the settled method
```

## Commands

```
make test                          # 112 unit tests, no GPU, no network
make lint                          # ruff
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
