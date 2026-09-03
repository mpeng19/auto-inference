"""The traffic a sweep replays: real coding-agent sessions, rescaled to the market.

Entry points, called in this order by `runner.modal_runner.sweep`:

    raw = tracelab.load_sessions(min_rounds=4, max_rounds=40, max_sessions=300)
    scaled, how = tracelab.scale_to_market(raw, MARKET_IN_PER_REQ, MARKET_OUT_PER_REQ)
    pool = tracelab.to_sessions(scaled)       # -> sessions.Session objects
    tracelab.describe(scaled)                 # -> dict for the run record

`sessions.Session` / `sessions.Turn` are the plain records the load generator
consumes; `prompts` supplies the filler text that gives each turn its token
count. `MARKET_IN_PER_REQ` / `MARKET_OUT_PER_REQ` are the marketplace's
per-request token counts, which `api.Simulator` passes to the sweep.

Reads: the TraceLab rounds parquet from the Hugging Face Hub (pinned to
`tracelab.REVISION`), via `huggingface_hub`'s cache. Writes nothing.
"""
