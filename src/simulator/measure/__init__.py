"""Everything that observes a running server: load, latency, GPU time, quality.

All of it runs inside the GPU container against a local SGLang server, called
by `runner.modal_runner.sweep`; only `equivalence` is driven from the client
side. Entry points:

    server.wait_until_ready / warmup / scrape / diff / wait_idle / BatchSampler
        lifecycle and the Prometheus counters the price is read from
    loadgen.run_concurrent_users(make_session, url, model, n_users, seconds)
        hold N closed-loop users for one level -> list[metrics.RequestResult]
    metrics.summarize(results, slo, warmup_s) / detect_collapse(results)
        percentiles, goodput and the runaway-latency check for one level
    quality.run(url, model, suite, n) -> QualityResult; quality.regressed(...)
        accuracy on pinned GSM8K / LongBench / MMLU slices, before load
    canary.run(url, model)
        six fixed prompts, digested, so a changed sampler is visible
    equivalence.measure(sim) -> dict
        token-level agreement with stock on the workbench (client side)

Reads: dataset files from the Hugging Face Hub (pinned revisions), and JSON
scoring records on the Modal results volume under `/results/equivalence/`.
Writes: `equivalence` scoring records to that same directory, through the
workbench. Nothing else here touches disk.
"""
