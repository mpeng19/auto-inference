"""Where a sweep actually executes: the Modal app, deployed with `make deploy`.

`modal_runner` defines the app (`SIMULATOR_APP_NAME`, default `auto-inference`)
and its functions, which `api.Simulator` looks up by name and never imports:

    sweep(serving, slo, stack, levels, ...) -> record dict
        launch SGLang with the stack applied, score quality, hold each
        concurrency level, return every percentile and counter
    workbench(stack, script, timeout_s, files) -> dict
        run one script on an H100 against the stack; stdout/stderr tails back
    read_results(paths) -> dict        JSON files off the results volume
    fetch_profile(dir) -> dict         a captured GPU trace, base64
    ls(limit) -> list[str]             stored sweep filenames

Disk, all on Modal volumes: model weights under `/cache/huggingface`; sweep
records to `/results/runs/<stamp>-sweep-<digest>.json`; quality scores cached
under `/results/quality/`; equivalence records under `/results/equivalence/`;
profiles under `/results/profiles/`.
"""
