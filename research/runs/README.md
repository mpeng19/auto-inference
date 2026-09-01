# runs/

One directory per orchestration run. Layout is designed for ten agents
iterating, which produces gigabytes: bulk per-request rows are stored
separately from summaries so they can be pruned without losing the record.

    runs/<run_id>/
      ledger.jsonl              append-only experiment record
      score.json                objective definition in force for this run
      agents/<agent_id>/
        trace.jsonl             ContextManagerService: full agent trace
        experiments/<exp_id>/
          config.json           ServingConfig
          overlay.diff          -> artifacts/<sha256>/ (content addressed)
          result.json           metrics only, small, keep
          per_request.parquet   bulk rows, ~13MB/suite run, prunable

`artifacts/<sha256>/` is content-addressed so ten agents producing
near-identical patches store one copy — and the same digest keys the
ExperimentQueue's result cache.
