# The product is `simulator`. Everything below is a thin wrapper over it.
.PHONY: test lint run submit collect rescore ls spend research-test help

test:            ## unit tests, no GPU, no network
	uv run pytest -q

run:             ## full evaluation: submit, wait, write artifacts (25-60 GPU-min)
	uv run simulate run --root $(ROOT) --mkdir $(ARGS)

submit:          ## start a sweep and return; it outlives this terminal
	uv run simulate submit --root $(ROOT) --mkdir $(ARGS)

collect:         ## pick up a submitted sweep and write its artifacts
	uv run simulate collect --root $(ROOT) $(ARGS)

rescore:         ## re-judge and re-price a stored sweep, no GPU
	uv run simulate rescore --root $(ROOT) $(ARGS)

ls:              ## list sweeps on the results volume
	uv run simulate ls

deploy:          ## push the runner so `submit` can find it
	uv run modal deploy src/simulator/runner/modal_runner.py

spend:           ## current Modal spend
	uv run python ops/spend_monitor.py

market:          ## refresh data/market-*.json from OpenRouter
	uv run python -m simulator.price.market_pull

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t18
