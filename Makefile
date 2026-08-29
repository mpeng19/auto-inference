# Common operations. Everything runs on one H100 unless stated.
#
# GPU runs use `modal run --detach`. Without it, `modal run` creates an
# *ephemeral* app tied to the local client: kill the terminal and Modal tears
# the run down mid-flight. That cost us a ~$4 run that produced no result.
# Detached runs survive client death and still persist to the results Volume.
.PHONY: test suite suite30 staircase noise realism results capacity probe fmt

test:            ## local tests, no GPU
	uv run pytest -q

capacity:        ## regenerate docs/capacity.md from the roofline model
	uv run python scripts/capacity_report.py

suite:           ## eval suite, ~10 min of traces (+4 min model load)
	uv run modal run --detach src/autoinf/modal_app.py::suite

suite30:         ## eval suite, 30 min of traces -- the consistency run
	uv run modal run --detach src/autoinf/modal_app.py::suite --minutes 30

staircase:       ## step to 100% of roofline, stop when SLO collapses
	uv run modal run --detach src/autoinf/modal_app.py::staircase

noise:           ## same config N times: the noise floor
	uv run modal run --detach src/autoinf/modal_app.py::noise --repeats 5

realism:         ## LLM-driven virtual users (multi-turn, abandonment)
	uv run modal run --detach src/autoinf/modal_app.py::realism

results:         ## list stored runs
	uv run modal run scripts/results.py::ls

ledger:          ## research-loop health: novelty, diversity, progress
	uv run python scripts/ledger_report.py

spend:           ## current Modal spend
	uv run python scripts/spend_monitor.py

help:
	@grep -E '^[a-z0-9]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t18
