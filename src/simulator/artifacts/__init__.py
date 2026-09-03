"""Everything a run leaves behind in its `root_dir`, beyond the JSON.

Called by `api.Simulator.write_artifacts` after `analyse`:

    report.render(sim, res) -> str
        the plain-text report: the per-level table, the priced point, both
        ranks on the board, capacity; written to `<root>/report.txt`
    plots.render_all(sim, res, root) -> {name: path}
        one PNG per SLO bound (`slo-<bound>.png`), `price-vs-share.png`,
        `price-vs-demand.png`

Reads nothing. Writes only under the `root` it is given.
"""
from . import plots, report

__all__ = ["plots", "report"]
