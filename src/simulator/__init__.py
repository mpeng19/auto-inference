"""Price an LLM serving stack the way a marketplace would.

Give it an inference stack -- stock SGLang, or stock plus a set of diffs to
`srt/` -- and it answers one question: **what effective price can this stack
serve marketplace traffic at, and how much of that market can it hold?**

    from simulator import Simulator, InferenceStack

    sim = Simulator(root_dir="runs/my-diff",
                    stack=InferenceStack.from_dir("my-diff/"))
    result = await sim.eval()
    print(result.summary())

The method, in four steps:

  1. Sweep offered load on real coding-agent traffic, rescaled to the
     marketplace's own token mix, until the SLOs stop holding. The last level
     that held is N*. **Every evaluation needs its own sweep** -- a diff moves
     N*, and pricing it at the baseline's N* understates every latency win.
  2. Read phase-split GPU time at N* from SGLang's CUDA-event device timer:
     `forward_execution_seconds_total` labelled `extend` and `decode`.
  3. Divide. `eff_in = extend / ALL input tokens`, `out = decode / output
     tokens`. No regression: splitting input into cached and uncached is
     needed only to re-blend at someone else's hit rate, and caching well
     *is* serving well, so we price at our own.
  4. Multiply by the rate, divide by utilisation, and score both against the
     board -- on effective input price, which is what OpenRouter sorts on, and
     on the whole bill, which is what a buyer pays. They disagree.

Cache hit rate is an outcome, never a control.
"""
from .api import EvalResult, Point, Simulator
from .price.market import Economics, Market
from .slo import MARKET_SLO, SLO, Bound
from .stack import InferenceStack

__all__ = [
           "MARKET_SLO",
           "SLO",
           "Bound",
           "Economics",
           "EvalResult",
           "InferenceStack",
           "Market",
           "Point",
           "Simulator",
]
__version__ = "0.1.0"
