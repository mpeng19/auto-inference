"""Request trace generation and the eval suite.

Three properties matter more than realism:

1. **Open-loop.** Arrival times are drawn *before* the run and the client fires
   at those times whether or not earlier requests finished. A closed-loop
   generator silently converts overload into slowdown, which flatters a bad
   scheduler and hides queueing.

2. **Deterministic.** Same config + seed gives a byte-identical trace, and
   `Trace.digest` goes into the run record. Comparing two configs against two
   different traces is the easiest way to fool yourself.

3. **Variance, not just mean.** Arrival *shape* stresses a serving system as
   much as arrival rate. `bursty` and `sustained` carry the same mean rate on
   purpose, so any difference between them is attributable to burstiness alone.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Callable

from . import prompts as _prompts





@dataclass(frozen=True)
class Turn:
    """One user message inside a conversation."""
    text: str
    max_tokens: int
    category: str = ""


@dataclass(frozen=True)
class Session:
    """A conversation: arrives once, then runs closed-loop turn by turn."""
    idx: int
    arrival_s: float
    system: str                     # stable preamble, shared across sessions
    turns: tuple[Turn, ...]
    think_s: tuple[float, ...]      # gap after each turn before the next

    @property
    def n_turns(self) -> int:
        return len(self.turns)






# A token is ~4 characters of English. Filler text is generated to a target
# token count rather than tokenized for real, so trace building needs no GPU
# and no tokenizer. True token counts come back from the server.
_CHARS_PER_TOKEN = 4
_WORDS = (
    "system latency throughput scheduler batch cache prefix decode prefill "
    "tensor expert routing kernel memory request token attention parallel"
).split()


def _filler(n_tokens: int, rng: random.Random) -> str:
    target = max(1, n_tokens * _CHARS_PER_TOKEN)
    out, size = [], 0
    while size < target:
        w = rng.choice(_WORDS)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)


def _lognormal_int(rng: random.Random, mu: float, sigma: float, cap: int) -> int:
    return max(1, min(cap, int(rng.lognormvariate(mu, sigma))))


# ── arrival processes ────────────────────────────────────────────









# ── the eval suite ───────────────────────────────────────────────
# Each entry stresses a different part of the serving stack. Run the whole
# suite against a config; a change that helps one pattern and wrecks another is
# a trade-off to see explicitly, not to average away.



# The hardware the suite rates were derived from. Rates are meaningless on
# anything else, and getting this wrong is the single most expensive mistake
# available: the first calibration was 10x low and measured an idle server.












# ── multi-turn ───────────────────────────────────────────────────



