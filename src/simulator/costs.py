"""What a GPU-hour costs, by provider.

Separated from everything else because it is the one input that is **purely an
assumption** and the one most likely to change: prices move, providers differ
by 1.8x on the same silicon, and the whole price scales linearly with this
number. It has no business being buried in a pricing function.

Two rules the catalog encodes:

**Serverless retail is not a serving cost basis.** Modal's $3.95/hr is what
*we* pay to run experiments; nobody serving at scale pays it. Those rows are
marked `serving_basis=False` and `rate()` refuses them unless asked explicitly,
because quoting a market price against retail serverless would flatter every
competitor by ~30%.

**With no provider named you get the agreed default**, which for H100 is
$3.00/hr -- between Nebius committed ($2.50) and list on-demand ($3.85),
covering a realistic on-demand/preemptible mix. It is a decision, not an
estimate, and it is reported with every result for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass

# The agreed basis, used when no provider is named. Only for GPUs we have
# actually reasoned about -- inventing a number for unfamiliar silicon would
# be worse than refusing.
DEFAULT_USD_PER_HOUR: dict[str, float] = {
    "H100": 3.00,
    "H200": 3.50,
    "L40S": 1.60,
}
FALLBACK_GPU = "H100"


@dataclass(frozen=True)
class GpuCost:
    gpu: str
    provider: str
    usd_per_hour: float
    note: str = ""
    as_of: str = ""
    # False for serverless retail: real money we spend, but not a basis to
    # price a serving business against.
    serving_basis: bool = True

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.gpu}"


CATALOG: tuple[GpuCost, ...] = (
    # Nebius list prices fetched 2026-08-30; "committed" applies their stated
    # up-to-35% discount for a commitment.
    GpuCost("H100", "nebius", 3.85, "list on-demand", "2026-08-30"),
    GpuCost("H100", "nebius-committed", 2.50, "on-demand less 35%", "2026-08-30"),
    GpuCost("H100", "nebius-preemptible", 2.15, "can be reclaimed", "2026-08-30"),
    GpuCost("H200", "nebius-committed", 2.93, "on-demand less 35%", "2026-08-30"),
    GpuCost("H200", "nebius-preemptible", 2.45, "can be reclaimed", "2026-08-30"),
    # Modal, from the account's own price list. Serverless retail.
    GpuCost("H100", "modal", 3.95, "serverless retail", "2026-08-29", False),
    GpuCost("H200", "modal", 4.54, "serverless retail", "2026-08-29", False),
    GpuCost("B200", "modal", 6.25, "serverless retail", "2026-08-29", False),
    GpuCost("L40S", "modal", 1.95, "serverless retail", "2026-08-29", False),
    GpuCost("A10G", "modal", 1.10, "serverless retail", "2026-08-29", False),
    GpuCost("L4", "modal", 0.80, "serverless retail", "2026-08-29", False),
)

BY_KEY = {c.key: c for c in CATALOG}

# What a Modal container bills besides its GPU. The GPU rate is the headline
# and it is not the bill: the sweep container asks for 16 vCPUs, which at
# retail is more than half a GPU again. Modal pricing page, 2026-08-29.
MODAL_USD_PER_VCPU_HOUR = 0.1368       # $0.000038 / core-second
MODAL_USD_PER_GIB_HOUR = 0.0240        # $0.00000667 / GiB-second


def container_rate(gpu: str, n_gpu: int = 1, vcpu: float = 0.0,
                   memory_gib: float = 0.0, provider: str = "modal") -> float:
    """$/hour for one running container: GPUs plus the CPU and memory it
    reserves. This is what our experiments are billed at, and it is the
    number budgets must be checked against -- charging the GPU rate alone
    understated a night of sweeps by a third."""
    try:
        gpu_rate = rate(gpu, provider, allow_retail=True)
    except KeyError:
        gpu_rate = 3.95
    return (gpu_rate * max(1, n_gpu) + vcpu * MODAL_USD_PER_VCPU_HOUR
            + memory_gib * MODAL_USD_PER_GIB_HOUR)


def rate(gpu: str = FALLBACK_GPU, provider: str | None = None,
         allow_retail: bool = False) -> float:
    """$/GPU-hour for one GPU of `gpu`, from `provider` if named.

        rate()                          -> 3.00   the agreed default
        rate("H100", "nebius-committed") -> 2.50
        rate("L40S", "modal", allow_retail=True) -> 1.95

    With no provider the agreed default for that GPU is returned. Raises for a
    GPU we have no basis for rather than guessing, because this number scales
    the entire answer.
    """
    if provider is None:
        if gpu not in DEFAULT_USD_PER_HOUR:
            raise KeyError(
                f"no agreed default rate for {gpu!r}. Name a provider "
                f"(have {sorted({c.provider for c in CATALOG if c.gpu == gpu})}) "
                f"or pass rate_per_gpu_hour= explicitly.")
        return DEFAULT_USD_PER_HOUR[gpu]
    c = BY_KEY.get(f"{provider}:{gpu}")
    if c is None:
        raise KeyError(f"no price for {provider}:{gpu}; have {sorted(BY_KEY)}")
    if not c.serving_basis and not allow_retail:
        raise ValueError(
            f"{c.key} is serverless retail ({c.note}) at ${c.usd_per_hour}/hr. "
            "That is what we pay to run experiments, not a basis to price a "
            "serving business against -- it would flatter every competitor. "
            "Pass allow_retail=True if you mean it.")
    return c.usd_per_hour


def describe(gpu: str, provider: str | None, usd_per_hour: float) -> str:
    """One line naming the basis, so a result never quotes a bare number."""
    if provider is None:
        return f"${usd_per_hour:.2f}/GPU-hr ({gpu}, agreed default basis)"
    c = BY_KEY.get(f"{provider}:{gpu}")
    tail = f" -- {c.note}" if c and c.note else ""
    return f"${usd_per_hour:.2f}/GPU-hr ({provider} {gpu}{tail})"
