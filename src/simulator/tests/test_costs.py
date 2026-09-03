"""GPU rates: the one input that is purely an assumption, kept in one place."""
import pytest

from simulator.costs import CATALOG, DEFAULT_USD_PER_HOUR, describe, rate


def test_no_provider_gives_the_agreed_default():
    assert rate() == 3.00
    assert rate("H100") == 3.00


def test_a_named_provider_gives_its_own_quote():
    assert rate("H100", "nebius-committed") == 2.50
    assert rate("H100", "nebius-preemptible") == 2.15


def test_serverless_retail_is_not_a_serving_basis():
    """$3.95/hr is what we pay to run experiments. Pricing a serving business
    against it would flatter every competitor by ~30%."""
    with pytest.raises(ValueError, match="retail"):
        rate("H100", "modal")
    assert rate("H100", "modal", allow_retail=True) == 3.95


def test_unknown_gpu_refuses_rather_than_guessing():
    """This number scales the entire answer, so inventing one is worse than
    failing."""
    with pytest.raises(KeyError):
        rate("B200")
    with pytest.raises(KeyError):
        rate("H100", "no-such-provider")


def test_the_default_sits_inside_the_real_range():
    quotes = [c.usd_per_hour for c in CATALOG if c.gpu == "H100" and c.serving_basis]
    assert min(quotes) < DEFAULT_USD_PER_HOUR["H100"] < max(quotes)


def test_every_row_is_dated_and_explained():
    for c in CATALOG:
        assert c.as_of and c.note, c


def test_describe_never_quotes_a_bare_number():
    assert "agreed default" in describe("H100", None, 3.00)
    assert "nebius-committed" in describe("H100", "nebius-committed", 2.50)


def test_a_container_costs_more_than_its_gpu():
    """Charging the GPU rate alone understated a night of sweeps by a third:
    the sweep container reserves 16 vCPUs, which at retail is more than half a
    GPU again."""
    from simulator.costs import MODAL_USD_PER_VCPU_HOUR, container_rate, rate

    gpu = rate("H100", "modal", allow_retail=True)
    got = container_rate("H100", 1, vcpu=16.0)
    assert got == pytest.approx(gpu + 16.0 * MODAL_USD_PER_VCPU_HOUR)
    assert got > gpu * 1.5


def test_container_rate_scales_with_gpu_count_and_memory():
    from simulator.costs import MODAL_USD_PER_GIB_HOUR, container_rate

    a = container_rate("H100", 1)
    b = container_rate("H100", 4)
    assert b == pytest.approx(4 * a)
    assert container_rate("H100", 1, memory_gib=32.0) == pytest.approx(
        a + 32.0 * MODAL_USD_PER_GIB_HOUR)


def test_an_unpriced_gpu_still_bills_rather_than_stopping_the_fleet():
    """`rate()` refuses to guess; `container_rate` deliberately does not, because
    it only ever bills our own experiments and a budget check that raised over a
    missing catalog row would stop a run for no good reason."""
    from simulator.costs import container_rate, rate

    assert container_rate("GB200", 1) == pytest.approx(
        rate("H100", "modal", allow_retail=True))
