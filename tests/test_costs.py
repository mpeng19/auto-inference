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
