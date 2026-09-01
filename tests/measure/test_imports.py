"""Every product module must import cleanly.

Not busywork. Splitting `bench.py` into `measure/server.py` silently dropped
its module-level `aiohttp`, `os`, `time` and `asyncio` imports, and moving
`rank_vs_market` into `price/market.py` dropped `effective_in`. Both would have
failed at run time -- `wait_until_ready` is the *first* thing a sweep calls,
about six GPU-minutes into a run that costs real money.

Nothing else in the suite exercises the server lifecycle, because doing so
needs a server. This is the cheap guard that covers the gap.
"""
import importlib
import pkgutil

import pytest

import simulator

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(simulator.__path__, "simulator.")
)


def test_walk_found_the_whole_package():
    assert len(MODULES) >= 15, MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_public_api_is_importable_from_the_top():
    for n in simulator.__all__:
        assert hasattr(simulator, n), n
