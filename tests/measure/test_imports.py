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


def test_importing_the_runner_makes_no_network_call():
    """It runs at decorator time on every import.

    An existence check on a Modal secret used to sit here; under a blocked
    socket it retried for 57 seconds, and it would have failed a fresh clone's
    first deploy on a secret the user has no reason to own. The offline fixture
    turns any regression into a hard failure, so this only has to be fast.
    """
    import importlib
    import time

    import simulator.runner.modal_runner as r

    t0 = time.perf_counter()
    importlib.reload(r)
    assert time.perf_counter() - t0 < 5.0
    assert r._hf_secret() == [] or r.HF_SECRET
