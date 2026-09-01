import json
import pathlib

import pytest

# Anchored to this file, so `pytest` works from the repo root, from `src/`,
# from a service directory, or against a single test path.
DATA = pathlib.Path(__file__).parent / "tests" / "data"


@pytest.fixture
def sweep() -> dict:
    """The real 1xH100 baseline sweep (run 1788287578), trimmed to what the
    product reads. Every number the product claims is reproduced from this."""
    return json.loads((DATA / "sweep-1xh100.json").read_text())


@pytest.fixture
def root(tmp_path) -> pathlib.Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No test may open a socket.

    The whole suite is offline by construction: it runs on stored records and
    synthetic rows. A test that quietly starts calling Modal would be slow,
    flaky, and cost real money on every CI run -- and would only be noticed
    once someone looked at a bill. Failing loudly at the socket layer is
    cheaper than trusting everyone to remember.

    `SIMULATOR_ALLOW_NETWORK=1` opts a specific run out, for the rare occasion
    you genuinely want to exercise the live path locally.
    """
    import os
    import socket

    if os.environ.get("SIMULATOR_ALLOW_NETWORK"):
        return

    def blocked(*a, **k):
        raise RuntimeError(
            "a test tried to open a network connection. The suite is offline "
            "by design -- use a stored record or a fixture. Set "
            "SIMULATOR_ALLOW_NETWORK=1 to override deliberately.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
