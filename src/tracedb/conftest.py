"""No tracedb test may open a socket.

The fixture is generated (`synth.py`), never downloaded, and `modal_trace` is
a tool rather than a fixture precisely because it costs money. CI states that
every package's conftest enforces offline at the socket layer; this one makes
that true for tracedb. Same opt-out as the other packages.
"""
import os
import socket

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    if os.environ.get("SIMULATOR_ALLOW_NETWORK"):
        return

    def blocked(*a, **k):
        raise RuntimeError(
            "a tracedb test tried to open a network connection. The suite is "
            "offline by design -- generate the fixture with tracedb.synth, or "
            "set SIMULATOR_ALLOW_NETWORK=1 deliberately.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
