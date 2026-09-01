import pathlib

import pytest

from harness.context import JsonlContext
from harness.memory import SqliteMemory


@pytest.fixture
def memory(tmp_path) -> SqliteMemory:
    return SqliteMemory(tmp_path / "memory.db")


@pytest.fixture
def context(tmp_path) -> JsonlContext:
    return JsonlContext(tmp_path / "traces")


@pytest.fixture
def stock_dir(tmp_path) -> pathlib.Path:
    """A fake stock SGLang tree, so tests never touch the network.

    The real `WheelSource` is exercised once, in its own test, behind a
    network-availability guard; everything else uses this.
    """
    root = tmp_path / "stock" / "sglang"
    (root / "srt" / "managers").mkdir(parents=True)
    (root / "srt" / "mem_cache").mkdir(parents=True)
    (root / "srt" / "managers" / "schedule_policy.py").write_text(
        "CHUNK = 8192\n\n\nclass SchedulePolicy:\n    pass\n")
    (root / "srt" / "mem_cache" / "radix_cache.py").write_text(
        "class RadixCache:\n    evict = 'lru'\n")
    return root


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No harness test may open a socket.

    Deliberately duplicated from `simulator/conftest.py` rather than shared:
    a conftest importing another package's conftest is a fragile dependency,
    and twenty lines is cheaper than that coupling. The reason is the same --
    a test that quietly starts calling Modal or PyPI would be slow, flaky, and
    would only be noticed on a bill.

    `SIMULATOR_ALLOW_NETWORK=1` opts out, which is how the one real
    `WheelSource` test runs.
    """
    import os
    import socket

    if os.environ.get("SIMULATOR_ALLOW_NETWORK"):
        return

    def blocked(*a, **k):
        raise RuntimeError(
            "a harness test tried to open a network connection. The suite is "
            "offline by design -- use the `stock_dir` fixture, a fake "
            "evaluator, or set SIMULATOR_ALLOW_NETWORK=1 deliberately.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
