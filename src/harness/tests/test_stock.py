"""Where stock SGLang comes from.

One test here actually touches the network, because the thing worth verifying
is that a real wheel yields real source. It is skipped unless
`SIMULATOR_ALLOW_NETWORK=1`, so the default suite stays offline.
"""
import os

import pytest

from harness.agent.stock import SGLANG_VERSION, InstalledSglang, WheelSource, stock

live = pytest.mark.skipif(not os.environ.get("SIMULATOR_ALLOW_NETWORK"),
                          reason="needs the network; set SIMULATOR_ALLOW_NETWORK=1")


def test_prefers_an_installed_package_when_present():
    s = stock()
    expected = InstalledSglang if InstalledSglang.available() else WheelSource
    assert isinstance(s, expected)


def test_cache_is_shared_across_the_fleet():
    """Ten agents downloading the same 24 MB wheel is ten times the wait."""
    a, b = WheelSource(), WheelSource()
    assert a.root == b.root
    assert ".cache" in str(a.root)


@live
def test_a_real_wheel_yields_real_sources():
    """SGLang 0.5.18 ships manylinux wheels and no sdist, so `pip download` on
    a Mac refuses every file. Fetching by URL is what makes this work at all."""
    s = WheelSource(version=SGLANG_VERSION)
    files = s.ls("srt")
    assert len(files) > 500
    text = s.read("srt/managers/schedule_policy.py")
    assert "class SchedulePolicy" in text
    # the hash the vendored copy recorded before this source existed
    assert s.sha("srt/managers/schedule_policy.py") == "7fa154986e574cab"
