"""GPU utilisation sampling: the denominator of every cost number."""
import subprocess
import time

from autoinf.gpuwatch import GpuWatch, Sample


class _FakeRun:
    """Stand in for nvidia-smi so this runs anywhere."""

    def __init__(self, lines):
        self.lines = lines
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1

        class R:
            stdout = self.lines
        return R()


def test_summary_integrates_utilisation(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeRun("50, 1024\n"))
    w = GpuWatch(interval_s=0.01).start()
    time.sleep(0.15)
    w.stop()
    s = w.summary(wall_s=100.0, n_gpu=1)
    assert s["available"]
    assert s["busy_frac"] == 0.5
    # Half-busy for 100 wall-seconds is 50 GPU-seconds of work.
    assert s["gpu_seconds"] == 50.0
    assert s["wall_gpu_seconds"] == 100.0


def test_averages_across_gpus(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeRun("100, 2048\n0, 512\n"))
    w = GpuWatch(interval_s=0.01).start()
    time.sleep(0.1)
    w.stop()
    s = w.summary(wall_s=10.0, n_gpu=2)
    assert s["busy_frac"] == 0.5              # mean of 100% and 0%
    assert s["gpu_seconds"] == 10.0           # 10s * 0.5 * 2 GPUs
    assert s["mem_used_max_mib"] == 2560.0    # summed across GPUs


def test_idle_gpu_is_not_charged(monkeypatch):
    """The whole point: an idle GPU must not bill its time to a few tokens."""
    monkeypatch.setattr(subprocess, "run", _FakeRun("2, 900\n"))
    w = GpuWatch(interval_s=0.01).start()
    time.sleep(0.1)
    w.stop()
    s = w.summary(wall_s=90.0, n_gpu=1)
    assert s["gpu_seconds"] < 5.0             # ~1.8s of real work
    assert s["wall_gpu_seconds"] == 90.0      # vs 90s of wall time


def test_reports_unavailable_rather_than_guessing(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", boom)
    w = GpuWatch(interval_s=0.01).start()
    time.sleep(0.05)
    w.stop()
    s = w.summary(wall_s=10.0)
    assert not s["available"]
    assert s["gpu_seconds"] is None           # never a fabricated number
    assert "nvidia-smi" in (s["error"] or "")


def test_context_manager(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeRun("80, 100\n"))
    with GpuWatch(interval_s=0.01) as w:
        time.sleep(0.05)
    assert w.summary(wall_s=1.0)["busy_frac"] == 0.8
