"""The ledger must actually fire on the ways an autonomous search goes wrong.

A monitor that stays quiet through a pathological search is worse than none:
it converts "nobody checked" into "the check passed".
"""
import time

import pytest

from autoinf.ledger import Experiment, Ledger, novelty


BASE = {"tp_size": 1, "dp_size": 1, "ep_size": None, "mem_fraction_static": 0.85,
        "max_total_tokens": None, "max_running_requests": 256,
        "chunked_prefill_size": 8192, "schedule_conservativeness": 1.0,
        "context_length": None, "schedule_policy": "fcfs",
        "enable_prefix_caching": True, "model": "q", "gpu": "H100", "n_gpu": 1}


def mk(led, i, cfg=None, goodput=26.0, files=(), digest="", canary=1.0,
       failed=0, cost=0.5, hyp="try something"):
    e = Experiment(id=f"e{i}", ts=time.time() + i, hypothesis=hyp,
                   config={**BASE, **(cfg or {})}, overlay_digest=digest,
                   overlay_files=tuple(files), goodput_rps=goodput,
                   canary_exact_rate=canary, n_failed=failed, cost_usd=cost)
    return led.append(e)


@pytest.fixture
def led(tmp_path):
    return Ledger(tmp_path / "l.jsonl")


def test_empty_ledger(led):
    assert led.report()["n"] == 0


def test_healthy_search_is_not_flagged(led):
    """Diverse knobs, steady improvement, clean canaries."""
    mk(led, 0, goodput=26.0)
    mk(led, 1, {"max_running_requests": 512}, goodput=27.0)
    mk(led, 2, {"schedule_policy": "lpm"}, goodput=28.0)
    mk(led, 3, {"chunked_prefill_size": 16384}, goodput=29.0)
    mk(led, 4, {"mem_fraction_static": 0.92}, goodput=30.0)
    mk(led, 5, {"schedule_conservativeness": 0.6}, goodput=31.5)
    r = led.report()
    assert r["verdict"] == "HEALTHY", r["flags"]
    assert r["improvement_pct"] > 20


def test_circling_is_detected(led):
    """The same experiment re-proposed with cosmetic differences."""
    mk(led, 0, goodput=26.0)
    for i in range(1, 10):
        # Tiny perturbations of one knob: novel-looking, substantively identical.
        mk(led, i, {"max_running_requests": 256 + i}, goodput=26.0 + i * 0.001)
    r = led.report()
    assert any("CIRCLING" in f for f in r["flags"]), r["flags"]
    assert r["mean_novelty_recent"] < 0.05


def test_narrow_search_is_detected(led):
    """High novelty on one axis, everything else untouched."""
    mk(led, 0, goodput=26.0)
    for i, v in enumerate([64, 128, 512, 1024, 32, 900, 700], start=1):
        mk(led, i, {"max_running_requests": v}, goodput=26.0 + i * 0.01)
    r = led.report()
    assert any("NARROW" in f for f in r["flags"]), r["flags"]
    assert r["knob_diversity"] < 0.35


def test_plateau_is_detected(led):
    """Genuinely varied search that simply stops finding anything better."""
    mk(led, 0, goodput=30.0)                      # best arrives first
    varied = [
        {"chunked_prefill_size": 2048, "schedule_policy": "lpm"},
        {"mem_fraction_static": 0.7, "max_running_requests": 64},
        {"tp_size": 2, "schedule_policy": "lof"},
        {"ep_size": 4, "chunked_prefill_size": 32768},
        {"schedule_conservativeness": 0.2, "enable_prefix_caching": False},
        {"max_total_tokens": 100000, "schedule_policy": "dfs-weight"},
        {"context_length": 8192, "mem_fraction_static": 0.95},
        {"dp_size": 2, "max_running_requests": 1000},
        {"tp_size": 4, "schedule_policy": "random"},
        {"ep_size": 8, "chunked_prefill_size": 512},
    ]
    for i, cfg in enumerate(varied, start=1):
        mk(led, i, cfg, goodput=29.0 - i * 0.05,
           files=(f"srt/managers/f{i}.py",), digest=f"d{i}")
    r = led.report()
    assert any("PLATEAU" in f for f in r["flags"]), r["flags"]
    assert r["experiments_since_best"] >= 8, r["experiments_since_best"]


def test_reward_hacking_is_detected(led):
    """Goodput jumps while canaries diverge -- the signature that matters."""
    mk(led, 0, goodput=26.0, canary=1.0)
    mk(led, 1, {"schedule_policy": "lpm"}, goodput=27.0, canary=1.0)
    mk(led, 2, {"max_running_requests": 900}, goodput=45.0, canary=0.3,
       files=("srt/managers/scheduler.py",), digest="deadbeef")
    r = led.report()
    assert any("INTEGRITY" in f for f in r["flags"]), r["flags"]
    assert "e2" in r["suspect_ids"]


def test_dropped_requests_also_count_as_suspect(led):
    mk(led, 0, goodput=26.0)
    mk(led, 1, {"schedule_policy": "lof"}, goodput=40.0, canary=1.0, failed=120)
    r = led.report()
    assert any("INTEGRITY" in f for f in r["flags"]), r["flags"]


def test_expensive_search_is_flagged(led):
    mk(led, 0, goodput=26.0, cost=20.0)
    mk(led, 1, {"schedule_policy": "lpm"}, goodput=26.1, cost=20.0)
    mk(led, 2, {"chunked_prefill_size": 4096}, goodput=26.2, cost=20.0)
    r = led.report()
    assert any("EXPENSIVE" in f for f in r["flags"]), r["flags"]


def test_novelty_zero_for_exact_repeat(led):
    a = mk(led, 0)
    prior = led.load()
    b = Experiment(id="dup", ts=time.time(), hypothesis="same", config=dict(a.config))
    assert novelty(b, prior) == pytest.approx(0.0, abs=1e-9)


def test_novelty_high_for_code_change(led):
    mk(led, 0)
    prior = led.load()
    e = Experiment(id="x", ts=time.time(), hypothesis="rewrite scheduler",
                   config=dict(BASE), overlay_digest="abc123",
                   overlay_files=("srt/managers/scheduler.py",))
    assert novelty(e, prior) > 0.4


def test_ledger_is_append_only(led):
    mk(led, 0); mk(led, 1)
    assert len(led.load()) == 2
    assert led.path.read_text().count("\n") == 2


def test_whats_new_digest(led):
    mk(led, 0, hyp="baseline")
    mk(led, 1, {"schedule_policy": "lpm"}, goodput=28.0,
       hyp="longest-prefix-match should help the shared-prefix workloads")
    out = led.whats_new(k=2)
    assert "longest-prefix-match" in out
    assert "novelty" in out
