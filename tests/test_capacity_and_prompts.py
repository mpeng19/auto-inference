"""Tests for the capacity model, human prompts, and the staircase arrival."""
import math
import random

from autoinf.config import WorkloadConfig
from autoinf.flops import (H100, QWEN3_30B_A3B, capacity, request_cost)
from autoinf.prompts import (ALL_CATEGORIES, CATEGORIES, LONG_FORM,
                             make_request, sample_category)
from autoinf.workload import build_trace, roofline_rps, staircase, suite


# ── capacity model ───────────────────────────────────────────────

def test_param_counts_match_the_model_card():
    m = QWEN3_30B_A3B
    assert abs(m.total_params / 1e9 - 30.5) < 0.3, m.total_params / 1e9
    assert 2.9 < m.active_params / 1e9 < 3.4, m.active_params / 1e9


def test_moe_touches_nearly_all_experts_at_batch():
    """The key MoE fact: decode reads total params, not active params."""
    m = QWEN3_30B_A3B
    assert m.experts_touched(1) == 8
    assert m.experts_touched(64) > 120
    assert m.experts_touched(256) > 127.9


def test_decode_is_bandwidth_bound_and_prefill_is_not():
    m, hw = QWEN3_30B_A3B, H100
    short_prompt = capacity(m, hw, 50, 800, batch=146)     # decode-dominated
    long_prompt = capacity(m, hw, 8000, 20, batch=146)     # prefill-dominated
    assert short_prompt["bound_by"] == "decode/bandwidth"
    assert long_prompt["bound_by"] == "prefill/compute"


def test_roofline_brackets_the_measured_throughput():
    """Measured 34.3 rps must sit below the ceiling but not absurdly so.

    A model that predicted less than measurement would be wrong; one predicting
    100x would be useless. 40-90% is the band where the ratio is informative.
    """
    c = capacity(QWEN3_30B_A3B, H100, 537, 233, batch=146)
    frac = 34.3 / c["max_rps_roofline"]
    assert 0.40 < frac < 0.90, (c["max_rps_roofline"], frac)


def test_request_cost_scales_with_tokens():
    m = QWEN3_30B_A3B
    a = request_cost(m, 500, 100)["total_tflops"]
    b = request_cost(m, 1000, 100)["total_tflops"]
    assert b > a
    # Attention is quadratic in prompt length, so doubling input more than doubles
    # prefill cost -- but dense FLOPs dominate here, so stay under 2.5x.
    assert 1.5 < (b / a) < 2.5, b / a


# ── prompts ──────────────────────────────────────────────────────

def test_prompts_are_deterministic():
    a = make_request(random.Random(5), "chat", 40)
    b = make_request(random.Random(5), "chat", 40)
    assert a == b


def test_prompts_look_like_human_requests():
    rng = random.Random(11)
    for name in ALL_CATEGORIES:
        p = make_request(rng, name, 200)
        assert len(p) > 20, name
        # Must not leak unfilled template slots. Code categories legitimately
        # contain braces, so check only the prose ones.
        if name != "code_debug":
            assert "{" not in p and "}" not in p, (name, p[:120])
        assert p[0].isupper() or p.startswith(("<", "```")), (name, p[:60])


def test_code_language_matches_the_code_block():
    """A question about TypeScript must not contain SQL."""
    rng = random.Random(2)
    for _ in range(25):
        p = make_request(rng, "code_debug", 300)
        lang = p.split(" code is giving")[0].replace("This ", "")
        assert f"```{lang.lower()}" in p, (lang, p[:100])


def test_long_form_hits_target_length_short_form_stays_natural():
    rng = random.Random(4)
    for name in LONG_FORM:
        p = make_request(rng, name, 800)
        est = len(p) / 4
        assert 0.6 * 800 < est < 1.4 * 800, (name, est)
    # A chat question stays short no matter what target is requested.
    p = make_request(rng, "chat", 2000)
    assert len(p) / 4 < 120, len(p) / 4


def test_category_weights_are_respected():
    rng = random.Random(9)
    n = 20000
    counts = {}
    for _ in range(n):
        c = sample_category(rng, ALL_CATEGORIES)
        counts[c.name] = counts.get(c.name, 0) + 1
    total_w = sum(c.weight for c in CATEGORIES)
    for c in CATEGORIES:
        observed = counts.get(c.name, 0) / n
        assert abs(observed - c.weight / total_w) < 0.02, (c.name, observed, c.weight)


def test_human_workload_has_varied_categories_and_lengths():
    tr = build_trace(suite(seed=3)["human"])
    cats = {r.category for r in tr.requests}
    assert len(cats) >= 8, cats
    ins = [r.prompt_tokens_est for r in tr.requests]
    # Realistic traffic spans orders of magnitude, not one narrow band.
    assert max(ins) / max(1, min(ins)) > 20, (min(ins), max(ins))


def test_shared_prefixes_are_real_system_prompts():
    tr = build_trace(suite(seed=3)["human"])
    shared = [r for r in tr.requests if r.prefix_id is not None]
    assert shared
    assert any("assistant" in r.prompt[:400].lower()
               or "support" in r.prompt[:400].lower()
               or "reviewer" in r.prompt[:400].lower() for r in shared)


# ── staircase ────────────────────────────────────────────────────

def test_staircase_levels_and_duration():
    c = WorkloadConfig(arrival="staircase", request_rate=40.0, stair_step_s=60.0,
                       n_requests=None, duration_s=None)
    lv = c.stair_levels()
    assert lv[0] == 5.0 and lv[-1] == 100.0
    assert len(lv) == 20
    assert c.stair_duration() == 1200.0


def test_staircase_holds_each_plateau_at_its_target_rate():
    peak, step = 40.0, 60.0
    c = WorkloadConfig(arrival="staircase", request_rate=peak, stair_step_s=step,
                       n_requests=None, duration_s=None, seed=5)
    tr = build_trace(c)
    for i, pct in enumerate(c.stair_levels()):
        lo, hi = i * step, (i + 1) * step
        obs = sum(1 for r in tr.requests if lo <= r.arrival_s < hi) / step
        target = peak * pct / 100.0
        assert abs(obs - target) / target < 0.25, (pct, target, obs)


def test_staircase_is_monotonically_increasing():
    c = WorkloadConfig(arrival="staircase", request_rate=40.0, stair_step_s=60.0,
                       n_requests=None, duration_s=None, seed=6)
    tr = build_trace(c)
    per = [sum(1 for r in tr.requests if i * 60 <= r.arrival_s < (i + 1) * 60)
           for i in range(len(c.stair_levels()))]
    # Allow Poisson jitter between adjacent plateaus, but the trend must climb.
    assert per[-1] > per[0] * 10, (per[0], per[-1])
    # Levels 55-100% sum to 775 against 275 for levels 5-50%, so the expected
    # ratio is ~2.8, not 3.
    assert sum(per[10:]) > 2.5 * sum(per[:10]), (sum(per[:10]), sum(per[10:]))


def test_staircase_is_sized_to_the_roofline():
    st = staircase()
    assert abs(st.request_rate - roofline_rps()) < 0.5
    half = staircase(peak_fraction=0.5)
    assert abs(half.request_rate - roofline_rps() * 0.5) < 0.5
