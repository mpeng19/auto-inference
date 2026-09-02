"""The quality gate: did we only make it faster?"""
import pytest

from simulator.measure import quality as Q


def test_gsm8k_scoring_prefers_the_marked_answer():
    assert Q.score("gsm8k", "long reasoning\n#### 42", "42")
    assert not Q.score("gsm8k", "long reasoning\n#### 41", "42")


def test_gsm8k_falls_back_to_the_last_number():
    """A degraded model often still reaches an answer while losing the format;
    scoring that as wrong would blame the wrong thing."""
    assert Q.score("gsm8k", "so it costs 1,234 dollars", "1234")
    assert Q.score("gsm8k", "#### 18.0", "18")
    assert not Q.score("gsm8k", "I am not sure", "18")


def test_mmlu_accepts_a_letter_anywhere():
    assert Q.score("mmlu", "B", "B")
    assert Q.score("mmlu", "The answer is C.", "C")
    assert not Q.score("mmlu", "A", "D")


def test_regression_needs_a_baseline_to_compare_against():
    r = Q.QualityResult(suite="gsm8k", n=50, correct=30)
    assert Q.regressed(r) == (False, ""), "no baseline means nothing to claim"


def test_small_movement_is_noise_not_a_regression():
    """Greedy decoding is not bitwise deterministic across batch compositions,
    so a point or two on 100 items is expected."""
    r = Q.QualityResult(suite="gsm8k", n=100, correct=71, baseline_accuracy=0.72)
    assert not Q.regressed(r, tolerance_pp=2.0)[0]


def test_a_real_drop_is_caught():
    r = Q.QualityResult(suite="gsm8k", n=100, correct=60, baseline_accuracy=0.72)
    bad, why = Q.regressed(r, tolerance_pp=2.0)
    assert bad and "not a win" in why


def test_widespread_errors_count_as_a_regression():
    r = Q.QualityResult(suite="gsm8k", n=100, correct=70, errors=20,
                        baseline_accuracy=0.72)
    bad, why = Q.regressed(r)
    assert bad and "errored" in why


def test_nothing_scored_is_a_regression_not_a_pass():
    bad, why = Q.regressed(Q.QualityResult(suite="gsm8k"))
    assert bad and "no items" in why


@pytest.mark.parametrize("suite", ["gsm8k", "mmlu"])
def test_datasets_are_pinned_by_revision(suite):
    """A benchmark that moves underneath you cannot detect a regression."""
    rev = Q.GSM8K_REV if suite == "gsm8k" else Q.MMLU_REV
    assert len(rev) == 40 and rev != "main"


# ── cross-check against SGLang's own benchmark ───────────────────────────

def test_crosscheck_parses_the_metric_table():
    from simulator.measure import crosscheck

    out = crosscheck._parse("""
============ Serving Benchmark Result ============
Request throughput (req/s):              0.61
Output token throughput (tok/s):         287.30
Mean TTFT (ms):                          358.02
Median TTFT (ms):                        296.90
P99 TTFT (ms):                           1017.90
Mean TPOT (ms):                          19.30
Median TPOT (ms):                        19.50
P99 TPOT (ms):                           21.80
""")
    assert out["median_ttft_ms"] == 296.90
    assert out["p99_tpot_ms"] == 21.80
    assert out["output_throughput"] == 287.30


def test_crosscheck_agrees_when_the_clients_agree():
    from simulator.measure import crosscheck

    ours = {"ttft_ms": {"mean": 358.0, "p50": 296.9, "p99": 1017.9},
            "tpot_ms": {"mean": 19.3, "p50": 19.5, "p99": 21.8}}
    theirs = {"mean_ttft_ms": 360.0, "median_ttft_ms": 300.0,
              "p99_ttft_ms": 1000.0, "mean_tpot_ms": 19.5,
              "median_tpot_ms": 19.6, "p99_tpot_ms": 22.0}
    got = crosscheck.compare(ours, theirs)
    assert got["agrees"] and got["worst_deviation"] < 0.05


def test_crosscheck_catches_a_factor_of_two():
    """The failure worth catching is a broken client, not a few percent."""
    from simulator.measure import crosscheck

    ours = {"tpot_ms": {"mean": 40.0}}
    theirs = {"mean_tpot_ms": 20.0}
    got = crosscheck.compare(ours, theirs)
    assert not got["agrees"] and got["fields"]["mean_tpot_ms"]["ratio"] == 2.0


def test_crosscheck_says_so_when_there_is_nothing_to_compare():
    from simulator.measure import crosscheck

    got = crosscheck.compare({}, {})
    assert not got["agrees"] and "no overlapping fields" in got["note"]
