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


def test_longbench_is_scored_by_token_f1_against_any_gold():
    from simulator.measure.quality import qa_f1, score

    assert qa_f1("Miller v. California", ["Miller v. California"]) == 1.0
    assert qa_f1("The Miller v California case.", ["Miller v. California"]) > 0.8
    assert 0 < qa_f1("Miller", ["Miller v. California"]) < 1
    assert qa_f1("no idea", ["Miller v. California"]) == 0.0
    assert qa_f1("1990", ["1989", "1990"]) == 1.0
    assert score("longbench", "some reasoning first\nParis", "Paris\x1fparis, france") == 1.0
    assert score("gsm8k", "#### 42", "42") == 1.0 and score("mmlu", "B", "B") == 1.0
    assert score("mmlu", "Option A looks wrong; C fits.\n#### C", "C") == 1.0
    assert score("mmlu", "Option A looks wrong; C fits.\n#### C", "A") == 0.0


def test_longbench_slice_is_pinned_and_long(tmp_path, monkeypatch):
    """A gate must exercise the path it guards: the first build-mode 'win'
    only engaged above 4,096 tokens and GSM8K never got there."""
    import io
    import json
    import zipfile

    from simulator.measure import quality

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in quality.LONGBENCH_SETS:
            rows = [{"input": f"q{i}", "context": "word " * 40000, "answers": f"['a{i}']"}
                    for i in range(5)]
            z.writestr(f"data/{name}.jsonl", "\n".join(json.dumps(r) for r in rows))
    p = tmp_path / "data.zip"
    p.write_bytes(buf.getvalue())
    monkeypatch.setattr(quality, "hf_hub_download", lambda *a, **k: str(p), raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(p))
    items = quality.load("longbench", n=4)
    assert len(items) == 4 and items[0].max_tokens == 256
    assert len(items[0].prompt) <= quality.LONGBENCH_MAX_CHARS + 600
    assert "Question:" in items[0].prompt and items[0].answer.startswith("a")
    assert quality.LONGBENCH_REV and len(quality.LONGBENCH_REV) == 40


def test_canary_localises_where_two_runs_part():
    """`first_divergence` is the point: one token nudged late in a long
    generation and a response derailed from the start are different bugs."""
    from simulator.measure import canary

    got = canary.compare({"a": "the answer is 408", "b": "hello"},
                         {"a": "the answer is 407", "b": "hello"})
    assert got["n"] == 2 and got["n_identical"] == 1
    assert got["exact_match_rate"] == 0.5
    assert got["per_canary"]["b"]["status"] == "identical"
    d = got["per_canary"]["a"]
    assert d["status"] == "diverged" and d["first_divergence"] == 16
    assert d["frac_identical"] > 0.9


def test_canary_reports_a_missing_result_rather_than_scoring_it():
    from simulator.measure import canary

    got = canary.compare({"a": "x"}, {})
    assert got["per_canary"]["a"]["status"] == "missing"
    assert got["n_identical"] == 0


def test_canary_verdict_refuses_to_judge_without_a_measured_floor():
    """Greedy decoding is not bitwise deterministic across batch compositions,
    so divergence with no baseline is not evidence of anything."""
    from simulator.measure import canary

    assert "no baseline" in canary.verdict({"exact_match_rate": 0.5}, None)


def test_canary_verdict_is_judged_against_the_same_config_floor():
    from simulator.measure import canary

    floor = {"exact_match_rate": 0.6}
    assert canary.verdict({"exact_match_rate": 0.8}, floor).startswith("OK")
    assert canary.verdict({"exact_match_rate": 0.6}, floor).startswith("OK")
    assert canary.verdict({"exact_match_rate": 0.5}, floor).startswith("MARGINAL")
    assert canary.verdict({"exact_match_rate": 0.1}, floor).startswith("SUSPECT")


def test_longbench_answer_is_read_after_the_marker_or_from_the_last_line():
    from simulator.measure.quality import score, suite_digest

    reasoning = "We need the answer. The passages say the film was an erotic thriller.\n#### erotic thriller film"
    assert score("longbench", reasoning, "erotic thriller film") == 1.0
    assert score("longbench", "Thinking...\nParis", "Paris") == 1.0
    assert score("longbench", "Thinking...\nParis", "London") == 0.0
    assert suite_digest("longbench") != suite_digest("gsm8k")
    assert len(suite_digest("mmlu")) == 10


def test_a_level_starts_clean_or_says_so(monkeypatch):
    """Cancelled replies from the previous level must not run into the
    next one: the first deadline-cut baseline had a running batch of 10
    with 8 users."""
    import asyncio

    from simulator.measure import server

    seen = iter([{"sglang:num_running_reqs": 3, "sglang:num_queue_reqs": 1},
                 {"sglang:num_running_reqs": 1, "sglang:num_queue_reqs": 0},
                 {"sglang:num_running_reqs": 0, "sglang:num_queue_reqs": 0}])

    class Snap:
        def __init__(self, g):
            self.gauges = g

    async def fake_scrape(url, timeout_s=10.0):
        return Snap(next(seen))

    monkeypatch.setattr(server, "scrape", fake_scrape)
    got = asyncio.run(server.wait_idle("http://x", timeout_s=5, poll_s=0.01))
    assert got["clean"] and got["running"] == 0

    async def never_idle(url, timeout_s=10.0):
        return Snap({"sglang:num_running_reqs": 2, "sglang:num_queue_reqs": 0})

    monkeypatch.setattr(server, "scrape", never_idle)
    got = asyncio.run(server.wait_idle("http://x", timeout_s=0.05, poll_s=0.01))
    assert not got["clean"] and got["running"] == 2
