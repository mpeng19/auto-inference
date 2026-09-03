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
