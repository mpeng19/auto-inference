"""The sharp quality gate: did the kernel change what the model computes?

Everything here is synthetic and offline. The two halves that need a GPU -- the
scoring itself and the noise floor the thresholds should be set from -- are
noted where they are asserted, so a passing suite is not mistaken for a
validated gate.
"""
import pytest

from simulator.measure import equivalence as E
from simulator.measure import quality


def scoring(kind="reference", *, top1, logprob, token_id=None, i=0):
    """One prompt's worth of scored positions, in the shape the script writes."""
    ids = token_id if token_id is not None else list(range(100, 100 + len(top1)))
    return {"kind": kind, "model": "m", "n_prompts": 1,
            "scores": [{"i": i, "token_id": ids, "top1": list(top1),
                        "logprob": list(logprob)}]}


# ── compare ──────────────────────────────────────────────────────────────

def test_a_stack_that_changed_nothing_agrees_everywhere():
    ref = scoring(top1=[1, 2, 3, 4], logprob=[-0.1, -0.2, -0.3, -0.4])
    res = E.compare(ref, scoring("candidate", top1=[1, 2, 3, 4],
                                 logprob=[-0.1, -0.2, -0.3, -0.4]))
    assert res.aligned and res.n == 4 and res.n_prompts == 1
    assert res.top1_agreement == 1.0
    assert res.mean_abs_dlogprob == 0.0 and res.max_abs_dlogprob == 0.0


def test_a_moved_argmax_lowers_the_agreement():
    ref = scoring(top1=[1, 2, 3, 4], logprob=[-0.1] * 4)
    res = E.compare(ref, scoring("candidate", top1=[1, 2, 9, 4], logprob=[-0.1] * 4))
    assert res.top1_agreement == 0.75 and res.n == 4


def test_the_logprob_drift_is_reported_as_mean_and_worst():
    """The mean says whether the whole distribution moved; the max says whether
    one position went badly wrong. A kernel bug can look like either."""
    ref = scoring(top1=[1, 1, 1, 1], logprob=[-1.0, -1.0, -1.0, -1.0])
    res = E.compare(ref, scoring("candidate", top1=[1, 1, 1, 1],
                                 logprob=[-1.0, -1.1, -1.0, -1.5]))
    assert res.mean_abs_dlogprob == pytest.approx(0.15, abs=1e-6)
    assert res.max_abs_dlogprob == pytest.approx(0.5, abs=1e-6)


def test_positions_are_paired_only_where_both_runs_have_them():
    """A candidate that returned fewer positions is compared on the overlap,
    not padded -- padding would invent agreement."""
    ref = scoring(top1=[1, 2, 3, 4], logprob=[-0.1] * 4)
    res = E.compare(ref, scoring("candidate", top1=[1, 2], logprob=[-0.1] * 2))
    assert res.n == 2 and res.aligned


def test_runs_scored_on_different_tokens_are_refused_not_scored():
    """Teacher forcing is the whole design: if the token ids differ the two
    runs were asked different questions, and a number from that comparison
    would be worse than no number."""
    ref = scoring(top1=[1, 2], logprob=[-0.1, -0.2], token_id=[10, 11])
    cand = scoring("candidate", top1=[1, 2], logprob=[-0.1, -0.2],
                   token_id=[10, 99])
    res = E.compare(ref, cand)
    assert not res.aligned and "teacher forced" in res.note
    assert E.regressed(res)[0]


def test_two_runs_with_no_prompts_in_common_are_refused():
    res = E.compare(scoring(top1=[1], logprob=[-0.1], i=0),
                    scoring("candidate", top1=[1], logprob=[-0.1], i=7))
    assert not res.aligned and "no prompts in common" in res.note


def test_an_empty_pair_is_refused_rather_than_scored_perfect():
    res = E.compare({}, {})
    assert not res.aligned and E.regressed(res)[0]


# ── the gate ─────────────────────────────────────────────────────────────

def test_an_equivalent_stack_passes():
    ref = scoring(top1=list(range(100)), logprob=[-1.0] * 100)
    cand = scoring("candidate", top1=list(range(100)), logprob=[-1.0] * 100)
    assert E.regressed(E.compare(ref, cand)) == (False, "")


def test_small_jitter_passes_because_fp8_greedy_is_not_deterministic():
    """The floor is not zero. Kernel selection, batch composition and reduction
    order all move the last bits, and the coarse gate already shows how wide
    that is -- two stock sweeps scored 62% and 70% on the same 50 items."""
    top = list(range(100))
    ref = scoring(top1=top, logprob=[-1.0] * 100)
    # One position in a hundred flips, and the logits move by 0.01 nats.
    cand = scoring("candidate", top1=[999, *top[1:]], logprob=[-1.01] * 100)
    assert E.regressed(E.compare(ref, cand)) == (False, "")


def test_a_stack_computing_a_different_distribution_is_caught():
    top = list(range(100))
    ref = scoring(top1=top, logprob=[-1.0] * 100)
    cand = scoring("candidate", top1=[t + 1 for t in top], logprob=[-1.0] * 100)
    bad, why = E.regressed(E.compare(ref, cand))
    assert bad and "top-1 agreement" in why


def test_a_quiet_drift_is_caught_even_when_the_argmax_holds():
    """The failure GSM8K cannot see at all: every answer still right, every
    logit moved. This is the reason the gate reads logprobs and not accuracy."""
    top = list(range(100))
    ref = scoring(top1=top, logprob=[-1.0] * 100)
    cand = scoring("candidate", top1=top, logprob=[-1.4] * 100)
    bad, why = E.regressed(E.compare(ref, cand))
    assert bad and "logits moved" in why


def test_the_thresholds_are_arguments_so_a_measured_floor_can_replace_them():
    top = list(range(100))
    res = E.compare(scoring(top1=top, logprob=[-1.0] * 100),
                    scoring("candidate", top1=[999, *top[1:]],
                            logprob=[-1.0] * 100))
    assert not E.regressed(res, min_agreement=0.97)[0]
    assert E.regressed(res, min_agreement=0.999)[0]


# ── the script that does the scoring ─────────────────────────────────────

@pytest.mark.parametrize("mode", ["reference", "candidate"])
def test_the_generated_script_is_parseable_python(mode):
    """It is a string until a GPU runs it, so nothing else would catch a typo
    before it had cost six minutes of H100."""
    src = E.build_script("Qwen/Qwen3.8-27B-FP8", mode=mode,
                         out_path="/results/equivalence/out.json",
                         reference_path="/results/equivalence/ref.json")
    compile(src, f"<{mode}>", "exec")


def test_the_script_pins_the_same_dataset_the_quality_gate_uses():
    """The script re-derives the slice instead of importing `quality`, so this
    is what stops the two drifting into scoring different questions."""
    src = E.build_script("m", mode="reference", out_path="/o.json")
    for pinned in (quality.LONGBENCH_REPO, quality.LONGBENCH_FILE, quality.LONGBENCH_REV):
        assert pinned in src
    assert "step by step" in src, "the prompt template must travel too"


def test_a_candidate_script_will_not_be_built_without_a_reference():
    with pytest.raises(ValueError, match="reference"):
        E.build_script("m", mode="candidate", out_path="/o.json")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        E.build_script("m", mode="both", out_path="/o.json")


def test_the_reference_run_is_told_where_to_cache_itself():
    src = E.build_script("m", mode="reference",
                         out_path="/results/equivalence/ref.json")
    assert "/results/equivalence/ref.json" in src
    assert 'MODE == "reference" and os.path.exists(OUT_PATH)' in src, \
        "a cached reference must not pay for an engine load"


# ── identity ─────────────────────────────────────────────────────────────

def test_a_reference_is_keyed_by_model_and_prompt_set():
    """Reusing one model's reference for another is the one way this gate can
    be silently, confidently wrong."""
    a = E.reference_name("Qwen/Qwen3.8-27B-FP8")
    assert a != E.reference_name("Qwen/Qwen3-4B-FP8")
    assert a != E.reference_name("Qwen/Qwen3.8-27B-FP8", n=16)
    assert a != E.reference_name("Qwen/Qwen3.8-27B-FP8", max_new_tokens=32)
    assert a == E.reference_name("Qwen/Qwen3.8-27B-FP8")


def test_the_promptset_digest_is_computed_from_the_pin_not_the_data():
    """It has to work on a laptop with no network -- the whole offline suite
    would fail here otherwise, which is the assertion."""
    assert len(E.promptset_digest()) == 12


def test_a_candidate_file_is_named_after_the_stack_that_produced_it():
    a = E.candidate_name("m", "aaa")
    assert a != E.candidate_name("m", "bbb")
    # Same model and same prompt set as the reference it will be compared to,
    # so a mismatched pair is visible in the two file names alone.
    assert E.reference_name("m").removesuffix(".json").split("-")[1:] == \
        a.removesuffix(".json").split("-")[1:3]


# ── the two runs it orchestrates ─────────────────────────────────────────

def _fake_runs(monkeypatch, store, made):
    """Stand in for both Modal calls: the volume read and the workbench.

    The workbench fake parses the constants back out of the generated script,
    which is also a check that they are there to be read.
    """
    import json as _json

    from simulator import Simulator

    async def read(path):
        return store.get(path)

    async def workbench(self, text, files=None, timeout_s=600):
        const = dict(ln.split(" = ", 1) for ln in text.splitlines()
                     if " = " in ln and ln[0].isupper())
        mode = _json.loads(const["MODE"])
        out = _json.loads(const["OUT_PATH"])
        made.append({"mode": mode, "stack_digest": self.stack.digest,
                     "timeout_s": timeout_s, "out": out,
                     "reference": _json.loads(const["REFERENCE_PATH"])})
        store[out] = {"kind": mode, "scores": [
            {"i": 0, "token_id": [1, 2, 3], "top1": [7, 8, 9],
             "logprob": [-1.0, -1.0, -1.0]}]}
        return {"ok": True, "exit_code": 0, "cost_usd": 0.05, "elapsed_s": 200.0,
                "gpu": "H100", "dir": str(self.root)}

    monkeypatch.setattr(E, "_read", read)
    monkeypatch.setattr(Simulator, "workbench", workbench)


def test_a_cold_reference_is_computed_on_stock_and_the_candidate_on_the_diff(
        root, monkeypatch):
    """The reference has to be what the *unmodified* stack computes. Running it
    with the candidate applied would compare a stack against itself and pass
    every kernel ever written."""
    import asyncio

    from simulator import InferenceStack, Simulator

    store, made = {}, []
    _fake_runs(monkeypatch, store, made)
    stack = InferenceStack(files={"srt/managers/schedule_policy.py": "X = 1\n"})
    rec = asyncio.run(E.measure(Simulator(root_dir=root, stack=stack)))

    assert [m["mode"] for m in made] == ["reference", "candidate"]
    assert made[0]["stack_digest"] == InferenceStack.stock().digest
    assert made[1]["stack_digest"] == stack.digest
    assert made[1]["reference"] == made[0]["out"], \
        "the candidate must be scored against the reference just written"
    assert rec["ok"] and not rec["regressed"]
    assert rec["cost_usd"] == 0.10, "both runs are billed"


def test_a_warm_reference_costs_one_run_not_two(root, monkeypatch):
    """The whole reason it is persisted: the hundredth candidate for a model
    should not pay for the reference the first one computed."""
    import asyncio

    from simulator import Simulator

    store, made = {}, []
    _fake_runs(monkeypatch, store, made)
    store[f"{E.RESULTS_DIR}/{E.reference_name('Qwen/Qwen3.8-27B-FP8')}"] = {
        "kind": "reference", "scores": [{"i": 0, "token_id": [1, 2, 3],
                                         "top1": [7, 8, 9],
                                         "logprob": [-1.0, -1.0, -1.0]}]}
    rec = asyncio.run(E.measure(Simulator(root_dir=root)))
    assert [m["mode"] for m in made] == ["candidate"]
    assert rec["cost_usd"] == 0.05


def test_a_candidate_run_is_never_served_from_a_cached_file(root, monkeypatch):
    """Re-running is how the noise floor gets measured, and a memoised
    measurement cannot show its own variance."""
    import asyncio

    from simulator import Simulator

    store, made = {}, []
    _fake_runs(monkeypatch, store, made)
    sim = Simulator(root_dir=root)
    asyncio.run(E.measure(sim))
    asyncio.run(E.measure(sim))
    assert [m["mode"] for m in made] == ["reference", "candidate", "candidate"]


def test_a_failed_run_reports_the_reason_and_still_bills(root, monkeypatch):
    """A container that died still rented the GPU, and an agent's budget is
    checked against what comes back."""
    import asyncio

    from simulator import Simulator

    async def read(path):
        return None

    async def workbench(self, text, files=None, timeout_s=600):
        return {"ok": False, "exit_code": 1, "cost_usd": 0.03,
                "stderr": "torch.cuda.OutOfMemoryError: tried to allocate",
                "dir": str(self.root)}

    monkeypatch.setattr(E, "_read", read)
    monkeypatch.setattr(Simulator, "workbench", workbench)
    rec = asyncio.run(E.measure(Simulator(root_dir=root)))
    assert not rec["ok"] and "OutOfMemoryError" in rec["error"]
    assert rec["cost_usd"] == 0.03


def test_a_run_that_wrote_nothing_is_not_read_as_equivalent(root, monkeypatch):
    """`ok=0` with an empty volume must never fall through to a pass."""
    import asyncio

    from simulator import Simulator

    async def read(path):
        return None

    async def workbench(self, text, files=None, timeout_s=600):
        return {"ok": True, "exit_code": 0, "cost_usd": 0.05, "dir": str(self.root)}

    monkeypatch.setattr(E, "_read", read)
    monkeypatch.setattr(Simulator, "workbench", workbench)
    rec = asyncio.run(E.measure(Simulator(root_dir=root)))
    assert not rec["ok"] and "wrote nothing" in rec["error"]


def test_scoring_script_guards_main_for_spawned_workers():
    """sglang.Engine spawns its scheduler, which re-imports the script; the
    first reference run died on exactly this."""
    from simulator.measure import equivalence as eq

    src = eq.build_script("Qwen/Qwen3.8-27B-FP8", mode="reference", out_path="/results/x.json")
    assert 'if __name__ == "__main__":' in src
    assert src.rstrip().endswith("raise SystemExit(main())")
    compile(src, "script.py", "exec")


def test_equivalence_prompts_are_long_context_and_the_reference_name_moved():
    from simulator.measure import equivalence as eq
    from simulator.measure import quality

    src = eq.build_script("Qwen/Qwen3.8-27B-FP8", mode="reference", out_path="/results/x.json")
    assert 'SETS = ["hotpotqa", "2wikimqa"]' in src and "MAX_CHARS" in src
    assert "zipfile" in src and "r[\"context\"][:MAX_CHARS]" in src
    compile(src, "script.py", "exec")
    name = eq.reference_name("Qwen/Qwen3.8-27B-FP8")
    assert "a50ba120b271" not in name                 # the GSM8K-era reference is retired
    assert quality.LONGBENCH_REV in src
