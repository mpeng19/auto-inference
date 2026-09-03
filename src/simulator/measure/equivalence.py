"""Did the kernel change what the model computes?

`quality.py` asks the same question through GSM8K accuracy, and for a scheduler
tweak that is the right instrument: a stack that serves a different model
scores near zero. For a kernel it is far too blunt. Accuracy is a coarse
function of the logits -- a rewritten attention kernel can perturb every
position and still get the same 36 of 50 answers right, and the noise floor is
so wide (two *stock* sweeps on the same 50 items scored 62% and 70%) that
anything smaller than a catastrophe hides inside it.

So measure the logits directly, teacher forced:

  1. **A fixed prompt set** -- the same pinned GSM8K slice `quality.py` uses,
     first 32 questions.
  2. **A reference**, computed once on *stock*: greedy-generate 64 tokens per
     prompt, then score the whole prompt+completion sequence and keep, at every
     completion position, the argmax token and the logprob of the token that
     was actually taken. Persisted to the results volume under the model and
     prompt-set digests, so the next hundred candidates read it instead of
     recomputing it.
  3. **The candidate** scores the *same* sequences. No generation: teacher
     forcing means both stacks are asked about identical positions, so a
     difference is the stack and not a different sentence.

That gives ~2,000 paired positions per run instead of 50 booleans, and reports
`top1_agreement` (did the argmax move), `mean_abs_dlogprob` and
`max_abs_dlogprob` (by how much). A kernel that changes the model shows up in
all three long before GSM8K notices.

**What it costs.** An engine load is 3-5 minutes before the first prompt is
scored, so a candidate run is ~5-8 GPU-minutes -- against 17-35 for a sweep,
and it answers a question a sweep cannot answer at all. A cold reference pays
that twice, once.

**The thresholds are provisional.** FP8 greedy decoding is not bitwise
deterministic: kernel selection, batch composition and reduction order all move
the last bits, and none of that is a defect in the candidate. Until a
stock-vs-stock run measures the floor, `min_agreement=0.97` and
`max_mean_dlogprob=0.05` are guesses shaped to sit clear of it -- run
`simulate equivalence --root R` with no `--stack` twice and set them from what
comes back.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from . import quality

# 32 prompts x 64 tokens is ~2,000 scored positions: enough that a one-in-a-
# hundred disagreement is visible, small enough that scoring is one prefill
# batch and the run is dominated by the engine load either way.
N_PROMPTS = 32
MAX_NEW_TOKENS = 64
SEED = 0

# On the results volume, which is where a reference has to live: it is shared
# by every agent in the fleet and outlives the container that computed it.
RESULTS_DIR = "/results/equivalence"

# Provisional; see the module docstring.
MIN_AGREEMENT = 0.97
MAX_MEAN_DLOGPROB = 0.05


@dataclass(frozen=True)
class EquivalenceResult:
    """Two stacks scored on identical sequences, position by position."""
    n: int = 0
    n_prompts: int = 0
    top1_agreement: float = 0.0
    mean_abs_dlogprob: float = 0.0
    max_abs_dlogprob: float = 0.0
    # False when the two runs did not score the same tokens. Never a numerical
    # finding -- it means the comparison is invalid, so it is reported apart
    # from the numbers rather than as a bad-looking one.
    aligned: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def summary(self) -> str:
        return (f"{self.n} positions over {self.n_prompts} prompts   "
                f"top-1 agreement {self.top1_agreement:.4f}   "
                f"|dlogprob| mean {self.mean_abs_dlogprob:.4f} "
                f"max {self.max_abs_dlogprob:.4f}"
                + ("" if self.aligned else f"   NOT ALIGNED: {self.note}"))


# ── identity: what a reference is a reference *for* ──────────────────────

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def model_digest(model: str) -> str:
    return _sha(model)


def promptset_digest(n: int = N_PROMPTS, max_new_tokens: int = MAX_NEW_TOKENS,
                     seed: int = SEED) -> str:
    """Everything that decides which sequences get scored.

    Computed from the pinned constants rather than from the downloaded rows, so
    it needs no network: the dataset revision is what makes the slice fixed,
    and if any of this moves the old reference must not be reused.
    """
    return _sha(json.dumps({"repo": quality.LONGBENCH_REPO,
                            "file": quality.LONGBENCH_FILE,
                            "revision": quality.LONGBENCH_REV,
                            "sets": list(quality.LONGBENCH_SETS),
                            "max_chars": quality.LONGBENCH_MAX_CHARS,
                            "truncation": "middle",
                            "prompt": quality.LONGBENCH_PROMPT,
                            "n": n, "seed": seed,
                            "max_new_tokens": max_new_tokens}, sort_keys=True))


def reference_name(model: str, n: int = N_PROMPTS,
                   max_new_tokens: int = MAX_NEW_TOKENS, seed: int = SEED) -> str:
    return (f"reference-{model_digest(model)}-"
            f"{promptset_digest(n, max_new_tokens, seed)}.json")


def candidate_name(model: str, stack_digest: str, n: int = N_PROMPTS,
                   max_new_tokens: int = MAX_NEW_TOKENS, seed: int = SEED) -> str:
    return (f"candidate-{model_digest(model)}-"
            f"{promptset_digest(n, max_new_tokens, seed)}-{stack_digest}.json")


# ── the scoring script, run inside the workbench ─────────────────────────

_BODY = '''

def load_prompts():
    """The same pinned slice `simulator.measure.quality.load` takes.

    Re-derived here rather than imported: this script runs against the
    *candidate* stack in a container where only sglang and the standard
    scientific stack are guaranteed, and the constants above are interpolated
    from quality.py so the two cannot drift apart silently.
    """
    import random
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV)
    rows = []
    with zipfile.ZipFile(path) as z:
        for name in SETS:
            rows += [json.loads(line) for line in
                     z.read(f"data/{name}.jsonl").decode("utf-8").splitlines() if line.strip()]
    random.Random(SEED).shuffle(rows)
    # ~15k-token contexts: the market's shape, and long enough that anything
    # gated on sequence length (page selection, sparse attention, KV
    # compression) is actually running while it is scored.
    def trunc(t):
        if len(t) <= MAX_CHARS:
            return t
        h = MAX_CHARS // 2
        return t[:h] + "\\n...\\n" + t[-h:]

    return [PROMPT.format(context=trunc(r["context"]), input=r["input"])
            for r in rows[:N_PROMPTS]]


def score(engine, prompt_ids, completion_ids):
    """Teacher-forced scoring: no generation, identical positions both runs.

    `logprob_start_len` is where the model starts reporting, so the returned
    entries cover the sequence from there on and there is one fewer of them
    than tokens (position i's logprob is conditioned on everything before it).
    Both runs pass the same value over the same ids, so the arrays line up by
    construction -- and `token_id` is kept so `compare` can prove it did.
    """
    seqs = [list(p) + list(c) for p, c in zip(prompt_ids, completion_ids)]
    starts = [len(p) for p in prompt_ids]
    out = engine.generate(
        input_ids=seqs,
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True, logprob_start_len=starts, top_logprobs_num=1)
    if isinstance(out, dict):
        out = [out]

    scores = []
    for i, r in enumerate(out):
        meta = r.get("meta_info") or {}
        chosen = meta.get("input_token_logprobs") or []
        top = meta.get("input_top_logprobs") or []
        tok, top1, logp = [], [], []
        for j, ent in enumerate(chosen):
            # The first position of a sequence has no logprob, and SGLang
            # reports that as None rather than dropping it.
            if not ent or ent[0] is None or j >= len(top) or not top[j]:
                continue
            logp.append(round(float(ent[0]), 6))
            tok.append(int(ent[1]))
            top1.append(int(top[j][0][1]))
        scores.append({"i": i, "token_id": tok, "top1": top1, "logprob": logp})
    return scores


def main():
    import os
    import time

    if MODE == "reference" and os.path.exists(OUT_PATH):
        # Computed once, then read by every candidate after it. Exiting before
        # the engine loads is what makes the cached case ~30 seconds of H100
        # rather than four minutes.
        print(json.dumps({"wrote": OUT_PATH, "cached": True}))
        return 0

    t0 = time.time()
    prompts = load_prompts()
    import sglang

    engine = sglang.Engine(
        model_path=MODEL, mem_fraction_static=MEM_FRACTION, tp_size=TP_SIZE,
        # No prefix cache: a shared prefix served from the radix tree takes a
        # different path through attention than one recomputed, and this is
        # measuring exactly that kind of difference.
        disable_radix_cache=True, random_seed=SEED, log_level="error")
    print("engine ready in %.1fs" % (time.time() - t0), flush=True)

    if MODE == "reference":
        gen = engine.generate(
            prompt=prompts,
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS,
                             "temperature": 0.0},
            # The completion is needed as token ids, not text: re-tokenising
            # text is lossy and would score a different sequence.
            return_logprob=True)
        tokenizer = engine.tokenizer_manager.tokenizer
        prompt_ids = [list(tokenizer.encode(p)) for p in prompts]
        completion_ids = [[int(t[1]) for t in
                           (g["meta_info"].get("output_token_logprobs") or [])]
                          for g in gen]
    else:
        with open(REFERENCE_PATH) as f:
            ref = json.load(f)
        prompt_ids = ref["prompt_ids"]
        completion_ids = ref["completion_ids"]

    scores = score(engine, prompt_ids, completion_ids)
    rec = {"kind": MODE, "model": MODEL, "model_digest": MODEL_DIGEST,
           "promptset_digest": PROMPTSET_DIGEST, "sglang_version":
               getattr(sglang, "__version__", "unknown"),
           "n_prompts": len(prompts), "max_new_tokens": MAX_NEW_TOKENS,
           "created_at": time.time(), "prompt_ids": prompt_ids,
           "completion_ids": completion_ids, "scores": scores}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(rec, f)

    n = sum(len(s["logprob"]) for s in scores)
    # Deliberately small: the workbench keeps the last 20 KB of stdout, and the
    # arrays go to the volume for `read_results` to fetch.
    print(json.dumps({"wrote": OUT_PATH, "n_positions": n,
                      "n_prompts": len(prompts)}))
    engine.shutdown()
    return 0


# Guarded because sglang.Engine starts its scheduler with the "spawn" start
# method, which re-imports this script in the child; an unguarded call would
# try to start a second engine from inside the first and multiprocessing
# refuses ("_check_not_importing_main"). Cost us one reference run.
if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_script(model: str, *, mode: str, out_path: str,
                 reference_path: str = "", n_prompts: int = N_PROMPTS,
                 max_new_tokens: int = MAX_NEW_TOKENS, seed: int = SEED,
                 mem_fraction_static: float = 0.85, tp_size: int = 1) -> str:
    """The scoring script, as text, for `workbench` to run.

    Self-contained on purpose. It has to execute against the *candidate*
    sglang, and a script that imported `simulator` would be importing this
    machinery back into the container it is supposed to be measuring from the
    outside. The pinned dataset constants are interpolated from `quality.py`,
    so there is still one source of truth for what the prompt set is.
    """
    if mode not in ("reference", "candidate"):
        raise ValueError(f"mode must be 'reference' or 'candidate', got {mode!r}")
    if mode == "candidate" and not reference_path:
        raise ValueError("a candidate run needs the reference it is scored against")

    consts = {
        "MODE": mode,
        "MODEL": model,
        "MODEL_DIGEST": model_digest(model),
        "PROMPTSET_DIGEST": promptset_digest(n_prompts, max_new_tokens, seed),
        "REPO": quality.LONGBENCH_REPO,
        "FILE": quality.LONGBENCH_FILE,
        "REV": quality.LONGBENCH_REV,
        "SETS": list(quality.LONGBENCH_SETS),
        "MAX_CHARS": quality.LONGBENCH_MAX_CHARS,
        "PROMPT": quality.LONGBENCH_PROMPT,
        "N_PROMPTS": n_prompts,
        "MAX_NEW_TOKENS": max_new_tokens,
        "SEED": seed,
        "MEM_FRACTION": mem_fraction_static,
        "TP_SIZE": tp_size,
        "OUT_PATH": out_path,
        "REFERENCE_PATH": reference_path,
    }
    head = ['"""Token-level equivalence scoring. Generated by '
            'simulator.measure.equivalence."""', "import json", ""]
    head += [f"{k} = {json.dumps(v)}" for k, v in consts.items()]
    return "\n".join(head) + "\n" + _BODY


# ── comparing two scorings ───────────────────────────────────────────────

def compare(reference: dict, candidate: dict) -> EquivalenceResult:
    """Two scoring records in, one verdict out. Pure; no GPU, no network.

    Every position is paired by prompt index and offset, and the token actually
    scored is checked on both sides. A mismatch there means the runs were not
    asked the same question, which is a broken comparison rather than a large
    difference -- reporting it as a low agreement would be a lie in the
    dangerous direction.
    """
    ref = {s["i"]: s for s in (reference or {}).get("scores", [])}
    cand = {s["i"]: s for s in (candidate or {}).get("scores", [])}
    shared = sorted(set(ref) & set(cand))
    if not shared:
        return EquivalenceResult(aligned=False,
                                 note="no prompts in common between the two runs")

    n = matches = 0
    total_d = max_d = 0.0
    misaligned = []
    for i in shared:
        a, b = ref[i], cand[i]
        k = min(len(a["logprob"]), len(b["logprob"]))
        if a["token_id"][:k] != b["token_id"][:k]:
            misaligned.append(i)
            continue
        for j in range(k):
            n += 1
            matches += int(a["top1"][j] == b["top1"][j])
            d = abs(float(a["logprob"][j]) - float(b["logprob"][j]))
            total_d += d
            max_d = max(max_d, d)

    if misaligned:
        return EquivalenceResult(
            n=n, n_prompts=len(shared) - len(misaligned), aligned=False,
            note=(f"{len(misaligned)} prompt(s) scored different tokens "
                  f"(first: {misaligned[0]}); the runs were not teacher forced "
                  "on the same sequences"))
    if n == 0:
        return EquivalenceResult(n_prompts=len(shared), aligned=False,
                                 note="the prompts matched but scored no positions")
    return EquivalenceResult(
        n=n, n_prompts=len(shared),
        top1_agreement=round(matches / n, 6),
        mean_abs_dlogprob=round(total_d / n, 6),
        max_abs_dlogprob=round(max_d, 6))


def regressed(result: EquivalenceResult, min_agreement: float = MIN_AGREEMENT,
              max_mean_dlogprob: float = MAX_MEAN_DLOGPROB) -> tuple[bool, str]:
    """Has this stack changed what the model computes? Returns (yes, why).

    Both thresholds are **provisional** and should be set from a measured noise
    floor rather than kept. The floor is not zero: FP8 greedy decoding is not
    bitwise deterministic -- kernel selection, batch composition and reduction
    order all move the last bits -- and the coarse version of this gate already
    shows how wide that can be, with two stock sweeps scoring 62% and 70% on
    the same 50 GSM8K items. Run this stock against stock twice and read the
    numbers off; until then 0.97 / 0.05 are chosen to sit clear of what
    ordinary numerical jitter should produce, not measured to.

    `aligned=False` is always a rejection: the comparison did not happen.
    """
    if not result.aligned:
        return True, result.note or "the two runs did not score the same sequences"
    if result.n == 0:
        return True, "no positions scored"
    if result.top1_agreement < min_agreement:
        return True, (f"top-1 agreement {result.top1_agreement:.4f} is below "
                      f"{min_agreement:.4f} over {result.n} positions; this "
                      "stack computes a different distribution")
    if result.mean_abs_dlogprob > max_mean_dlogprob:
        return True, (f"mean |dlogprob| {result.mean_abs_dlogprob:.4f} exceeds "
                      f"{max_mean_dlogprob:.4f}; the argmax mostly held but the "
                      "logits moved")
    return False, ""


# ── running it ───────────────────────────────────────────────────────────

async def measure(sim, timeout_s: int = 1800,
                  min_agreement: float = MIN_AGREEMENT,
                  max_mean_dlogprob: float = MAX_MEAN_DLOGPROB) -> dict:
    """Score `sim.stack` against stock, on the workbench. Returns a dict.

    Two workbench runs at worst, one in the common case: the reference is keyed
    by (model, prompt set) and cached on the results volume, so only the first
    candidate for a given model pays for it. The candidate is never cached --
    re-running it is how the noise floor gets measured, and a memoised
    measurement cannot show its own variance.
    """
    from dataclasses import replace

    from ..stack import InferenceStack

    ref_path = f"{RESULTS_DIR}/{reference_name(sim.model)}"
    cand_path = f"{RESULTS_DIR}/{candidate_name(sim.model, sim.stack.digest)}"
    out: dict = {"ok": False, "model": sim.model, "stack": sim.stack.describe(),
                 "stack_digest": sim.stack.digest, "reference_path": ref_path,
                 "candidate_path": cand_path, "cost_usd": 0.0, "runs": []}

    def _kw(mode: str) -> dict:
        return {"mode": mode, "mem_fraction_static": sim.mem_fraction_static,
                "tp_size": max(1, sim.n_gpu)}

    have = await _read(ref_path)
    if have is None:
        # Stock, deliberately: the reference is what the unmodified stack
        # computes, and running it with the candidate applied would compare a
        # stack against itself and always pass.
        stock = replace(sim, stack=InferenceStack.stock())
        r = await stock.workbench(
            build_script(sim.model, out_path=ref_path, **_kw("reference")),
            timeout_s=timeout_s)
        out["runs"].append({"kind": "reference", **_trim(r)})
        out["cost_usd"] += float(r.get("cost_usd") or 0.0)
        if not r.get("ok"):
            out["error"] = "the reference run failed: " + _tail(r)
            return out
        have = await _read(ref_path)
        if have is None:
            out["error"] = f"the reference run wrote nothing to {ref_path}"
            return out

    r = await sim.workbench(
        build_script(sim.model, out_path=cand_path, reference_path=ref_path,
                     **_kw("candidate")),
        timeout_s=timeout_s)
    out["runs"].append({"kind": "candidate", **_trim(r)})
    out["cost_usd"] += float(r.get("cost_usd") or 0.0)
    if not r.get("ok"):
        out["error"] = "the candidate run failed: " + _tail(r)
        return out

    cand = await _read(cand_path)
    if cand is None:
        out["error"] = f"the candidate run wrote nothing to {cand_path}"
        return out

    res = compare(have, cand)
    bad, why = regressed(res, min_agreement, max_mean_dlogprob)
    out.update({"ok": True, "result": res.as_dict(), "regressed": bad,
                "why": why, "cost_usd": round(out["cost_usd"], 4),
                "summary": res.summary()})
    return out


def _trim(r: dict) -> dict:
    """A workbench result, minus the parts nobody reads twice."""
    return {k: r.get(k) for k in ("ok", "exit_code", "elapsed_s", "gpu",
                                  "cost_usd", "dir")}


def _tail(r: dict) -> str:
    return ((r.get("stderr") or r.get("stdout") or "no output").strip()
            .splitlines() or ["no output"])[-1][:400]


async def _read(path: str) -> dict | None:
    """One scoring record off the results volume, or None if it is not there."""
    import modal

    from ..api import APP_NAME

    fn = modal.Function.from_name(APP_NAME, "read_results")
    got = await fn.remote.aio([path])
    row = (got or {}).get(path) or {}
    return row.get("json") if row.get("ok") else None
