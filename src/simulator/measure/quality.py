"""Did the model still answer correctly, or did we only make it faster?

`canary.py` digests six short outputs on an idle server. That catches gross
breakage -- a scheduler that truncates, a sampler that changed -- and would
miss the failure that actually matters here: an optimisation that subtly
changes numerics and degrades reasoning while every latency number improves.

An agent maximising goodput has an obvious cheat available: serve worse
answers faster. Nothing in the price model can see it. So a diff is measured
on accuracy as well, and **a speed win that costs accuracy is not a win**.

Two suites, both exactly scorable so no judge is needed and no second model is
in the loop:

    gsm8k   grade-school word problems. Answer is a number after '####'.
            Multi-step, so a small numerical error compounds into a wrong
            answer -- which is exactly the sensitivity we want.
    mmlu    four-way multiple choice. One token of output, so it is cheap,
            but a single token is a weak signal per item; it earns its place
            by covering knowledge rather than arithmetic.

Both are pinned by dataset revision. A benchmark that moves underneath you
cannot detect a regression -- it *is* one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

GSM8K_REPO, GSM8K_FILE = "openai/gsm8k", "main/test-00000-of-00001.parquet"
MMLU_REPO, MMLU_FILE = "cais/mmlu", "all/test-00000-of-00001.parquet"
# Pinned. `main` would let the benchmark drift, which would show up as a
# quality regression in whatever diff happened to run next.
GSM8K_REV = "e53f048856ff4f594e959d75785d2c2d37b678ee"
MMLU_REV = "c30699e8356da336a370243923dbaf21066bb9fe"

# Long-context QA. GSM8K prompts are a few hundred tokens, and the first
# build-mode "win" (page selection that only engages above 4,096 tokens)
# scored 70% on GSM8K because the path it changed never ran. A gate has to
# exercise the code it guards: these contexts are ~15k tokens, the market's
# shape, and the metric is LongBench's own token F1.
LONGBENCH_REPO = "THUDM/LongBench"
LONGBENCH_FILE = "data.zip"
LONGBENCH_REV = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_SETS = ("hotpotqa", "2wikimqa")
LONGBENCH_MAX_CHARS = 72_000            # ~18k tokens; keeps under the context length
LONGBENCH_PROMPT = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\nThe following are given passages.\n"
    "{context}\n\nAnswer the question based on the given passages. Only give me "
    "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:")

GSM8K_PROMPT = (
    "Solve the problem. Think step by step, then give the final numeric answer "
    "on its own last line in the form '#### <number>'.\n\nProblem: {q}\n")
MMLU_PROMPT = (
    "Answer with a single letter: A, B, C or D. No explanation.\n\n"
    "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    answer: str
    max_tokens: int


@dataclass
class QualityResult:
    suite: str
    n: int = 0
    correct: float = 0.0            # a sum of per-item scores; a count for exact suites
    errors: int = 0
    items: list[dict] = field(default_factory=list)
    baseline_accuracy: float | None = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def delta_pct(self) -> float | None:
        """Percentage points against the baseline, when one is known."""
        if self.baseline_accuracy is None:
            return None
        return round((self.accuracy - self.baseline_accuracy) * 100, 2)

    def as_dict(self) -> dict:
        return {"suite": self.suite, "n": self.n, "correct": self.correct,
                "errors": self.errors, "accuracy": round(self.accuracy, 4),
                # A few examples travel with the number: a score nobody can
                # look behind is a score nobody can debug (stock's first
                # LongBench run said 6% and the record could not say why).
                "examples": [{k: (str(v)[:160] if isinstance(v, str) else v)
                              for k, v in it.items()} for it in self.items[:5]],
                "baseline_accuracy": self.baseline_accuracy,
                "delta_pct": self.delta_pct}


# ── loading, pinned and cached ───────────────────────────────────────────

def load(suite: str = "gsm8k", n: int = 100, seed: int = 0) -> list[Item]:
    """A fixed, deterministic slice. Same items every run, or it proves nothing."""
    import random

    import pandas as pd
    from huggingface_hub import hf_hub_download

    if suite == "gsm8k":
        p = hf_hub_download(GSM8K_REPO, GSM8K_FILE, repo_type="dataset",
                            revision=GSM8K_REV)
        df = pd.read_parquet(p)
        rows = df.to_dict("records")
        random.Random(seed).shuffle(rows)
        out = []
        for i, r in enumerate(rows[:n]):
            gold = str(r["answer"]).split("####")[-1].strip()
            out.append(Item(f"gsm8k-{i}", GSM8K_PROMPT.format(q=r["question"]),
                            gold, max_tokens=320))
        return out

    if suite == "longbench":
        rows = load_longbench_rows()
        random.Random(seed).shuffle(rows)
        return [Item(f"longbench-{i}",
                     LONGBENCH_PROMPT.format(context=_truncate_middle(r["context"], LONGBENCH_MAX_CHARS),
                                             input=r["input"]),
                     "\x1f".join(_answers(r)), max_tokens=32)
                for i, r in enumerate(rows[:n])]

    p = hf_hub_download(MMLU_REPO, MMLU_FILE, repo_type="dataset",
                        revision=MMLU_REV)
    df = pd.read_parquet(p)
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:n]):
        ch = list(r["choices"])
        out.append(Item(f"mmlu-{i}",
                        MMLU_PROMPT.format(q=r["question"], a=ch[0], b=ch[1],
                                           c=ch[2], d=ch[3]),
                        "ABCD"[int(r["answer"])], max_tokens=4))
    return out


def load_longbench_rows(sets: tuple[str, ...] = LONGBENCH_SETS) -> list[dict]:
    """The pinned LongBench rows for `sets`, from its data.zip."""
    import json
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LONGBENCH_REPO, LONGBENCH_FILE, repo_type="dataset",
                           revision=LONGBENCH_REV)
    rows: list[dict] = []
    with zipfile.ZipFile(path) as z:
        for name in sets:
            rows += [json.loads(line) for line in
                     z.read(f"data/{name}.jsonl").decode("utf-8").splitlines() if line.strip()]
    return rows


def _truncate_middle(text: str, max_chars: int) -> str:
    """LongBench's own truncation: keep the head and the tail, drop the
    middle. Head-only truncation drops the passages nearest the question."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _answers(r: dict) -> list[str]:
    a = r.get("answers")
    if isinstance(a, str):
        try:
            import ast
            a = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            a = [a]
    return [str(x) for x in (a or [])]


def _qa_normalise(s: str) -> list[str]:
    """LongBench's normalisation: lowercase, drop punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return [w for w in s.split() if w not in ("a", "an", "the")]


def qa_f1(prediction: str, golds: list[str]) -> float:
    """Token F1 against the best of several gold answers, LongBench-style."""
    from collections import Counter

    pred = _qa_normalise(prediction.strip().split("\n")[0])
    best = 0.0
    for g in golds:
        gold = _qa_normalise(g)
        common = sum((Counter(pred) & Counter(gold)).values())
        if not common:
            continue
        p, r = common / len(pred), common / len(gold)
        best = max(best, 2 * p * r / (p + r))
    return best


# ── scoring, exact and dumb on purpose ───────────────────────────────────

def score(suite: str, output: str, gold: str) -> float:
    """1.0 / 0.0 for exact suites; token F1 in [0, 1] for long-context QA.
    `correct` sums these, so `accuracy` is mean F1 there."""
    if suite == "longbench":
        return qa_f1(output, gold.split("\x1f"))
    if suite == "mmlu":
        m = re.search(r"\b([ABCD])\b", output.strip().upper())
        return float(bool(m) and m.group(1) == gold.strip().upper())
    # GSM8K: prefer the '####' form, else the last number in the output. The
    # fallback matters because a degraded model often still reaches an answer
    # while losing the format, and calling that wrong would blame the wrong
    # thing.
    text = output.split("####")[-1] if "####" in output else output
    nums = _NUM.findall(text.replace(",", ""))
    if not nums:
        return 0.0
    return float(_norm(nums[-1]) == _norm(gold))


def _norm(x: str) -> str:
    try:
        f = float(str(x).replace(",", "").rstrip("."))
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return str(x).strip()


# ── running it against a live server ─────────────────────────────────────

async def run(base_url: str, model: str, suite: str = "gsm8k", n: int = 100,
              seed: int = 0, concurrency: int = 16,
              baseline_accuracy: float | None = None,
              items: list[Item] | None = None) -> QualityResult:
    """Score a suite against a running server.

    Deliberately at low concurrency and greedy: this measures the model, not
    the scheduler, and a quality number that moves with load would be useless
    for deciding whether a diff broke anything.
    """
    import asyncio

    from .server import complete

    items = items if items is not None else load(suite, n, seed)
    res = QualityResult(suite=suite, baseline_accuracy=baseline_accuracy)
    sem = asyncio.Semaphore(concurrency)

    async def one(it: Item) -> dict:
        async with sem:
            try:
                out = await complete(base_url, model, it.prompt, it.max_tokens)
            except Exception as e:
                return {"id": it.id, "error": f"{type(e).__name__}: {e}"}
        return {"id": it.id, "ok": score(suite, out or "", it.answer),
                "gold": it.answer.replace("\x1f", " | "), "got": (out or "")[-120:]}

    for r in await asyncio.gather(*(one(i) for i in items)):
        res.n += 1
        if "error" in r:
            res.errors += 1
        else:
            res.correct += float(r["ok"])
        res.items.append(r)
    return res


def regressed(result: QualityResult, tolerance_pp: float = 10.0) -> tuple[bool, str]:
    """Is this a quality regression? Returns (yes, why).

    `tolerance_pp` is percentage points, not a ratio. The gate has to sit
    outside the run-to-run noise of the *stock* stack or it rejects stock:
    two 1xH100 sweeps on 2026-09-02, same 50 items, same greedy decoding,
    scored 62% and 70% -- FP8 kernels are not bitwise deterministic and four
    borderline items flipped. What this exists to catch is a stack that
    serves a different model (a broken KV path scores near zero), and ten
    points on 100 items is well outside the noise and well inside that.
    """
    if result.n == 0:
        return True, "no items scored"
    if result.errors > result.n * 0.05:
        return True, f"{result.errors}/{result.n} requests errored"
    d = result.delta_pct
    if d is None:
        return False, ""
    if d < -tolerance_pp:
        return True, (f"accuracy fell {abs(d):.1f} points "
                      f"({result.baseline_accuracy:.1%} -> {result.accuracy:.1%}) "
                      f"on {result.suite}; a speed win that costs accuracy is not a win")
    return False, ""


def summarise(results: list[QualityResult]) -> str:
    return json.dumps([r.as_dict() for r in results], indent=1)
