"""Run SGLang's own benchmark against the same server, as a second opinion.

Our load generator and `sglang.benchmark.serving` were written independently
and measure the same things -- TTFT and TPOT percentiles at a fixed
concurrency. If they agree, that is real evidence our client is sound. If they
disagree, one of us has a bug and it is worth knowing which before pricing
anything on it.

Not a dependency: it runs as a subprocess, its result is recorded beside ours,
and nothing downstream reads it. That matters because SGLang moved
`sglang.bench_serving` to `sglang.benchmark.serving` *in the version we pin*,
leaving a deprecation shim -- an actively reorganised surface is a bad thing to
put in the path of a price.

Their `agentic-trace` dataset is the closest match to what we replay: multi-turn
conversations fed round by round with the server's real reply appended, which
is the prefix-growth structure the whole cache-hit story depends on.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys


def run(base_url: str, model: str, *, concurrency: int, num_prompts: int = 200,
        dataset: str = "random", dataset_path: str = "",
        random_input: int = 20583, random_output: int = 2076,
        timeout_s: float = 1800) -> dict:
    """Invoke SGLang's benchmark and return its metrics, or why it did not run."""
    host, _, port = base_url.replace("http://", "").replace("https://", "").partition(":")
    cmd = [sys.executable, "-m", "sglang.benchmark.serving",
           "--backend", "sglang-oai-chat", "--host", host, "--port", port or "30000",
           "--model", model, "--dataset-name", dataset,
           "--num-prompts", str(num_prompts),
           "--max-concurrency", str(concurrency),
           "--request-rate", "inf",
           "--output-file", "/dev/null"]
    if dataset == "random":
        cmd += ["--random-input-len", str(random_input),
                "--random-output-len", str(random_output),
                "--random-range-ratio", "1.0"]
    elif dataset_path:
        cmd += ["--dataset-path", dataset_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception as e:
        return {"ran": False, "error": f"{type(e).__name__}: {e}"}
    if r.returncode != 0:
        return {"ran": False, "error": (r.stderr or r.stdout)[-800:], "cmd": cmd}
    return {"ran": True, "metrics": _parse(r.stdout), "cmd": cmd}


def _parse(stdout: str) -> dict:
    """Pull the metrics out of the human-readable table it prints.

    It writes JSON only with `--output-file`, and we deliberately do not want a
    file on the results volume for a diagnostic, so the table is the interface.
    """
    want = {
        "Request throughput (req/s):": "request_throughput",
        "Output token throughput (tok/s):": "output_throughput",
        "Mean TTFT (ms):": "mean_ttft_ms",
        "Median TTFT (ms):": "median_ttft_ms",
        "P99 TTFT (ms):": "p99_ttft_ms",
        "Mean TPOT (ms):": "mean_tpot_ms",
        "Median TPOT (ms):": "median_tpot_ms",
        "P99 TPOT (ms):": "p99_tpot_ms",
    }
    out: dict = {}
    for line in stdout.splitlines():
        for key, name in want.items():
            if key in line:
                tail = line.split(key)[-1].strip()
                with contextlib.suppress(ValueError, IndexError):
                    out[name] = float(tail.split()[0])
    return out


def compare(ours: dict, theirs: dict, tolerance: float = 0.20) -> dict:
    """Agreement on the metrics both report, as a ratio per field.

    `tolerance` is generous (20%): the two drive different traffic, so this is
    checking for a factor-of-two disagreement -- a broken client -- not for
    equality.
    """
    pairs = {"mean_ttft_ms": ("ttft_ms", "mean"),
             "median_ttft_ms": ("ttft_ms", "p50"),
             "p99_ttft_ms": ("ttft_ms", "p99"),
             "mean_tpot_ms": ("tpot_ms", "mean"),
             "median_tpot_ms": ("tpot_ms", "p50"),
             "p99_tpot_ms": ("tpot_ms", "p99")}
    rows, worst = {}, 0.0
    for their_key, (block, stat) in pairs.items():
        a = (ours.get(block) or {}).get(stat)
        b = theirs.get(their_key)
        if not a or not b:
            continue
        ratio = a / b
        rows[their_key] = {"ours": round(a, 2), "theirs": round(b, 2),
                           "ratio": round(ratio, 3)}
        worst = max(worst, abs(ratio - 1.0))
    return {"fields": rows, "worst_deviation": round(worst, 3),
            "agrees": bool(rows) and worst <= tolerance,
            "note": "" if rows else "no overlapping fields to compare"}
