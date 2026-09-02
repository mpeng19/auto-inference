"""Typed configuration.

`ServingConfig` *is* the search space: every field here is a knob an optimizer
is allowed to turn. Anything not in this dataclass is held fixed, and anything
held fixed must be recorded in the run record so results stay comparable.

Frozen + hashable on purpose. The digest is what dedupes a sweep and what keys
a cached result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


# Flag names verified against sglang 0.5.18 by `probe.py::probe_env` on
# 2026-08-29: all 12 present. Re-run that probe after any SGLang upgrade —
# SGLang renames flags between minor versions, and a silently-ignored flag
# produces a run that looks successful while measuring the wrong config.
@dataclass(frozen=True)
class ServingConfig:
    # ── fixed within a study (recorded, not searched) ────────────
    #
    # Default is the small dense model on a cheap GPU: $1.10/hr against $3.95,
    # so harness iteration costs a third as much. Be clear about what that
    # trades away -- Qwen3-4B on an A10G is **prefill-bound** at our request
    # shapes, while Qwen3-30B-A3B on an H100 is **decode-bound**. Opposite sides
    # of the roofline. Use the small setup to develop and debug the harness; use
    # the 30B (or the 235B on 8xH100) whenever a serving-policy result is meant
    # to transfer.
    #
    # Two constraints picked the GPU, and both rule out the cheapest option:
    #
    #   * **FP8 needs SM89+.** A10G is SM86 (Ampere) with no native FP8, so an
    #     FP8 checkpoint cannot run there at all. L4/L40S (Ada) and H100
    #     (Hopper) can.
    #   * **KV cache dominates at long context.** Qwen3-4B costs 144 KiB/token
    #     of KV, so at a 40k-token agentic context an L4 holds only 3 concurrent
    #     sequences -- too few to study batching. L40S holds 7, H100 13.
    #
    #   Qwen/Qwen3-4B-Instruct-2507-FP8    L40S   $1.95/hr   262k ctx
    #   Qwen/Qwen3-8B-FP8                  L40S   $1.95/hr    41k ctx
    #   Qwen/Qwen3-30B-A3B-Instruct-2507-FP8  H100  $3.95/hr  MoE, decode-bound
    #   Qwen/Qwen3-235B-A22B-Instruct-2507-FP8  8xH100  $31.60/hr
    # Standard experiment environment, agreed 2026-09-01: ONE H100 at an
    # assumed $3.00/GPU-hr. Chosen over the cheaper A100 80GB because Ampere
    # (SM80) has no FP8 tensor cores and this checkpoint is block-quantised
    # FP8 -- it would dequantise to bf16, measuring a different machine.
    # Chosen over L40S because decode is bandwidth-bound: L40S is half the
    # hourly price and ~1.9x the cost per token, and holds only ~12
    # conversations of KV against the H100's ~30.
    model: str = "Qwen/Qwen3.8-27B-FP8"
    gpu: str = "H100"
    n_gpu: int = 1

    # ── parallelism ──────────────────────────────────────────────
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int | None = None          # expert parallelism; None = disabled

    # ── memory ───────────────────────────────────────────────────
    mem_fraction_static: float = 0.85   # VRAM share for weights + KV pool
    max_total_tokens: int | None = None

    # ── batching / scheduling (the interesting knobs) ────────────
    max_running_requests: int = 256
    chunked_prefill_size: int = 8192
    # Verified against sglang 0.5.18 --help by probe_env. Seven policies,
    # not the four originally assumed: lof, priority and routing-key are
    # extra search-space dimensions we would otherwise have missed.
    schedule_policy: str = "fcfs"
    # {lpm, random, fcfs, dfs-weight, lof, priority, routing-key}
    schedule_conservativeness: float = 1.0

    # ── caching ──────────────────────────────────────────────────
    enable_prefix_caching: bool = True

    # ── observability ────────────────────────────────────────────
    # Server-side histograms (TTFT, ITL, queue time, cache hit rate). These are
    # measured from request arrival inside the inference system, so they are
    # independent of where load is generated -- which is what makes an
    # agent-driven client on another machine comparable to the in-container one.
    enable_metrics: bool = True

    # ── misc ─────────────────────────────────────────────────────
    context_length: int | None = None
    extra_args: tuple[str, ...] = ()

    # What a candidate may not change: these name the machine being priced,
    # and `enable_metrics` is how the price is read. Everything else is the
    # serving stack and is the candidate's to set -- a KV layout that changes
    # what "static" memory means, or a prefill kernel that wants a different
    # chunk, is not a complete idea without its launch line.
    LOCKED = ("model", "gpu", "n_gpu", "enable_metrics")

    def with_overrides(self, overrides: dict) -> ServingConfig:
        """This config with a candidate's launch overrides applied.

        Unknown keys and locked keys are errors, not warnings: a typo that
        silently launched stock would produce a stock price under a
        candidate's name. `extra_args` are appended, not replaced, so the
        study's own extra flags survive.
        """
        from dataclasses import fields, replace

        known = {f.name for f in fields(self)}
        bad = sorted(k for k in overrides if k not in known)
        if bad:
            raise ValueError(f"unknown serving override(s): {bad}; "
                             f"known: {sorted(known - set(self.LOCKED))}")
        locked = sorted(k for k in overrides if k in self.LOCKED)
        if locked:
            raise ValueError(f"serving override(s) not allowed: {locked} "
                             "(they define the machine being priced)")
        d = dict(overrides)
        if "extra_args" in d:
            d["extra_args"] = tuple(self.extra_args) + tuple(d["extra_args"])
        if "ep_size" in d and not d["ep_size"]:
            d["ep_size"] = None
        return replace(self, **d)

    def to_sglang_args(self) -> list[str]:
        """Render as `sglang.launch_server` CLI arguments."""
        a = [
            "--model-path", self.model,
            "--tp-size", str(self.tp_size),
            "--dp-size", str(self.dp_size),
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--max-running-requests", str(self.max_running_requests),
            "--chunked-prefill-size", str(self.chunked_prefill_size),
            "--schedule-policy", self.schedule_policy,
            "--schedule-conservativeness", str(self.schedule_conservativeness),
        ]
        # 0 means "not requested". Emitting `--ep-size 0` is not the same as
        # omitting the flag -- SGLang rejects it -- and launch.py passes 0 as
        # its default. Expert parallelism is meaningless on a dense model
        # anyway (Qwen3.8-27B has `has_moe: false`).
        if self.ep_size:
            a += ["--ep-size", str(self.ep_size)]
        if self.max_total_tokens is not None:
            a += ["--max-total-tokens", str(self.max_total_tokens)]
        if self.context_length is not None:
            a += ["--context-length", str(self.context_length)]
        if not self.enable_prefix_caching:
            a += ["--disable-radix-cache"]
        if self.enable_metrics:
            a += ["--enable-metrics"]
        a += list(self.extra_args)
        return a

    def validate(self) -> list[str]:
        """Catch launch-time constraints before spending GPU minutes on them.

        Two 8xH100 attempts died in the first minutes for reasons a check could
        have caught: TP=8 requested against a single allocated GPU, and an
        MoE/FP8 block-size constraint. Both were cheap only because the
        fast-fail path works; neither needed a GPU to detect.
        """
        from .specs import MODELS

        problems: list[str] = []
        if self.tp_size > self.n_gpu:
            problems.append(
                f"tp_size={self.tp_size} exceeds n_gpu={self.n_gpu}. n_gpu must "
                f"also be allocated at call time -- it does not allocate itself.")
        if self.ep_size and self.ep_size > self.n_gpu:
            problems.append(f"ep_size={self.ep_size} exceeds n_gpu={self.n_gpu}")

        spec = MODELS.get(self.model)
        # SGLang applies its quantized-MoE block check to some dense models
        # too, with a fallback intermediate size of its own. Qwen3.8-27B-FP8
        # died 128s into a run this way: the model has no experts, so the
        # `n_experts > 1` gate below skipped the check entirely, and SGLang
        # then rejected `tp=8, ep=1` because (512 / 8) % 128 != 0.
        if spec is not None and spec.sglang_moe_check_intermediate:
            block, sg = 128, spec.sglang_moe_check_intermediate
            moe_tp = max(1, self.tp_size // max(1, self.ep_size or 1))
            if (sg / moe_tp) % block != 0:
                ok = [e for e in (1, 2, 4, 8, 16)
                      if e <= self.n_gpu
                      and (sg / max(1, self.tp_size // e)) % block == 0]
                problems.append(
                    f"SGLang runs its quantized-MoE block check on this model "
                    f"even though it is dense: {sg} / moe_tp={moe_tp} = "
                    f"{sg / moe_tp:.0f}, not a multiple of {block}. "
                    f"Set ep_size to one of {ok or 'none -- change tp_size'}.")
        if spec is not None and spec.n_experts > 1:
            # FP8 weights are stored in 128x128 blocks, so each rank's slice of
            # an expert must be a whole number of blocks.
            block = 128
            moe_tp = max(1, self.tp_size // max(1, self.ep_size or 1))
            per_rank = spec.moe_intermediate / moe_tp
            if per_rank % block != 0:
                ok = [e for e in (1, 2, 4, 8, 16)
                      if e <= self.n_gpu
                      and (spec.moe_intermediate /
                           max(1, self.tp_size // e)) % block == 0]
                problems.append(
                    f"MoE/FP8 block constraint: moe_intermediate="
                    f"{spec.moe_intermediate} / moe_tp={moe_tp} = {per_rank:.0f}, "
                    f"not a multiple of {block}. Valid ep_size for tp_size="
                    f"{self.tp_size}: {ok or 'none -- change tp_size'}")
        return problems

    def digest(self) -> str:
        return _digest(asdict(self))






def _digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
