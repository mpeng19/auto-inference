"""A book as a producer of idea records.

Text comes out of the PDF in page windows (pypdf), each window goes to the
model with the same brief the arXiv path uses, and the JSON that comes back
becomes records. The model call and the text extractor are both injected so
the chunking and the record shape are testable without a PDF or a model.

A window is ~20 pages: long enough that a mechanism spread over a section
lands in one call, short enough that the model does not summarise the book
instead of extracting from it.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable, Iterable

from ..contracts.ideas import IdeaRecord
from .sqlite import record_from_dict

_CHUNK_PROMPT = """You are curating an idea bank for an auto-research loop that
lowers the cost per output token of SGLang 0.5.18 serving Qwen3.8-27B-FP8 on one
H100 under a latency SLO (p90 TTFT 2818 ms, p90 TPOT 25 ms, mean TPOT 20 ms).
The decode-time per-sequence KV read runs at only 22-28% of memory bandwidth and
does not amortise with batch; that is the largest lever. Stock SGLang already has
radix prefix caching, chunked prefill, continuous batching, paged KV, FlashInfer
attention, CUDA graphs and FP8 GEMM -- do not list those.

Below are pages {first}-{last} of "{book}". Extract every DISTINCT mechanism an
engineer could implement inside the sglang package (Python + Triton, or a CUDA
extension; no new pip dependencies) that plausibly lowers cost per output token
under that SLO. Prefer large ideas: kernel rewrites, fused operations, attention
algorithms, KV compression or layout, speculative decoding, memory systems,
scheduling architectures. Skip knob-tuning.

Reply with a JSON array (possibly empty) and nothing else. Each element has keys:
title, mechanism (one paragraph, concrete), hypothesis ("X will lower cost per
output token because Y"), source (like "book:p.{first}-{last}"), source_title
(section name if visible), scale (kernel|architecture|memory|scheduler|
parallelism|numerics|other), targets (list of sglang paths under srt/),
expected_gain (with the book's numbers, or "unknown"), risks, prerequisites,
tags (3-6 words).

TEXT:
{text}"""


def windows(n_pages: int, size: int = 20) -> Iterable[tuple[int, int]]:
    first = 1
    while first <= n_pages:
        yield first, min(n_pages, first + size - 1)
        first += size


def extract_text(pdf: str | pathlib.Path, first: int, last: int) -> str:
    """Pages `first..last` (1-based, inclusive) as text."""
    from pypdf import PdfReader

    r = PdfReader(str(pdf))
    return "\n".join((r.pages[i - 1].extract_text() or "")
                     for i in range(first, min(last, len(r.pages)) + 1))


def page_count(pdf: str | pathlib.Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def parse_records(text: str, first: int, last: int, book: str) -> list[IdeaRecord]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for d in items if isinstance(items, list) else []:
        if not isinstance(d, dict) or not d.get("title"):
            continue
        d.setdefault("source", f"book:p.{first}-{last}")
        d.setdefault("source_title", book)
        out.append(record_from_dict(d))
    return out


def harvest(bank, pdf: str | pathlib.Path, ask: Callable[[str], str],
            book: str = "", size: int = 20,
            reader: Callable[[str | pathlib.Path, int, int], str] = extract_text,
            n_pages: int | None = None,
            progress: Callable[[str], None] | None = None) -> int:
    """Every window of the book through the model; records into the bank."""
    n = n_pages if n_pages is not None else page_count(pdf)
    book = book or pathlib.Path(pdf).stem
    added = 0
    for first, last in windows(n, size):
        text = reader(pdf, first, last)
        if not text.strip():
            continue
        reply = ask(_CHUNK_PROMPT.format(first=first, last=last, book=book,
                                         text=text[:60000]))
        recs = parse_records(reply, first, last, book)
        for r in recs:
            bank.add(r)
        added += len(recs)
        if progress:
            progress(f"pages {first}-{last}: {len(recs)} records")
    return added
