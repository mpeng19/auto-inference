"""arXiv as a producer of idea records.

The API is plain Atom over HTTP, so the fetch is stdlib. Turning an abstract
into an `IdeaRecord` -- naming the mechanism, the SGLang files it would touch,
the expected gain and the risk -- is a model's job, and is injected as a
callable so the parsing and the record shape are testable offline.

Queries are kept in one place because they are the taste of the bank: what
we search for is what the fleet will build.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

from ..contracts.ideas import IdeaRecord
from .sqlite import record_from_dict

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

# What we are looking for. Each is one arXiv query; results are merged.
DEFAULT_QUERIES = (
    "LLM inference decode attention kernel memory bandwidth",
    "KV cache compression quantization LLM inference serving",
    "speculative decoding draft model acceptance serving throughput",
    "fused kernel LLM inference latency",
    "paged attention KV cache layout serving",
    "LLM serving scheduling prefill decode disaggregation",
)


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    published: str
    url: str


def fetch(query: str, max_results: int = 25, timeout: int = 60) -> tuple[Paper, ...]:
    q = urllib.parse.urlencode({
        "search_query": "all:" + query, "start": 0, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending"})
    with urllib.request.urlopen(f"{API}?{q}", timeout=timeout) as r:
        return parse_atom(r.read().decode("utf-8", errors="replace"))


def parse_atom(xml_text: str) -> tuple[Paper, ...]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        ident = (e.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]
        ident = re.sub(r"v\d+$", "", ident)
        out.append(Paper(
            arxiv_id=ident,
            title=" ".join((e.findtext("a:title", "", NS) or "").split()),
            abstract=" ".join((e.findtext("a:summary", "", NS) or "").split()),
            published=(e.findtext("a:published", "", NS) or "")[:10],
            url=f"https://arxiv.org/abs/{ident}"))
    return tuple(out)


_RECORD_PROMPT = """You are curating an idea bank for an auto-research loop that
lowers the cost per output token of SGLang 0.5.18 serving Qwen3.8-27B-FP8 on one
H100 under a latency SLO (p90 TTFT 2818 ms, p90 TPOT 25 ms, mean TPOT 20 ms).
The decode-time per-sequence KV read runs at only 22-28% of memory bandwidth and
does not amortise with batch; that is the largest lever. Stock SGLang already has
radix prefix caching, chunked prefill, continuous batching, paged KV, FlashInfer
attention, CUDA graphs and FP8 GEMM.

Paper: {title} ({arxiv_id}, {published})
Abstract: {abstract}

If this paper contains a mechanism an engineer could implement inside the
sglang package (Python + Triton, or a CUDA extension; no new pip dependencies)
that plausibly lowers cost per output token under that SLO, reply with ONE JSON
object and nothing else, keys: title, mechanism (one paragraph, concrete),
hypothesis ("X will lower cost per output token because Y"), scale (one of
kernel|architecture|memory|scheduler|parallelism|numerics|other), targets
(list of sglang paths under srt/, e.g. "srt/layers/attention/triton_backend.py"),
expected_gain (with the paper's numbers), risks, prerequisites, tags (3-6 words).
If it is not implementable here, or is already stock behaviour, reply with the
single word NO."""


def to_record(paper: Paper, ask: Callable[[str], str]) -> IdeaRecord | None:
    """One paper -> one record, or None. `ask` is the model call."""
    text = ask(_RECORD_PROMPT.format(
        title=paper.title, arxiv_id=paper.arxiv_id, published=paper.published,
        abstract=paper.abstract[:4000])).strip()
    if not text or text.upper().startswith("NO"):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    d.setdefault("title", paper.title)
    d["source"] = f"arxiv:{paper.arxiv_id}"
    d["source_title"] = paper.title
    d["url"] = paper.url
    return record_from_dict(d)


def harvest(bank, ask: Callable[[str], str], queries=DEFAULT_QUERIES,
            per_query: int = 25, fetcher: Callable[[str, int], tuple[Paper, ...]] | None = None,
            known_sources: set[str] | None = None) -> tuple[int, int]:
    """Fetch, dedupe against the bank, ask the model, add. Returns
    (papers seen, records added)."""
    fetcher = fetcher or (lambda q, n: fetch(q, n))
    known = known_sources if known_sources is not None else {
        r.source for r in bank.list()}
    seen: dict[str, Paper] = {}
    for q in queries:
        for p in fetcher(q, per_query):
            seen.setdefault(p.arxiv_id, p)
    added = 0
    for p in seen.values():
        if f"arxiv:{p.arxiv_id}" in known:
            continue
        rec = to_record(p, ask)
        if rec is not None:
            bank.add(rec)
            known.add(rec.source)
            added += 1
    return len(seen), added
