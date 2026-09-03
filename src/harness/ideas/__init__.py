"""The idea bank: where the fleet's ideas come from, as a service.

Contract: `harness.contracts.ideas.IdeaBankService`, implemented by
`SqliteIdeaBank` (one SQLite file, default `~/.auto-inference/ideas.db`,
shared by every run on the machine; `HARNESS_HOME` moves it).

    bank = SqliteIdeaBank(default_bank_path())
    bank.seed("book")                       # the packaged 27 records, idempotent
    arxiv.harvest(bank, ask_with("opus"))   # more, from the feed, via a model
    rec = bank.claim("a01", avoid=(...))    # one exclusive record, least like what is live
    rec = bank.claim("a01", seed="fuse the KV read with ...")   # steered instead
    bank.related(rec.id)                    # its neighbours, any status
    bank.search("paged attention", k=8)     # full-text, BM25
    bank.record_outcome(rec.id, experiment_id, status="tried")
    bank.release(rec.id)                    # back to available (an error, not a result)

Ids are content-addressed (`contracts.ideas.content_id`): the same title and
mechanism imported twice is one record. Similarity is hybrid: Jaccard over
word sets and FTS5 BM25 for the lexical half, cosine over sentence
embeddings (`harness.embeddings`, all-MiniLM-L6-v2 on CPU) for the other, so
a paraphrase of a live idea is recognised as its twin. The model is
optional -- `uv sync --group embeddings` enables it, and the constructor's
`embedder=` overrides the default -- and without it every read is lexical
alone. The bank is tens to hundreds of records, so cosine is one matrix
product and every decision stays explainable in the timeline.

Producers: `pdf.harvest` (a book, windowed, via a model), `arxiv.harvest`
(the feed, relevance-sorted, via a model), `SqliteIdeaBank.import_jsonl`
(anything an extractor wrote). The CLI is `harness ideas ...`.
"""
from .sqlite import SEEDS, SqliteIdeaBank, default_bank_path, record_from_dict

__all__ = ["SEEDS", "SqliteIdeaBank", "default_bank_path", "record_from_dict"]
