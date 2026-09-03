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
mechanism imported twice is one record. Similarity is lexical (Jaccard over
word sets for claims and neighbours, FTS5 BM25 for search): the bank is tens
to hundreds of records and every decision has to be explainable in the
timeline; an embedding index buys nothing at this size and would add a
model call to every claim.

Producers: `pdf.harvest` (a book, windowed, via a model), `arxiv.harvest`
(the feed, relevance-sorted, via a model), `SqliteIdeaBank.import_jsonl`
(anything an extractor wrote). The CLI is `harness ideas ...`.
"""
from .sqlite import SEEDS, SqliteIdeaBank, default_bank_path, record_from_dict

__all__ = ["SEEDS", "SqliteIdeaBank", "default_bank_path", "record_from_dict"]
