"""The idea bank: records of the right size, claimed one per agent."""
import json

import pytest

from harness.contracts import IdeaBankService, IdeaRecord
from harness.ideas import SqliteIdeaBank, record_from_dict
from harness.ideas import arxiv as ax
from harness.ideas import pdf as bk


@pytest.fixture
def bank(tmp_path):
    return SqliteIdeaBank(tmp_path / "ideas.db")


def _rec(title, mech, scale="kernel", **kw):
    return IdeaRecord(title=title, mechanism=mech, hypothesis=f"{title} lowers cost",
                      scale=scale, **kw)


def test_satisfies_the_contract(bank):
    assert isinstance(bank, IdeaBankService)


def test_add_list_search_roundtrip(bank):
    r = _rec("split-KV decode attention", "online softmax across KV chunks per FlashDecoding",
             targets=("srt/layers/attention/triton_backend.py",), tags=("attention", "decode"))
    bank.add(r)
    got = bank.get(r.id)
    assert got == r
    assert bank.list()[0].targets == ("srt/layers/attention/triton_backend.py",)
    assert bank.search("softmax chunks")[0].id == r.id
    assert bank.count() == 1 and bank.count("claimed") == 0


def test_claim_is_exclusive_and_picks_the_least_similar(bank):
    """Two agents on the same mechanism was the failure mode of night-5:
    a00 and a02 both widened the same LPM queue cutoff."""
    bank.add(_rec("widen LPM queue cutoff", "raise the queue length at which LPM scheduling is disabled",
                  scale="scheduler"))
    bank.add(_rec("KV cache int8 quantisation", "store K and V in int8 with per-head scales",
                  scale="memory"))
    bank.add(_rec("fused decode attention kernel", "fuse QK, softmax and PV in one Triton kernel",
                  scale="kernel"))
    live = "raise the LPM queue length cutoff so prefix scheduling stays on under load"
    first = bank.claim("a00", avoid=(live,), live_scales=("scheduler",))
    assert first is not None and "LPM" not in first.title
    assert first.status == "claimed" and first.claimed_by == "a00"
    second = bank.claim("a01", avoid=(live, first.text), live_scales=("scheduler", first.scale))
    assert second is not None and second.id != first.id
    third = bank.claim("a02", avoid=(live,))
    assert third is not None and third.id not in (first.id, second.id)
    assert bank.claim("a03") is None                # empty
    bank.record_outcome(first.id, "exp_1")
    assert bank.get(first.id).status == "tried"
    assert bank.get(first.id).experiment_ids == ("exp_1",)
    assert bank.get(first.id).claimed_by == ""
    bank.release(second.id)
    assert bank.get(second.id).status == "available"


def test_as_idea_points_back_at_the_bank(bank):
    r = _rec("speculative decoding with n-gram draft", "draft from prompt n-grams, verify in one step",
             targets=("srt/speculative/ngram_worker.py",))
    idea = r.as_idea()
    assert idea.seeded_by == r.id and idea.targets == r.targets
    assert idea.hypothesis == r.hypothesis


def test_import_jsonl_tolerates_drift(bank, tmp_path):
    p = tmp_path / "book.jsonl"
    p.write_text("\n".join([
        json.dumps({"title": "A", "mechanism": "m", "hypothesis": "h", "scale": "kernel",
                    "targets": ["srt/x.py"], "tags": ["a", "b"], "bogus": 1}),
        json.dumps({"title": "B", "mechanism": "m2", "scale": "not-a-scale",
                    "targets": "srt/y.py, srt/z.py"}),
        "",
    ]))
    assert bank.import_jsonl(p, source_default="book") == 2
    a, b = bank.list()
    assert a.targets == ("srt/x.py",) and a.tags == ("a", "b") and a.source == "book"
    assert b.scale == "other" and b.targets == ("srt/y.py", "srt/z.py")


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2309.06180v2</id>
    <title>Efficient Memory Management for Large Language Model
      Serving with PagedAttention</title>
    <summary>We propose PagedAttention, an attention algorithm inspired by
      virtual memory.</summary>
    <published>2023-09-12T00:00:00Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Split-K decode</title>
    <summary>Online softmax across KV chunks.</summary>
    <published>2024-01-01T00:00:00Z</published>
  </entry>
</feed>"""


def test_arxiv_parse_and_record(bank):
    papers = ax.parse_atom(ATOM)
    assert [p.arxiv_id for p in papers] == ["2309.06180", "2401.00001"]
    assert papers[0].title.startswith("Efficient Memory") and "  " not in papers[0].title

    def ask(prompt):
        if "PagedAttention" in prompt:
            return "NO"                            # already stock
        return json.dumps({"title": "Split-KV decode attention", "mechanism": "chunks",
                           "hypothesis": "split-KV lowers cost because bandwidth",
                           "scale": "kernel", "targets": ["srt/layers/attention/triton_backend.py"],
                           "expected_gain": "1.5x decode at long context",
                           "risks": "numerics", "tags": ["attention"]})

    seen, added = ax.harvest(bank, ask, queries=("q1", "q2"),
                             fetcher=lambda q, n: papers)
    assert seen == 2 and added == 1
    rec = bank.list()[0]
    assert rec.source == "arxiv:2401.00001" and rec.url.endswith("2401.00001")
    # a second harvest does not ask again about what the bank already holds
    calls = []
    ax.harvest(bank, lambda p: calls.append(p) or "NO", queries=("q1",),
               fetcher=lambda q, n: papers)
    assert len(calls) == 1                         # only the paper we said NO to


def test_book_windows_and_harvest(bank):
    assert list(bk.windows(45, 20)) == [(1, 20), (21, 40), (41, 45)]

    def reader(pdf, first, last):
        return "" if first > 40 else f"pages {first}-{last} about attention kernels"

    def ask(prompt):
        assert "TEXT:" in prompt
        first = int(prompt.split("pages ")[1].split("-")[0])
        return json.dumps([{"title": f"idea from {first}", "mechanism": "m",
                            "hypothesis": "h", "scale": "kernel",
                            "targets": ["srt/layers/attention/triton_backend.py"]}])

    n = bk.harvest(bank, "book.pdf", ask, book="IE", reader=reader, n_pages=45)
    assert n == 2
    srcs = sorted(r.source for r in bank.list())
    assert srcs == ["book:p.1-20", "book:p.21-40"]
    assert bk.parse_records("not json", 1, 2, "IE") == []


def test_record_from_dict_defaults():
    r = record_from_dict({"title": "t"})
    assert r.scale == "kernel" and r.status == "available" and r.targets == ()


def test_record_from_dict_flattens_lists_in_text_fields(bank):
    """The arXiv harvest died on its 22nd paper: the model returned `risks`
    as a list and SQLite refused to bind it."""
    r = record_from_dict({"title": "t", "risks": ["numerics", "accuracy"],
                          "prerequisites": {"needs": "triton 3"}})
    assert r.risks == "numerics; accuracy" and r.prerequisites == "needs: triton 3"
    bank.add(r)
    assert bank.get(r.id).risks == "numerics; accuracy"


def test_ids_are_content_addressed_so_reimport_is_one_record(bank, tmp_path):
    from harness.contracts.ideas import content_id
    from harness.ideas import record_from_dict

    a = record_from_dict({"title": "Paged  attention", "mechanism": "read pages"})
    b = record_from_dict({"title": "paged attention", "mechanism": "read  pages"})
    assert a.id == b.id == content_id("Paged attention", "read pages")
    assert a.id.startswith("idea_") and len(a.id) == 17
    p = tmp_path / "x.jsonl"
    p.write_text('{"title": "T", "mechanism": "M"}\n')
    bank.import_jsonl(p)
    bank.import_jsonl(p)
    assert bank.count() == 1


def test_the_packaged_seed_set_loads_idempotently(bank):
    n = bank.seed("book")
    assert n == 27 and bank.count() == 27
    assert bank.seed("book") == 27 and bank.count() == 27
    assert all(r.source.startswith("book") for r in bank.list())
    import pytest
    with pytest.raises(FileNotFoundError, match="no seed set"):
        bank.seed("nope")


def test_related_are_the_nearest_by_text(bank):
    from harness.contracts.ideas import IdeaRecord
    a = bank.add(IdeaRecord(title="paged attention kernel", mechanism="reads kv pages by block table"))
    b = bank.add(IdeaRecord(title="sparse kv pages", mechanism="reads a subset of kv pages by score"))
    c = bank.add(IdeaRecord(title="speculative decoding", mechanism="draft model proposes tokens"))
    rel = bank.related(a, k=2)
    assert rel[0].id == b
    assert c not in [r.id for r in rel]
    assert bank.related("missing") == ()


def test_a_seeded_claim_steers_toward_the_seed_but_not_into_what_is_live(bank):
    from harness.contracts.ideas import IdeaRecord
    live = "sparse kv pages: reads a subset of kv pages by attention score bounds"
    bank.add(IdeaRecord(title="sparse kv pages twin", mechanism="reads a subset of kv pages by attention score bounds"))
    kv4 = bank.add(IdeaRecord(title="int4 kv cache", mechanism="store kv values in four bits with per-block scales"))
    bank.add(IdeaRecord(title="speculative decoding", mechanism="draft model proposes tokens verified in one step"))
    got = bank.claim("a01", avoid=(live,), seed="quantise the kv cache to fewer bits")
    assert got.id == kv4
    assert got.status == "claimed" and got.claimed_by == "a01"
    # the twin of what is live is never handed back, however well the seed matches it
    got2 = bank.claim("a02", avoid=(live,), seed="sparse kv pages by attention score")
    assert got2 is not None and "twin" not in got2.title


def test_reseeding_an_old_bank_migrates_ids_instead_of_duplicating(bank):
    """Banks filled before ids were content hashes hold random `bank_` ids;
    seeding them again must not double every record."""
    from harness.contracts.ideas import IdeaRecord
    old = IdeaRecord(id="bank_deadbeef0000", title="Pre-RoPE key storage for position-independent KV reuse",
                     mechanism="old text", status="tried", experiment_ids=("exp_1",))
    bank.add(old)
    n = bank.seed("book")
    assert n == 27 and bank.count() == 27
    assert bank.get("bank_deadbeef0000") is None
    same = [r for r in bank.list() if r.title == old.title]
    assert len(same) == 1 and same[0].status == "tried" and same[0].experiment_ids == ("exp_1",)
    assert same[0].id.startswith("idea_")
