"""Facts learned across runs: contradictions supersede, nothing is lost."""
from harness.contracts import Fact, SkillBankService
from harness.skills import SqliteSkillBank, lexical_judge


def test_contract_and_roundtrip(tmp_path):
    b = SqliteSkillBank(tmp_path / "skills.db")
    assert isinstance(b, SkillBankService)
    f = Fact(claim="FP8 greedy decoding is not deterministic across batch compositions",
             topic="gsm8k-gate", evidence="stock scored 62% and 70% on the same 50 items",
             source="night-1", confidence=0.9, tags=("numerics", "gate"))
    fid, losers = b.add(f)
    assert fid == f.id and losers == ()
    assert b.get(fid) == f and b.list(topic="gsm8k-gate") == (f,)
    assert b.search("deterministic batch")[0].id == fid


def test_a_contradicting_fact_supersedes_the_old_one_and_keeps_it(tmp_path):
    b = SqliteSkillBank(tmp_path / "skills.db")
    old = Fact(claim="a 2 point GSM8K tolerance is enough", topic="gsm8k-gate", source="aug")
    b.add(old)
    new = Fact(claim="a 2 point GSM8K tolerance rejects stock; use 10 points on 100 items",
               topic="gsm8k-gate", source="night-1")
    # the judge decides what contradicts; here a model stand-in names the loser
    _fid, losers = b.add(new, judge=lambda n, existing: tuple(x.id for x in existing))
    assert losers == (old.id,)
    assert b.get(old.id).status == "superseded" and b.get(old.id).superseded_by == new.id
    assert b.list(topic="gsm8k-gate") == (new,)            # active only
    assert b.list(topic="gsm8k-gate", status=None) and len(b.list(status=None)) == 2
    # a fact on another topic is never judged against it
    other = Fact(claim="chunked prefill at 8192 is not binding at N<=24", topic="chunked-prefill")
    assert b.add(other, judge=lambda n, e: tuple(x.id for x in e))[1] == ()
    rendered = b.render()
    assert "## gsm8k-gate" in rendered and "rejects stock" in rendered
    assert "2 point GSM8K tolerance is enough" not in rendered
    assert rendered.startswith("---\nname: serving-facts")
    b.retract(other.id)
    assert b.list(topic="chunked-prefill") == ()


def test_lexical_judge_is_the_crude_fallback():
    a = Fact(claim="KV read runs at 25 percent of bandwidth on H100", topic="kv")
    b = Fact(claim="the KV read runs at about 25 percent of H100 bandwidth", topic="kv")
    c = Fact(claim="prefix cache hit rate is 0.7 under market load", topic="kv")
    assert lexical_judge(a, (b, c)) == (b.id,)
