"""Memory: the read is the hard part, so that is what these check."""
from harness.contracts import (
    Experiment,
    Finding,
    MemoryService,
    Provenance,
    Recall,
    Relation,
)


def _seed(m):
    a = Experiment(agent_id="a01", idea_id="i1", verdict="loss",
                   hypothesis="raise chunked_prefill_size to 16384",
                   summary="TTFT rose 40%, no throughput gain")
    b = Experiment(agent_id="a02", idea_id="i2", verdict="win",
                   hypothesis="lpm schedule policy for prefix reuse",
                   summary="hit 0.75 -> 0.81, bill -6%")
    c = Experiment(agent_id="a01", idea_id="i1", verdict="pending",
                   hypothesis="chunked prefill 12288 as a compromise")
    for e in (a, b, c):
        m.record(e)
    m.relate(Relation(src=a.id, dst=c.id, kind="derived_from", note="narrowing"))
    return a, b, c


def test_satisfies_the_contract(memory):
    assert isinstance(memory, MemoryService)


def test_recall_surfaces_the_failure_you_are_about_to_repeat(memory):
    """The expensive knowledge in a research loop is what already failed."""
    a, _, _ = _seed(memory)
    br = memory.recall(Recall(intent="I want to raise chunked prefill size",
                              agent_id="a03"))
    assert "did NOT work" in br.text
    assert any(h.experiment.id == a.id for h in br.hits)


def test_graph_neighbours_surface_without_sharing_vocabulary(memory):
    """The point of the edges: relevance that keyword search cannot see."""
    _, _, c = _seed(memory)
    memory.record(Experiment(id="exp_zzz", agent_id="a09", idea_id="i9",
                             hypothesis="totally unrelated wording",
                             verdict="loss", summary="also failed"))
    memory.relate(Relation(src=c.id, dst="exp_zzz", kind="contradicts"))
    br = memory.recall(Recall(intent="anything", agent_id="a01", idea_id="i1"))
    ids = {h.experiment.id for h in br.hits}
    assert "exp_zzz" in ids, "a 2-edge neighbour of my own work must surface"


def test_lineage_is_deduplicated_and_ordered(memory):
    a, _, c = _seed(memory)
    memory.relate(Relation(src=a.id, dst=c.id, kind="replicates"))
    lin = memory.lineage(c.id)
    assert [e.id for e in lin].count(a.id) == 1


def test_recall_returns_a_brief_not_a_result_list(memory):
    """`agent-db` measured retrieved-facts-in-context as a clean null
    out-of-sample; the condition that separated was a synthesised brief."""
    _seed(memory)
    memory.assert_finding(Finding(
        claim="chunked_prefill_size above 8192 costs TTFT with no gain",
        kind="negative", confidence=0.7))
    br = memory.recall(Recall(intent="chunked prefill", agent_id="a05"))
    assert br.text.startswith("State of knowledge")
    assert "Established:" in br.text


def test_brief_respects_the_reader_token_budget(memory):
    for i in range(40):
        memory.record(Experiment(hypothesis=f"idea {i} about prefill batching",
                                 verdict="loss", summary="x" * 400))
    br = memory.recall(Recall(intent="prefill batching", k=40, max_tokens=200))
    assert br.est_tokens <= 200


def test_empty_memory_says_so_rather_than_returning_nothing(memory):
    br = memory.recall(Recall(intent="anything at all"))
    assert "Nothing on record" in br.text


def test_stale_findings_are_pruned_by_provenance(memory):
    """A store full of un-invalidatable facts poisons every agent reading it."""
    f = Finding(claim="holds for stack abc", confidence=0.9,
                provenance=Provenance(stack_digest="abc"))
    memory.assert_finding(f)
    assert memory.prune_stale(current_stack="abc") == 0
    assert memory.prune_stale(current_stack="def") == 1
    br = memory.recall(Recall(intent="holds"))
    assert not [x for x in br.findings if x.id == f.id]


def test_negative_results_can_be_excluded_but_are_boosted_by_default(memory):
    _seed(memory)
    with_neg = memory.recall(Recall(intent="chunked prefill", include_negative=True))
    without = memory.recall(Recall(intent="chunked prefill", include_negative=False))
    assert len(with_neg.hits) > len(without.hits)
