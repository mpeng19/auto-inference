"""Ingest, query and render, against a deterministic synthetic trace.

The fixture is *generated*, not committed: `synth.py` plants two known
pathologies -- a ~300 us gap between `attn_out` and `mlp_in` every step, and a
3 ms dataloader stall every 5th step -- so the tests assert that the queries
find the things that were deliberately hidden, rather than that the output
matches a golden file nobody can reason about.
"""
import pytest

from tracedb import query as Q
from tracedb.ingest import ingest
from tracedb.render import timeline
from tracedb.store import TraceStore
from tracedb.synth import generate


@pytest.fixture
def store(tmp_path):
    generate(tmp_path / "t.json")
    ingest(tmp_path / "t.json", tmp_path / "t.sqlite")
    return TraceStore(tmp_path / "t.sqlite")


def test_ingest_reports_what_it_read(tmp_path):
    generate(tmp_path / "t.json", steps=5)
    got = ingest(tmp_path / "t.json", tmp_path / "t.sqlite")
    assert got["events"] > 100 and got["steps"] == 5 and got["span_us"] > 0


def test_gpu_and_cpu_tracks_are_separated(store):
    kinds = {t["name"]: t["kind"] for t in Q.summary(store)["tracks"]}
    assert kinds["CUDA stream 7 (compute)"] == "gpu"
    assert kinds["CUDA stream 20 (comm)"] == "gpu"


def test_it_finds_the_planted_gap(store):
    """The whole point: a pathology hidden in 600 events, found by a query."""
    out = Q.gaps(store, "attn_out", "mlp_in", 100)
    head = out[0]["summary"]
    assert head["instances"] >= 20
    assert 200 < head["gap_mean_us"] < 500


def test_a_gap_that_is_not_there_returns_nothing(store):
    assert Q.gaps(store, "attn_out", "mlp_in", 10_000) == []


def test_it_finds_the_planted_periodic_stall(store):
    """A 3 ms dataloader stall every 5th step shows up as step outliers."""
    s = Q.steps(store)
    assert s["steps"] == 20
    idxs = [o["idx"] for o in s["outliers"]]
    assert idxs, "the every-5th-step stall was not detected"
    assert all(i % 5 == 4 for i in idxs), idxs


def test_slowest_ranks_by_duration(store):
    rows = Q.slowest(store, "*", 5)
    durs = [r["dur"] for r in rows]
    assert durs == sorted(durs, reverse=True)


def test_grouping_collapses_numbered_variants(store):
    """20 `ProfilerStep#N` become one row; the family total is the signal.

    Two things this pins. `ops_grouped` is a **top-K view**, so it conserves
    counts only when the limit covers every family. And `canon_name` targets
    templated and numbered names -- `ProfilerStep#0..19`, `void cutlass::...<>`
    -- not arbitrary `_N` suffixes, which may be genuinely distinct ops.
    """
    raw = Q.ops(store, "ProfilerStep*", limit=10_000)
    grouped = Q.ops_grouped(store, "ProfilerStep*", limit=100)
    assert len({r["name"] for r in raw}) == 20, "each step is its own name"
    assert len(grouped) == 1, "and they collapse to one family"
    assert grouped[0]["name"] == "ProfilerStep#N"
    assert grouped[0]["cnt"] == sum(r["cnt"] for r in raw) == 20


def test_canon_name_targets_templates_and_indices():
    from tracedb.query import canon_name

    assert canon_name("void cutlass::Kernel<Gemm<float>>(Params)") == "cutlass::Kernel<>"
    assert canon_name("ProfilerStep#17") == "ProfilerStep#N"
    assert canon_name("ptr_0x7f3a01") == "ptr_0xN"
    # Not collapsed: a trailing index may be a genuinely distinct op.
    assert canon_name("nccl_allreduce_1") == "nccl_allreduce_1"


def test_overlap_measures_comm_against_compute(store):
    """The synthetic trace overlaps allreduce on the comm stream with backward
    kernels on the compute stream, which is the thing you profile to check."""
    ov = Q.overlap(store, "nccl_allreduce*", "kernel_*bwd*")
    assert ov["a_spans"] > 0 and ov["b_spans"] > 0
    assert 0 <= ov["overlap_us"] <= ov["a_total_us"] + 1
    assert ov["overlap_us"] > 0, "comm and compute should overlap here"


def test_between_reports_latency_from_a_to_the_next_b(store):
    out = Q.between(store, "fwd_launch*", "attn*")
    if out:
        assert out[0]["summary"]["latency_mean_us"] >= 0


def test_render_writes_a_window(store, tmp_path):
    got = timeline(store, 1000, 12000, str(tmp_path / "w.png"))
    assert (tmp_path / "w.png").is_file()
    assert got["tracks"] == 3 and got["spans_drawn"] > 0


def test_jsonl_events_ingest_too(tmp_path):
    """The other input shape: a stream of events rather than a chrome trace."""
    import json

    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in [
        {"ph": "X", "pid": 1, "tid": "s", "name": "k1", "cat": "kernel",
         "ts": 0, "dur": 10},
        {"ph": "X", "pid": 1, "tid": "s", "name": "k2", "cat": "kernel",
         "ts": 20, "dur": 5},
    ]))
    got = ingest(p, tmp_path / "t.sqlite")
    assert got["events"] == 2
    st = TraceStore(tmp_path / "t.sqlite")
    assert {r["name"] for r in Q.ops(st, "*")} == {"k1", "k2"}
