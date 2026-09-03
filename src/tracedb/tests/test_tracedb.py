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
    """Every `attn_qkv` span (CPU op and its kernel) is followed by a
    `kernel_attn_core`, so the query must pair all of them but the last
    kernel of the trace, which has no successor.

    Pinned with names the fixture actually contains: this test used to ask
    for `fwd_launch*`, which matches nothing, and passed by asserting nothing.
    """
    out = Q.between(store, "attn_qkv", "kernel_attn_core")
    assert out, "no pairs found"
    head, rows = out[0]["summary"], out[1:]
    assert head["instances"] == 39                    # 20 steps x 2 spans, minus the last kernel
    assert all(r["latency_us"] >= 0 for r in rows)
    assert rows == sorted(rows, key=lambda r: -r["latency_us"])


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


def test_step_diff_blames_the_slow_step_on_the_planted_stall(store):
    """The MCP tool an agent reaches for after `steps` names an outlier: it has
    to say *which op* was slow, not just that the step was."""
    got = Q.step_diff(store, 4)          # every 5th step carries the 3 ms stall
    assert got["step_dur_us"] > 1.5 * got["median_step_dur_us"]
    worst = got["top_regressions"][0]
    assert worst["op"] == "dataloader_next"
    assert worst["delta_us"] > 2000      # the planted 3 ms, minus jitter
    # And a step without the stall is unremarkable against the median.
    assert Q.step_diff(store, 3)["top_regressions"][0]["delta_us"] < 500


def test_step_diff_refuses_a_step_that_is_not_there(store):
    assert Q.step_diff(store, 999) == {"error": "no step 999"}


def test_gpu_idle_blames_the_gap_on_what_the_cpu_was_doing(store):
    """'Why is the GPU idle here' is the question; the answer has to name a CPU
    op, or it is just a list of holes."""
    got = Q.gpu_idle(store, min_gap_us=200)
    assert got["gaps"] > 0 and got["idle_total_us"] > 0
    assert got["blame_by_op"][0]["op"] == "dataloader_next"
    biggest = got["largest"][0]
    assert biggest["gap_us"] > 1000
    assert biggest["cpu_during_gap"][0]["op"] == "dataloader_next"
    # Sorted largest first, and the profiler's own markers never take the blame.
    gaps = [r["gap_us"] for r in got["largest"]]
    assert gaps == sorted(gaps, reverse=True)
    assert not any(r["op"].startswith("ProfilerStep")
                   for r in got["blame_by_op"])


def test_a_threshold_above_every_gap_finds_none(store):
    assert Q.gpu_idle(store, min_gap_us=10_000_000)["gaps"] == 0


def test_launches_says_so_when_the_trace_carries_no_correlation_ids(store):
    """The synthetic trace has none. Reporting zero pairs beats reporting a
    launch latency computed from nothing."""
    got = Q.launches(store)
    assert got["pairs"] == 0 and "correlation" in got["note"]


def test_launches_pairs_a_runtime_span_with_its_kernel(tmp_path):
    """CPU->GPU delay is recovered through kineto's correlation id, across
    tracks: the launch is on a CPU thread and the kernel on a stream.

    The `cpu_op` span is not decoration -- it is what marks the launching
    thread as a CPU track, and `launches` only pairs a non-GPU track with a GPU
    one. A real torch trace opens its CPU thread the same way.
    """
    import json

    rows = [{"ph": "X", "pid": 1, "tid": "cpu", "name": "aten::mm",
             "cat": "cpu_op", "ts": 0, "dur": 1}]
    for i, (launch_ts, kernel_ts) in enumerate([(10, 40), (110, 410)], start=1):
        rows.append({"ph": "X", "pid": 1, "tid": "cpu", "name": "cudaLaunchKernel",
                     "cat": "cuda_runtime", "ts": launch_ts, "dur": 5,
                     "args": {"correlation": i}})
        rows.append({"ph": "X", "pid": 1, "tid": "stream 7", "name": "gemm_kernel",
                     "cat": "kernel", "ts": kernel_ts, "dur": 20,
                     "args": {"correlation": i}})
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ingest(p, tmp_path / "t.sqlite")

    got = Q.launches(TraceStore(tmp_path / "t.sqlite"))
    assert got["pairs"] == 2
    assert got["lat_max_us"] == 295.0          # 410 - 110 - 5
    assert got["slowest"][0]["kernel"] == "gemm_kernel"
    assert got["slowest"][0]["launch"] == "cudaLaunchKernel"


def test_a_pattern_with_a_wildcard_anchors_and_one_without_does_not(store):
    """Worth pinning, because the two behave oppositely. `*` becomes SQL's `%`
    and is used as written, so `attn*` is a *prefix* match. A pattern carrying
    no wildcard is wrapped in them instead, so `attn_out` is a substring match
    and finds `kernel_attn_out` as well."""
    assert set(Q.name_ids(store, "attn*").values()) == {
        "attn_qkv", "attn_core", "attn_out", "attn_bwd"}
    assert set(Q.name_ids(store, "attn_out").values()) == {
        "attn_out", "kernel_attn_out"}
    assert Q.name_ids(store, "no_such_op") == {}


def test_cli_ingests_then_answers_from_the_same_database(tmp_path, capsys):
    """One ingest, many cheap queries -- and `--db` has to reach both halves."""
    import json

    from tracedb.cli import main

    generate(tmp_path / "t.json", steps=4)
    db = str(tmp_path / "t.sqlite")
    main(["--db", db, "ingest", str(tmp_path / "t.json")])
    assert json.loads(capsys.readouterr().out)["steps"] == 4

    main(["--db", db, "summary"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["meta"]["steps"] == 4 and summary["tracks"]

    main(["--db", db, "gaps", "attn_out", "mlp_in", "--min-gap", "100"])
    assert json.loads(capsys.readouterr().out)[0]["summary"]["instances"] >= 4


def test_gzipped_kineto_trace_ingests_like_the_plain_one(tmp_path):
    """The server writes `*.trace.json.gz`; build-4 ingested none of them
    because the reader opened the bytes as text."""
    import gzip
    import json

    from tracedb.ingest import ingest

    events = {"traceEvents": [
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 7, "args": {"name": "stream 7"}},
        {"ph": "X", "name": "ProfilerStep#3", "cat": "user_annotation", "pid": 1, "tid": 1,
         "ts": 0.0, "dur": 100.0, "args": {}},
        {"ph": "X", "name": "flash_fwd_kernel", "cat": "kernel", "pid": 1, "tid": 7,
         "ts": 10.0, "dur": 40.0, "args": {"correlation": 5}},
    ]}
    gz = tmp_path / "x-TP-0-DECODE.trace.json.gz"
    with gzip.open(gz, "wt") as f:
        json.dump(events, f)
    out = ingest(gz, tmp_path / "t.sqlite")
    assert out["events"] == 2 and out["steps"] == 1


def _sglang_like_trace(path):
    """A capture shaped like SGLang's: `step[...]` annotations instead of
    ProfilerStep, a Python stack frame spanning everything, and kernels."""
    import json
    ev = []
    ev.append({"ph": "X", "name": "threading.py(1030): _bootstrap", "cat": "python_function",
               "pid": 1, "tid": 1, "ts": 0.0, "dur": 10_000.0, "args": {}})
    for i, (name, ts) in enumerate([("step[EXTEND bs=3 toks=8192]", 100.0),
                                    ("step[DECODE bs=12]", 2100.0),
                                    ("step[DECODE bs=12]", 4100.0)]):
        ev.append({"ph": "X", "name": name, "cat": "user_annotation", "pid": 1, "tid": 1,
                   "ts": ts, "dur": 1500.0, "args": {}})
        ev.append({"ph": "X", "name": "scheduler.run_batch", "cat": "cpu_op", "pid": 1, "tid": 1,
                   "ts": ts, "dur": 300.0, "args": {}})
        ev.append({"ph": "X", "name": "flash_fwd_kernel", "cat": "kernel", "pid": 1, "tid": 7,
                   "ts": ts + 400.0, "dur": 900.0, "args": {"correlation": i}})
    path.write_text(json.dumps({"traceEvents": ev}))


def test_sglang_step_annotations_become_numbered_steps(tmp_path):
    from tracedb.ingest import ingest
    from tracedb.store import TraceStore

    tr = tmp_path / "t.json"; _sglang_like_trace(tr)
    out = ingest(tr, tmp_path / "t.sqlite")
    assert out["steps"] == 3
    st = TraceStore(tmp_path / "t.sqlite")
    assert [r[0] for r in st.conn.execute("SELECT idx FROM steps ORDER BY ts")] == [0, 1, 2]


def test_slowest_means_kernels_not_the_interpreter(tmp_path):
    from tracedb import query as Q
    from tracedb.ingest import ingest
    from tracedb.store import TraceStore

    tr = tmp_path / "t.json"; _sglang_like_trace(tr)
    ingest(tr, tmp_path / "t.sqlite")
    st = TraceStore(tmp_path / "t.sqlite")
    top = Q.slowest(st, k=3)
    assert top and all(r["kind"] == "gpu" for r in top)
    assert Q.slowest(st, k=1, kind="")[0]["name"].startswith("threading.py")


def test_idle_blame_skips_python_stack_frames(tmp_path):
    from tracedb import query as Q
    from tracedb.ingest import ingest
    from tracedb.store import TraceStore

    tr = tmp_path / "t.json"; _sglang_like_trace(tr)
    ingest(tr, tmp_path / "t.sqlite")
    st = TraceStore(tmp_path / "t.sqlite")
    idle = Q.gpu_idle(st, min_gap_us=100)
    assert idle["gaps"] >= 1
    assert not any(".py" in b["op"] for b in idle["blame_by_op"])


def test_gpu_idle_is_the_union_across_streams(tmp_path):
    """Two streams, each busy half the time, alternating: the GPU is never
    idle. Per-stream accounting said it was idle 100%."""
    import json

    from tracedb import query as Q
    from tracedb.ingest import ingest
    from tracedb.store import TraceStore

    ev = []
    for i in range(10):
        ev.append({"ph": "X", "name": "k_a", "cat": "kernel", "pid": 1, "tid": 7,
                   "ts": i * 200.0, "dur": 100.0, "args": {}})
        ev.append({"ph": "X", "name": "k_b", "cat": "kernel", "pid": 1, "tid": 8,
                   "ts": i * 200.0 + 100.0, "dur": 100.0, "args": {}})
    tr = tmp_path / "t.json"; tr.write_text(json.dumps({"traceEvents": ev}))
    ingest(tr, tmp_path / "t.sqlite")
    idle = Q.gpu_idle(TraceStore(tmp_path / "t.sqlite"), min_gap_us=10)
    assert idle["gaps"] == 0 and idle["idle_total_us"] == 0
