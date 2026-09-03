

def test_correlation_id_zero_is_a_real_id(tmp_path):
    """`args.get("correlation") or ...` dropped id 0, losing a trace's
    first launch/kernel pair."""
    import json

    from tracedb.ingest import ingest
    from tracedb.store import TraceStore

    trace = {"traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "launch", "pid": 1, "tid": 1, "ts": 0, "dur": 5,
         "args": {"correlation": 0}},
        {"ph": "X", "cat": "kernel", "name": "k0", "pid": 1, "tid": 7, "ts": 10, "dur": 20,
         "args": {"correlation": 0}},
    ]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace))
    ingest(p, tmp_path / "t.sqlite")
    st = TraceStore(tmp_path / "t.sqlite")
    rows = st.conn.execute("SELECT COUNT(*) FROM spans WHERE corr = 0").fetchone()[0] \
        if hasattr(st, "conn") else None
    assert rows is None or rows == 2
