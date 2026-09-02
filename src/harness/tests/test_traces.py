"""Trace persistence: the contract a downstream profile database relies on."""
import json

import pytest

from harness import traces
from harness.context import JsonlContext
from harness.contracts import TraceMeta, Turn


@pytest.fixture
def written(tmp_path):
    c = JsonlContext(tmp_path / "traces", session_id="sess-1")
    ref = c.open(TraceMeta(agent_id="a01", idea_id="i9", attempt=2, model="sonnet"))
    c.append(ref, Turn(kind="prompt", content="improve prefix reuse"))
    c.append(ref, Turn(kind="eval_submit", name="abc123",
                       data={"tier": "screen", "diff": "--- a\n+++ b"}))
    c.append(ref, Turn(kind="eval_result", name="abc123", data={"bill_per_1k": 11.4}))
    c.close(ref, outcome="won", cost_usd=1.5)
    return tmp_path, ref


def test_every_line_stands_alone(written):
    """`cat traces/*.jsonl | loader` must not lose which agent produced what."""
    tmp, ref = written
    lines = (tmp / "traces" / f"{ref}.jsonl").read_text().strip().splitlines()
    for i, ln in enumerate(lines):
        d = json.loads(ln)
        assert d["v"] == 1
        assert d["trace_id"] == ref
        assert d["seq"] == i
        assert d["session_id"] == "sess-1"
        assert d["agent_id"] == "a01" and d["idea_id"] == "i9"
        assert d["attempt"] == 2


def test_seq_orders_within_a_trace(written):
    """Two turns can share a timestamp; seq is the ordering key."""
    tmp, ref = written
    recs = traces.read(tmp / "traces" / f"{ref}.jsonl")
    assert [r["seq"] for r in recs] == [0, 1, 2]


def test_find_folds_in_the_sidecar(written):
    tmp, ref = written
    found = traces.find(tmp)
    assert len(found) == 1
    t = found[0]
    assert t.trace_id == ref and t.agent_id == "a01"
    assert t.outcome == "won" and t.cost_usd == 1.5 and t.n_turns == 3


def test_reading_filters_by_kind_and_query(written):
    tmp, ref = written
    p = tmp / "traces" / f"{ref}.jsonl"
    assert len(traces.read(p, kinds=("eval_submit",))) == 1
    assert len(traces.read(p, query="prefix reuse")) == 1
    assert traces.read(p, kinds=("eval_submit",))[0]["data"]["tier"] == "screen"


def test_pre_envelope_traces_stay_readable(tmp_path):
    """Files written before the envelope existed have no `v` and no `seq`.

    A debugging tool that refuses yesterday's traces is not much of one, and a
    downstream loader will meet the same files.
    """
    d = tmp_path / "traces"
    d.mkdir(parents=True)
    (d / "trc_old.jsonl").write_text(
        json.dumps({"kind": "prompt", "ts": 1.0, "content": "old"}) + "\n"
        + json.dumps({"kind": "message", "ts": 2.0, "content": "still old"}) + "\n")
    (d / "trc_old.meta.json").write_text(json.dumps(
        {"id": "trc_old", "agent_id": "a07", "idea_id": "i3", "attempt": 1}))
    recs = traces.read(d / "trc_old.jsonl")
    assert [r["seq"] for r in recs] == [0, 1]
    assert all(r["v"] == 0 for r in recs)
    assert recs[0]["agent_id"] == "a07", "ids must be recovered from the sidecar"


def test_a_newer_schema_is_refused_not_guessed_at(tmp_path):
    d = tmp_path / "traces"
    d.mkdir(parents=True)
    f = d / "trc_future.jsonl"
    f.write_text(json.dumps({"v": 99, "kind": "prompt", "seq": 0}) + "\n")
    with pytest.raises(ValueError, match="newer than this reader"):
        traces.read(f)


def test_export_carries_a_verifiable_manifest(written, tmp_path):
    tmp, _ = written
    out = tmp_path / "out"
    m = traces.export(out, tmp)
    assert m["schema_version"] == 1 and m["traces"] == 1 and m["lines"] == 3
    assert (out / "manifest.json").is_file()
    # the sidecar travels too, or `outcome` and `cost_usd` are lost
    assert len(list(out.glob("*.meta.json"))) == 1
    loaded = json.loads((out / "manifest.json").read_text())
    assert loaded["files"][0]["lines"] == 3
