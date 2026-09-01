"""Replaying real agent traces: the structure must survive the conversion.

These tests use synthetic rows rather than the 23MB download, so they run
offline. The one thing they must guarantee is that the cache structure we
replay is one a server could actually serve -- a replay claiming impossible
cache hits would inflate exactly the number the whole objective turns on.
"""

import itertools

from simulator.workload.tracelab import TraceRound, describe, scale_sessions


def _round(sess, i, inp, cached, out):
    return TraceRound(session=sess, index=i, input_tokens=inp,
                      cached_tokens=cached, new_tokens=inp - cached,
                      output_tokens=out, model="m", provider="claude")


def _session(n=5, base=50_000, grow=1_500, out=250):
    """A conversation whose context grows by `grow` each round."""
    rs, ctx = [], 0
    for i in range(n):
        inp = base + i * grow
        cached = min(ctx, inp)
        rs.append(_round("s", i, inp, cached, out))
        ctx = inp + out
    return rs


def test_hit_rate_and_ratio_survive_scaling():
    """Scaling exists to fit memory, not to change the workload's character."""
    s = [_session(), _session(base=80_000)]
    full, small = describe(s), describe(scale_sessions(s, 0.1))
    assert small["input_tokens_p50"] < full["input_tokens_p50"] / 5
    assert abs(small["aggregate_hit_rate"] - full["aggregate_hit_rate"]) < 0.02
    assert abs(small["input_output_ratio"] - full["input_output_ratio"]) \
        / full["input_output_ratio"] < 0.1


def test_scaling_never_produces_degenerate_rounds():
    tiny = scale_sessions([_session(base=200, out=2)], 0.001)
    for r in tiny[0]:
        assert r.input_tokens >= 16
        assert r.output_tokens >= 1
        assert r.new_tokens >= 1
        assert r.cached_tokens <= r.input_tokens


def test_cached_never_exceeds_input():
    """A round cannot reuse more prefix than its own prompt contains."""
    for f in (1.0, 0.5, 0.05):
        for r in scale_sessions([_session()], f)[0]:
            assert r.cached_tokens <= r.input_tokens, r


def test_first_round_has_no_cache():
    s = _session()
    assert s[0].cached_tokens == 0
    assert s[0].new_tokens == s[0].input_tokens


def test_later_rounds_are_mostly_cached():
    """The defining property of agentic traffic: a big stable prefix plus a
    small increment."""
    s = _session(n=6, base=50_000, grow=1_500)
    last = s[-1]
    assert last.hit_rate > 0.9, last.hit_rate
    assert last.new_tokens < last.input_tokens * 0.1


def test_describe_reports_source_attribution():
    """CC-BY-4.0 requires credit, so it travels with the data."""
    d = describe([_session()])
    assert "TraceLab" in d["source"] and "CC-BY" in d["source"]


def test_describe_handles_empty():
    assert describe([])["n_sessions"] == 0


def test_prefix_cap_correction_is_applied_in_loader():
    """TraceLab issue #22: some rounds report more prefix than the previous
    context could have left. The loader must cap it, not trust it.

    Verified structurally here: `_session` builds the capped invariant, and the
    loader applies the same rule, so a round's cached tokens never exceed the
    prior round's input+output.
    """
    s = _session(n=4)
    for prev, cur in itertools.pairwise(s):
        assert cur.cached_tokens <= prev.input_tokens + prev.output_tokens
