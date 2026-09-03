"""Goodput, percentiles, and the collapse detector.

TTFT and TPOT are *derived* from timestamps rather than stored, so these build
real timelines: it keeps the test honest about the arithmetic that turns wall
clock into the numbers the SLO is judged on.
"""
from simulator.measure.metrics import RequestResult, detect_collapse, summarize
from simulator.slo import SLO

N_OUT = 51                      # so TPOT divides by 50


def _r(i, ttft_ms, tpot_ms, ok=True):
    t0 = float(i)
    first = t0 + ttft_ms / 1000
    return RequestResult(
        idx=i, scheduled_s=t0, dispatched_s=t0, first_token_s=first if ok else None,
        end_s=first + tpot_ms * (N_OUT - 1) / 1000 if ok else None,
        prompt_tokens=100, output_tokens=N_OUT, ok=ok,
        error=None if ok else "boom")


def test_goodput_counts_only_requests_inside_the_envelope():
    slo = SLO.parse("ttft:p90:1000,tpot:p90:25")
    res = [_r(i, 100, 10) for i in range(8)] + [_r(8, 5000, 10), _r(9, 100, 90)]
    m = summarize(res, slo)
    assert m["n_ok"] == 10 and m["n_good"] == 8


def test_a_failed_request_is_never_good():
    m = summarize([_r(0, 100, 10), _r(1, 100, 10, ok=False)],
                  SLO.parse("ttft:p90:1000,tpot:p90:25"))
    assert m["n_failed"] == 1 and m["n_good"] == 1


def test_every_percentile_is_kept_so_the_slo_can_change_later():
    m = summarize([_r(i, 100 + i, 10) for i in range(50)], SLO.parse("ttft:p90:1000"))
    for k in ("mean", "p50", "p90", "p95", "p99", "max", "n"):
        assert k in m["ttft_ms"], k


def test_goodput_uses_the_loosest_bound_not_the_tightest():
    """`good_frac` is a blow-up detector. Judging each request against the
    tightest bound would fail most of a level whose p90 sits inside its limit.
    """
    slo = SLO.parse("tpot:p90:25,tpot:mean:20")
    m = summarize([_r(i, 100, 22) for i in range(20)], slo)
    assert m["n_good"] == 20


def test_collapse_is_measured_not_averaged_away():
    """Two identical 30-minute runs gave goodput 30.29 and 0.54. Averaging
    those describes a state the server is never in."""
    flat = [_r(i, 40, 10) for i in range(150)]
    assert not detect_collapse(flat)["collapsed"]
    runaway = [_r(i, 40 + max(0, i - 75) ** 2, 10) for i in range(150)]
    assert detect_collapse(runaway)["collapsed"]


def test_a_level_ends_at_the_deadline_not_at_the_last_reply():
    """Letting in-flight 2,000-token replies drain made a 120 s level take
    six minutes; the cut-off cancels what is still streaming."""
    import asyncio

    from simulator.measure.loadgen import run_until

    async def go():
        async def quick():
            await asyncio.sleep(0.01)
            return "done"

        async def slow():
            await asyncio.sleep(10)
            return "never"

        tasks = [asyncio.create_task(quick()), asyncio.create_task(slow()),
                 asyncio.create_task(slow())]
        cancelled = await run_until(tasks, 0.2)
        return cancelled, tasks[0].result(), tasks[1].cancelled()

    cancelled, first, second_cancelled = asyncio.run(go())
    assert cancelled == 2 and first == "done" and second_cancelled
    assert asyncio.run(run_until([], 1.0)) == 0


def test_littles_law_recovers_a_think_time_it_was_not_told():
    """L = lambda*W ties concurrency, throughput and residence time by an
    identity, so it costs nothing and catches what nothing else here does."""
    from simulator.measure.metrics import littles_law

    # 10 users, 2 rps, 4 in the server: 2 s in the server, 5 s per cycle.
    got = littles_law(n_users=10, throughput_rps=2.0, batch=4.0)
    assert got["available"]
    assert got["time_in_server_s"] == 2.0
    assert got["cycle_s"] == 5.0
    assert got["implied_think_s"] == 3.0


def test_littles_law_says_so_when_nothing_finished():
    from simulator.measure.metrics import littles_law

    assert littles_law(8, 0.0, 3.0) == {"available": False, "reason": "no throughput"}


def test_think_time_drifting_across_a_sweep_is_a_client_bug_not_a_finding():
    """Think time is a property of the workload. If it climbs with load the
    generator is not holding `n_users` in flight -- the failure that once moved
    N* from 128 to 32."""
    from simulator.measure.metrics import littles_law_drift

    flat = [{"n_users": n, "throughput_rps": n / 5.0, "batch": 2.0 * n / 5.0}
            for n in (10, 20, 40)]
    assert littles_law_drift(flat)["consistent"]

    # Same levels, but throughput stops scaling: the client fell behind.
    starved = [{"n_users": 10, "throughput_rps": 2.0, "batch": 4.0},
               {"n_users": 40, "throughput_rps": 2.2, "batch": 4.4}]
    got = littles_law_drift(starved)
    assert got["available"] and not got["consistent"]
    assert got["drift"] > 1.5


def test_drift_needs_more_than_one_usable_level():
    from simulator.measure.metrics import littles_law_drift

    assert littles_law_drift([]) == {"available": False}


def test_client_minus_server_ttft_is_the_client_overhead():
    """The module's whole claim: the *difference* between the two TTFTs is
    client-plus-network, so a run can prove its own client was clean."""
    from simulator.measure.server import compare_client_server

    got = compare_client_server(
        {"p50": 120.0}, {"sglang:time_to_first_token_seconds": {"p50": 0.100}})
    assert got["available"]
    assert got["server_p50_ms"] == 100.0 and got["client_p50_ms"] == 120.0
    assert got["overhead_ms"] == 20.0 and got["overhead_pct"] == 16.7


def test_no_verdict_when_either_side_is_missing():
    """Silence, not a fabricated zero: the check exists to catch a polluted
    measurement, and inventing one would defeat it."""
    from simulator.measure.server import compare_client_server

    assert compare_client_server({"p50": 120.0}, {}) == {"available": False}
    assert compare_client_server(
        {}, {"sglang:time_to_first_token_seconds": {"p50": 0.1}}) == {"available": False}


def test_powerlaw_reports_the_exponent_it_fitted():
    """`b` is the finding, not decoration: below 1 on throughput-versus-latency
    is the shape of a system running out of headroom."""
    from simulator.artifacts.plots import powerlaw

    xs = [1.0, 2.0, 4.0, 8.0]
    a, b, r2 = powerlaw(xs, [3.0 * x ** 0.5 for x in xs])
    assert abs(a - 3.0) < 1e-9 and abs(b - 0.5) < 1e-9
    assert abs(r2 - 1.0) < 1e-9


def test_powerlaw_refuses_data_that_cannot_support_a_fit():
    """None, not a line through two points: a fitted exponent that is really an
    interpolation would be read as a measurement."""
    from simulator.artifacts.plots import powerlaw

    assert powerlaw([1.0, 2.0], [1.0, 2.0]) is None          # too few points
    assert powerlaw([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None  # no x spread
    # Non-positive values cannot be logged, and dropping them can leave too few.
    assert powerlaw([1.0, 2.0, 4.0, 8.0], [1.0, 0.0, -1.0, 2.0]) is None


def test_startup_marks_read_the_server_log_timestamps(tmp_path):
    import datetime as dt

    from simulator.measure.server import startup_marks

    t0 = dt.datetime(2026, 9, 3, 10, 0, 0)
    def stamp(sec): return (t0 + dt.timedelta(seconds=sec)).strftime("[%Y-%m-%d %H:%M:%S TP0]")
    log = tmp_path / "sglang.log"
    log.write_text("\n".join([
        "some unstamped line",
        f"{stamp(5)} Load weight begin. avail mem=79.10 GB",
        f"{stamp(95)} Load weight end. type=Qwen3ForCausalLM",
        f"{stamp(120)} Capture cuda graph begin. This can take up to several minutes.",
        f"{stamp(160)} Capture cuda graph end. Time elapsed: 40.00 s.",
        f"{stamp(170)} The server is fired up and ready to roll!"]))
    marks = startup_marks(str(log), t0.timestamp() - 2.0)
    assert marks == {"weights_begin": 7.0, "weights_end": 97.0, "graph_begin": 122.0,
                     "graph_end": 162.0, "ready": 172.0}
    assert startup_marks(str(tmp_path / "missing.log"), 0.0) == {}


def test_a_level_ends_by_aborting_what_is_still_streaming():
    """The cancelled streams keep generating until the server notices the
    disconnect; on build-4 the next level's flush waited its full 90 s at
    four of five levels because of it. Ask the server, don't wait."""
    import asyncio
    import contextlib

    from simulator.measure.loadgen import abort_all

    seen = []

    class Http:
        def __init__(self, status=200, fail=False):
            self.status, self.fail = status, fail

        def post(self, url, json=None, timeout=None):
            seen.append((url, json))
            status, fail = self.status, self.fail

            @contextlib.asynccontextmanager
            async def cm():
                if fail:
                    raise OSError("connection refused")
                yield type("R", (), {"status": status})()
            return cm()

    assert asyncio.run(abort_all("http://s:1", Http())) is True
    assert asyncio.run(abort_all("http://s:1", Http(status=400))) is False
    assert asyncio.run(abort_all("http://s:1", Http(fail=True))) is False
    assert seen[0] == ("http://s:1/abort_request", {"abort_all": True})
