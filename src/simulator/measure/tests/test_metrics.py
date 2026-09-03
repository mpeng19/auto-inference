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
