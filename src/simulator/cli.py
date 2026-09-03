"""Thin command line over `Simulator`. Everything it does, the API does.

    simulate run     --root runs/baseline
    simulate submit  --root runs/baseline           # a sweep is 25-60 min
    simulate collect --root runs/baseline           # picks up the stored call
    simulate rescore --root runs/baseline --slo ttft:p99:1000,tpot:mean:20
    simulate ls

    simulate workbench   --root runs/k --stack k/ probe.py   # one script, one GPU
    simulate equivalence --root runs/k --stack k/            # same model or not?

`workbench` and `equivalence` are the inner loop for kernel work: minutes and
one script, rather than 25-60 minutes and a price. A kernel needs to be asked
whether it compiles and whether it still computes the same thing long before it
is worth asking what it costs to serve.

`rescore` is the one worth knowing about: it re-judges and re-prices a stored
sweep with no GPU at all, because every level keeps its full percentile set and
its raw counters. Changing the SLO, the cost basis or the utilisation
assumption should never cost 25 GPU-minutes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

from .api import APP_NAME, Simulator
from .measure import equivalence
from .slo import MARKET_SLO, SLO
from .stack import InferenceStack


def _build(a) -> Simulator:
    root = pathlib.Path(a.root)
    if a.mkdir:
        root.mkdir(parents=True, exist_ok=True)
    stack = InferenceStack.from_dir(a.stack) if a.stack else InferenceStack.stock()
    slo = SLO.parse(a.slo) if a.slo else SLO(bounds=MARKET_SLO)
    kw = {}
    if a.levels:
        kw["levels"] = tuple(int(x) for x in a.levels.split(",") if x.strip())
    return Simulator(root_dir=root, stack=stack, slo=slo, model=a.model,
                     gpu=a.gpu, n_gpu=a.n_gpu, seconds_per_level=a.seconds,
                     repeats=a.repeats, rate_per_gpu_hour=a.rate,
                     gpu_provider=a.provider,
                     utilisation=a.utilisation, note=a.note,
                     canaries=not a.no_canaries,
                     profile_level=a.profile_level,
                     profile_steps=a.profile_steps, **kw)


def _report(sim: Simulator, res) -> int:
    print((sim.root / "report.txt").read_text())
    print("artifacts:")
    for k, v in res.artifacts.items():
        print(f"  {k:<16} {v}")
    return 0 if res.ok else 1


def cmd_profile(a) -> int:
    """Download a captured GPU profile and ingest it into a trace database.

    The capture lives on the results volume; this brings it local and hands it
    to `tracedb`, after which the questions are queries rather than a 40 MB
    JSON nobody can read.
    """
    import base64

    import modal

    fn = modal.Function.from_name(APP_NAME, "fetch_profile")
    got = fn.remote(a.dir)
    if not got["files"]:
        print(f"no trace files under {got['dir']}", file=sys.stderr)
        return 1
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in got["files"]:
        p = out / f["name"]
        p.write_bytes(base64.b64decode(f["b64"]))
        written.append(p)
        print(f"  {p}  ({f['size']:,} bytes)")

    traces = [p for p in written if p.suffix in (".json", ".gz", ".jsonl")]
    if not traces:
        print("nothing that looks like a chrome trace; not ingesting")
        return 0
    from tracedb.ingest import ingest
    db = out / "trace.sqlite"
    print(json.dumps(ingest(traces[0], db), indent=1))
    print(f"\nquery it:  tracedb --db {db} summary")
    return 0


def cmd_workbench(sim: Simulator, a) -> int:
    """Run one script on an H100 against a stack. Minutes, not a sweep.

    The inner loop for kernel work: does it compile, is it faster, does it
    still compute the same thing. Everything the script printed comes back, and
    lands in `<root>/workbench-<n>/`.
    """
    script = pathlib.Path(a.script)
    if not script.is_file():
        print(f"no script at {script}", file=sys.stderr)
        return 2
    rec = asyncio.run(sim.workbench(script.read_text(), timeout_s=a.timeout))
    print(("OK" if rec.get("ok") else "FAILED")
          + f"   exit {rec.get('exit_code')}"
          + f"   {rec.get('elapsed_s', 0)}s on {rec.get('gpu', '?')}"
          + f"   ${rec.get('cost_usd', 0):.3f}")
    if rec.get("stdout"):
        print(rec["stdout"], end="" if rec["stdout"].endswith("\n") else "\n")
    if rec.get("stderr"):
        print("--- stderr ---", file=sys.stderr)
        print(rec["stderr"], file=sys.stderr)
    print(f"artifacts: {rec.get('dir')}")
    return 0 if rec.get("ok") else 1


def cmd_equivalence(sim: Simulator, a) -> int:
    """Token-level equivalence against stock: the gate GSM8K is too blunt for."""
    rec = asyncio.run(sim.equivalence(timeout_s=a.timeout,
                                      min_agreement=a.min_agreement,
                                      max_mean_dlogprob=a.max_mean_dlogprob))
    if not rec.get("ok"):
        print(f"NOT MEASURED: {rec.get('error', 'unknown')}", file=sys.stderr)
        print(f"  spent ${rec.get('cost_usd', 0):.3f}", file=sys.stderr)
        return 1
    print(f"{sim.stack.describe()}")
    print(f"  {rec['summary']}")
    print(f"  {'REGRESSION: ' + rec['why'] if rec['regressed'] else 'equivalent'}")
    print(f"  reference  {rec['reference_path']}")
    print(f"  spent      ${rec['cost_usd']:.3f}")
    return 1 if rec["regressed"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="simulate", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_root=True):
        p.add_argument("--root", required=need_root,
                       help="artifact directory; must exist unless --mkdir")
        p.add_argument("--mkdir", action="store_true")
        p.add_argument("--stack", default="", help="directory mirroring sglang/")
        p.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
        p.add_argument("--gpu", default="H100")
        p.add_argument("--n-gpu", dest="n_gpu", type=int, default=1)
        p.add_argument("--levels", default="", help="e.g. 4,8,12,16,24")
        p.add_argument("--seconds", type=float, default=120.0)
        p.add_argument("--repeats", type=int, default=1)
        p.add_argument("--slo", default="", help="ttft:p90:2818,tpot:mean:20")
        p.add_argument("--provider", default=None,
                       help="price against a named provider, e.g. nebius-committed")
        p.add_argument("--rate", type=float, default=None,
                       help="$/GPU-hour; overrides --provider")
        p.add_argument("--utilisation", type=float, default=0.50)
        p.add_argument("--no-canaries", action="store_true")
        p.add_argument("--profile-level", dest="profile_level", type=int, default=0,
                       help="capture a GPU profile at this concurrency level")
        p.add_argument("--profile-steps", dest="profile_steps", type=int, default=20)
        p.add_argument("--note", default="")

    common(sub.add_parser("run", help="submit, wait, analyse, write artifacts"))
    common(sub.add_parser("submit", help="start a sweep and return its call id"))
    c = sub.add_parser("collect", help="wait for a submitted sweep")
    common(c)
    c.add_argument("--call-id", default="", help="defaults to <root>/call_id")
    r = sub.add_parser("rescore", help="re-judge a stored sweep, no GPU")
    common(r)
    r.add_argument("--sweep", default="", help="defaults to <root>/sweep.json")
    w = sub.add_parser("workbench", help="run one script on a GPU against a stack")
    common(w)
    w.add_argument("script", help="a python file to run in the container")
    w.add_argument("--timeout", type=int, default=600,
                   help="seconds the script itself gets (an engine load is 3-5 min)")
    e = sub.add_parser("equivalence",
                       help="token-level equivalence against stock, no sweep")
    common(e)
    e.add_argument("--timeout", type=int, default=1800)
    e.add_argument("--min-agreement", dest="min_agreement", type=float,
                   default=equivalence.MIN_AGREEMENT)
    e.add_argument("--max-mean-dlogprob", dest="max_mean_dlogprob", type=float,
                   default=equivalence.MAX_MEAN_DLOGPROB)
    pr = sub.add_parser("profile", help="download and ingest a captured GPU profile")
    pr.add_argument("--dir", required=True, help="profile dir from the sweep record")
    pr.add_argument("--out", default="profiles", help="local destination")

    sub.add_parser("ls", help="list stored sweeps")

    a = ap.parse_args(argv)

    if a.cmd == "ls":
        import modal
        for n in modal.Function.from_name(APP_NAME, "ls").remote(40):
            print(n)
        return 0

    # `profile` names a captured directory, not a run: it has none of the
    # options `_build` needs and building a Simulator for it raised
    # AttributeError instead of downloading anything.
    if a.cmd == "profile":
        return cmd_profile(a)

    sim = _build(a)
    if a.cmd == "workbench":
        return cmd_workbench(sim, a)
    if a.cmd == "equivalence":
        return cmd_equivalence(sim, a)
    if a.cmd == "submit":
        cid = sim.submit()
        print(f"submitted  {cid}")
        print(f"collect:   simulate collect --root {sim.root}")
        return 0
    if a.cmd == "run":
        return _report(sim, asyncio.run(sim.eval()))
    if a.cmd == "collect":
        cid = a.call_id or (sim.root / "call_id").read_text().strip()
        return _report(sim, asyncio.run(sim.collect(cid)))
    if a.cmd == "rescore":
        path = pathlib.Path(a.sweep or (sim.root / "sweep.json"))
        return _report(sim, sim.finish(json.loads(path.read_text())))
    return 2


if __name__ == "__main__":
    sys.exit(main())
