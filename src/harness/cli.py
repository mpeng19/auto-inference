"""`harness` -- launch and steer a fleet from the terminal.

    harness start --agents 10          # detached; prints a session id and returns
    harness tui                        # watch and control it
    harness status                     # one-shot, scriptable
    harness scale 6                    # add or remove agents in flight
    harness agent pause a03
    harness stop                       # graceful: finish paid work, then wind up
    harness kill                       # flat: everything, now

    harness tool recall "raise chunked prefill"   # what the fleet already knows
    harness tool preflight --workspace agents/a01 # cheap checks before a GPU
    harness tool roofline --batch 12              # predicted step time and cost
    harness tool gpu-run bench.py                 # one script on an H100, minutes
    harness tool equivalence                      # same model, token by token?
    harness traces list | show <id> | export      # the debugging record

`start` is asynchronous by default because a fleet runs for hours and must
outlive the terminal that launched it. Everything after it talks to the session
store, so the CLI and the TUI are just two clients of the same interface and
neither needs the other to exist.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

from .contracts.session import Command
from .daemon import FleetConfig
from .session import SqliteSessionStore, default_store_path


def _store(a) -> SqliteSessionStore:
    return SqliteSessionStore(a.store or default_store_path())


def _resolve(store, session_id: str):
    v = store.read(session_id)
    if v is None:
        print("no session found. `harness start` first.", file=sys.stderr)
    return v


# ── start ────────────────────────────────────────────────────────────────

def _bank_path(value: str) -> str:
    if value == "__default__":
        from .ideas import default_bank_path
        return str(default_bank_path())
    return value


def cmd_start(a) -> int:
    session_id = a.session or f"sess-{int(time.time())}"
    root = pathlib.Path(a.root or (pathlib.Path.cwd() / "agents" / session_id))
    root.mkdir(parents=True, exist_ok=True)
    cfg = FleetConfig(
        session_id=session_id, root=str(root), agents=a.agents,
        eval_capacity=a.evals, budget_usd=a.budget, model=a.model,
        seed_model=a.seed_model, gpu=a.gpu, n_gpu=a.n_gpu,
        agent_max_attempts=a.max_attempts, agent_max_usd=a.agent_budget,
        dry_run=a.dry_run, fake_agents=a.fake_agents, note=a.note,
        bank=_bank_path(a.bank) if a.bank else "", mode=a.mode, manager=a.manager,
        seeds=tuple(s for s in (a.seed or [])),
        baseline=json.loads(a.baseline) if a.baseline else {})
    cfg_path = root / "fleet.json"
    cfg.save(cfg_path)

    log = (root / "daemon.log").open("a")
    # The daemon lives on this laptop because the agents are this laptop's
    # Claude Code login. If the laptop sleeps, so does the fleet: on
    # 2026-09-02 a closed lid froze three agents for five hours while the
    # GPUs they had rented kept running. `caffeinate -i -s` holds off idle
    # and AC sleep for as long as the daemon runs; it cannot override a
    # closed lid without an external display, so the TUI also reports gaps.
    daemon = [sys.executable, "-m", "harness.daemon", "--config", str(cfg_path)]
    caff = shutil.which("caffeinate")
    proc = subprocess.Popen(
        [caff, "-i", "-s", *daemon] if caff else daemon,
        stdout=log, stderr=subprocess.STDOUT,
        # Its own process group, so closing this terminal (or Ctrl-C here)
        # does not take down a fleet holding rented GPUs.
        start_new_session=True,
        env={**os.environ, "HARNESS_SESSION": session_id})
    (root / "daemon.pid").write_text(str(proc.pid))
    print(f"session   {session_id}")
    fakes = [n for n, on in (("no GPUs", a.dry_run), ("scripted agents", a.fake_agents)) if on]
    print(f"agents    {a.agents}   eval capacity {a.evals}"
          + (f"   [{', '.join(fakes)}]" if fakes else ""))
    print(f"root      {root}")
    print(f"pid       {proc.pid}   log {root / 'daemon.log'}")
    print(f"\nwatch:    uv run harness --session {session_id} tui")
    print(f"stop:     uv run harness --session {session_id} stop")
    return 0


# ── read ─────────────────────────────────────────────────────────────────

def cmd_status(a) -> int:
    v = _resolve(_store(a), a.session)
    if v is None:
        return 1
    if a.json:
        from dataclasses import asdict
        print(json.dumps(asdict(v), indent=1, default=str))
        return 0
    age = time.time() - v.updated_at
    print(f"{v.session_id}   {v.phase}   {v.live_agents}/{v.target_agents} agents"
          f"   ${v.cost_usd:.2f} of ${v.budget_usd:.0f}"
          f"   {v.tokens.total:,} tokens   updated {age:.0f}s ago")
    if v.note:
        print(f"note:  {v.note}")
    print(f"evals: {v.evals_running} running, {v.evals_queued} queued, "
          f"{v.evals_completed} done, {v.evals_deduped} deduped, "
          f"{v.gpu_utilisation:.0%} GPU utilisation")
    print()
    print(f"{'agent':<7}{'status':<12}{'idea':<26}{'att':>4}{'Δ%':>8}"
          f"{'$':>8}{'tokens':>11}  activity")
    for ag in v.agents:
        d = "-" if ag.best_delta_pct is None else f"{ag.best_delta_pct:+.1f}"
        print(f"{ag.agent_id:<7}{ag.status:<12}{ag.idea_title[:24]:<26}"
              f"{ag.attempt:>4}{d:>8}{ag.cost_usd:>8.2f}{ag.tokens.total:>11,}"
              f"  {ag.activity[:44]}")
    return 0


def cmd_sessions(a) -> int:
    for v in _store(a).sessions(limit=a.limit):
        print(f"{v.session_id:<22}{v.phase:<10}{len(v.agents):>3} agents"
              f"  ${v.cost_usd:>8.2f}  {v.tokens.total:>12,} tok")
    return 0


# ── control ──────────────────────────────────────────────────────────────

STALE_S = 15.0          # a live fleet publishes about once a second


def _send(a, kind: str, agent_id: str = "", value: str = "") -> int:
    store = _store(a)
    v = _resolve(store, a.session)
    if v is None:
        return 1
    cid = store.send_to(v.session_id, Command(kind=kind, agent_id=agent_id,
                                              value=value))
    # Do not wait on a fleet that is not ticking. A dead session should say so
    # at once rather than making the operator watch a spinner for five seconds
    # to be told the same thing.
    stale = (time.time() - v.updated_at) > STALE_S or v.phase == "stopped"
    if stale:
        print(f"queued {kind}; session {v.session_id} is {v.phase} and last "
              f"published {time.time() - v.updated_at:.0f}s ago", file=sys.stderr)
        return 1
    deadline = time.time() + a.wait
    while time.time() < deadline:
        time.sleep(0.05)
        c = store.command_status(cid)
        if c and c.applied_at:
            print(c.result)
            return 0
    print("sent; not acknowledged yet (is the fleet running?)", file=sys.stderr)
    return 1


def cmd_stop(a) -> int:
    return _send(a, "stop")


def cmd_scale(a) -> int:
    return _send(a, "scale", value=str(a.n))


def cmd_agent(a) -> int:
    return _send(a, a.action, agent_id=a.agent_id)


def cmd_kill(a) -> int:
    """Flat kill: signal the daemon directly rather than asking it politely.

    `stop` lets agents finish work already paid for. This does not -- it is for
    when something is wrong and the bill is the priority.
    """
    store = _store(a)
    v = _resolve(store, a.session)
    if v is None:
        return 1
    store.send_to(v.session_id, Command(kind="stop"))
    # The pid travels in the snapshot: reconstructing it from a conventional
    # path failed exactly when it was needed, which is the wrong time.
    pid = v.pid or _pid_from_disk(a, v)
    if not pid:
        print("no pid recorded; sent a stop command instead", file=sys.stderr)
        return 1
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"terminated process group {pid}")
    except (ProcessLookupError, PermissionError) as e:
        print(f"daemon {pid} already gone ({e})")
    return 0


def _pid_from_disk(a, v) -> int:
    root = pathlib.Path(a.root or v.root or "")
    f = root / "daemon.pid"
    return int(f.read_text().strip()) if f.is_file() else 0


def cmd_tool(a) -> int:
    from . import tools
    return tools.main(a.action, a)


# ── ideas ────────────────────────────────────────────────────────────────

def cmd_ideas(a) -> int:
    """The bank the fleet draws from. Filling it is a one-time cost per
    source; the fleet then claims records one per agent."""
    from .ideas import SqliteIdeaBank, default_bank_path

    bank = SqliteIdeaBank(a.bank or default_bank_path())
    if a.action == "list":
        rows = bank.list(status=a.status or None, scale=a.scale or None)
        if a.json:
            from dataclasses import asdict
            print(json.dumps([asdict(r) for r in rows], indent=1))
            return 0
        print(f"{bank.path}: {bank.count()} records, {bank.count('available')} available")
        for r in rows:
            who = f" [{r.claimed_by}]" if r.claimed_by else ""
            print(f"  {r.id}  {r.scale:12s} {r.status:9s}{who}  {r.title[:60]}  ({r.source})")
        return 0
    if a.action == "show":
        r = bank.get(a.arg)
        if r is None:
            print(f"no record {a.arg}")
            return 1
        from dataclasses import asdict
        print(json.dumps(asdict(r), indent=1))
        return 0
    if a.action == "search":
        for r in bank.search(a.arg, k=a.k):
            print(f"  {r.id}  {r.scale:12s} {r.status:9s}  {r.title[:70]}")
        return 0
    if a.action == "import":
        n = bank.import_jsonl(a.arg, source_default=a.source)
        print(f"imported {n} records from {a.arg} -> {bank.path}")
        return 0
    if a.action == "release":
        bank.release(a.arg)
        print(f"released {a.arg}")
        return 0
    if a.action == "extract-pdf":
        from .ideas import pdf as book
        from .ideas.llm import ask_with
        n = book.harvest(bank, a.arg, ask_with(a.model), book=a.source or "",
                         size=a.pages, progress=lambda m: print(m, flush=True))
        print(f"{n} records from {a.arg} -> {bank.path}")
        return 0
    if a.action == "arxiv":
        from .ideas import arxiv
        from .ideas.llm import ask_with
        queries = (a.arg,) if a.arg else arxiv.DEFAULT_QUERIES
        seen, added = arxiv.harvest(bank, ask_with(a.model), queries=queries,
                                    per_query=a.k)
        print(f"{seen} papers seen, {added} records added -> {bank.path}")
        return 0
    print(f"unknown action {a.action}")
    return 2


# ── traces ───────────────────────────────────────────────────────────────

def cmd_traces(a) -> int:
    from . import traces

    if a.action == "list":
        found = traces.find(a.root or None, session_id=a.session)
        if not found:
            print("no traces found", file=sys.stderr)
            return 1
        print(f"{'trace':<20}{'agent':<7}{'turns':>6}{'dur':>7}{'$':>7}  outcome")
        for t in found[:a.limit]:
            print(f"{t.trace_id:<20}{t.agent_id:<7}{t.n_turns:>6}"
                  f"{t.duration_s:>6.0f}s{t.cost_usd:>7.2f}  {t.outcome}")
        return 0

    if a.action == "show":
        found = [t for t in traces.find(a.root or None)
                 if t.trace_id == a.trace_id or t.trace_id.startswith(a.trace_id)]
        if not found:
            print(f"no trace matching {a.trace_id!r}", file=sys.stderr)
            return 1
        kinds = tuple(k for k in (a.kind or "").split(",") if k)
        for r in traces.read(found[0].path, kinds=kinds, query=a.query,
                             limit=a.limit):
            head = f"{r['seq']:>4} {r['kind']:<13}{(r.get('name') or '')[:16]:<17}"
            body = (r.get("content") or "").replace("\n", " ")
            print(head + body[:a.width])
            if a.full and r.get("data"):
                print(" " * 22 + json.dumps(r["data"], default=str)[:2000])
        return 0

    m = traces.export(a.out, a.root or None, session_id=a.session)
    print(f"exported {m['traces']} traces, {m['lines']} lines -> {a.out}")
    print(f"manifest: {pathlib.Path(a.out) / 'manifest.json'}")
    return 0


def cmd_tui(a) -> int:
    from .tui import run_tui
    return run_tui(store=_store(a), session_id=a.session)


# ── wiring ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description=__doc__.split("\n")[0])
    ap.add_argument("--store", default="", help="session database (default ~/.auto-inference)")
    ap.add_argument("--session", default="", help="session id (default: most recent)")
    ap.add_argument("--wait", type=float, default=5.0,
                    help="seconds to wait for a command to be acknowledged")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="launch a fleet, detached")
    s.add_argument("--agents", type=int, default=4)
    s.add_argument("--evals", type=int, default=2, help="concurrent GPU evaluations")
    s.add_argument("--budget", type=float, default=200.0, help="$ ceiling for the fleet")
    s.add_argument("--agent-budget", dest="agent_budget", type=float, default=40.0)
    s.add_argument("--max-attempts", dest="max_attempts", type=int, default=6)
    s.add_argument("--model", default="sonnet", help="claude model: sonnet | opus")
    s.add_argument("--seed-model", dest="seed_model", default="")
    s.add_argument("--gpu", default="H100")
    s.add_argument("--n-gpu", dest="n_gpu", type=int, default=1)
    s.add_argument("--root", default="")
    s.add_argument("--bank", nargs="?", const="__default__", default="",
                   help="claim ideas from the bank (default path when given no value)")
    s.add_argument("--manager", action="store_true",
                   help="review outcomes and stash reusable tools under <root>/tools/")
    s.add_argument("--mode", choices=["tune", "build"], default="tune",
                   help="build: kernel-scale ideas with a design note and workbench checks")
    s.add_argument("--seed", action="append", help="a starting hypothesis; repeatable")
    s.add_argument("--baseline", default="", help='JSON from stock sweeps: {"bill_per_1k": 14.96, "quality": {"gsm8k": 0.66}, "screen": {"bill_per_1k": 17.3}}')
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="fake the GPU evaluations (saves dollars, still runs "
                        "real Claude Code agents)")
    s.add_argument("--fake-agents", dest="fake_agents", action="store_true",
                   help="fake the agents too (saves subscription usage); "
                        "with --dry-run this exercises the whole fleet for free")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_start)

    st = sub.add_parser("status", help="one-shot snapshot")
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=cmd_status)

    ls = sub.add_parser("sessions", help="list recent sessions")
    ls.add_argument("--limit", type=int, default=10)
    ls.set_defaults(fn=cmd_sessions)

    sub.add_parser("tui", help="live dashboard").set_defaults(fn=cmd_tui)

    tl = sub.add_parser("tool", help="tools for agents (and for reading runs)")
    tl.add_argument("action", choices=["recall", "preflight", "roofline",
                                       "gpu-run", "equivalence"])
    tl.add_argument("intent", nargs="?", default="",
                    help="recall: what you are about to do; "
                         "gpu-run: the script to run")
    tl.add_argument("--workspace", default=".",
                    help="the agent directory (preflight, gpu-run, equivalence)")
    tl.add_argument("--timeout", type=int, default=0,
                    help="gpu-run/equivalence: seconds the script itself gets; "
                         "0 takes the tool's own default (600s, 1800s), which "
                         "already allows for a 3-5 minute engine load")
    tl.add_argument("--context", type=int, default=20583)
    tl.add_argument("--batch", type=int, default=12)
    tl.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    tl.add_argument("--gpu", default="H100")
    tl.add_argument("--n-gpu", dest="n_gpu", type=int, default=1)
    tl.add_argument("--root", default="", help="fleet root, for recall")
    tl.add_argument("-k", type=int, default=8)
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(fn=cmd_tool)

    ideas = sub.add_parser("ideas", help="the idea bank: fill it, inspect it")
    ideas.add_argument("action", choices=["list", "show", "search", "import", "release",
                                          "extract-pdf", "arxiv"])
    ideas.add_argument("arg", nargs="?", default="",
                       help="id | query | jsonl path | pdf path")
    ideas.add_argument("--bank", default="", help="database path (default: shared)")
    ideas.add_argument("--status", default="", help="list: filter")
    ideas.add_argument("--scale", default="", help="list: filter")
    ideas.add_argument("--source", default="", help="import/extract-pdf: source label")
    ideas.add_argument("--model", default="opus", help="extract-pdf/arxiv: claude model")
    ideas.add_argument("--pages", type=int, default=20, help="extract-pdf: window size")
    ideas.add_argument("-k", type=int, default=25, help="search: hits; arxiv: per query")
    ideas.add_argument("--json", action="store_true")
    ideas.set_defaults(fn=cmd_ideas)

    tr = sub.add_parser("traces", help="list, read, or export agent traces")
    tr.add_argument("action", choices=["list", "show", "export"])
    tr.add_argument("trace_id", nargs="?", default="")
    tr.add_argument("--root", default="", help="fleet root (default: ./agents)")
    tr.add_argument("--out", default="trace-export", help="export destination")
    tr.add_argument("--kind", default="", help="comma-separated turn kinds")
    tr.add_argument("--query", default="", help="substring filter")
    tr.add_argument("--limit", type=int, default=200)
    tr.add_argument("--width", type=int, default=140)
    tr.add_argument("--full", action="store_true", help="include the data blob")
    tr.set_defaults(fn=cmd_traces)
    sub.add_parser("stop", help="graceful: finish paid work, then wind up").set_defaults(fn=cmd_stop)

    k = sub.add_parser("kill", help="flat kill: everything, now")
    k.add_argument("--root", default="")
    k.set_defaults(fn=cmd_kill)

    sc = sub.add_parser("scale", help="change the number of agents in flight")
    sc.add_argument("n", type=int)
    sc.set_defaults(fn=cmd_scale)

    ag = sub.add_parser("agent", help="pause / resume / kill one agent")
    ag.add_argument("action", choices=["pause", "resume", "kill"])
    ag.add_argument("agent_id")
    ag.set_defaults(fn=cmd_agent)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
