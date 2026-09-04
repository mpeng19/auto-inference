"""Launch and steer a fleet of coding agents from the terminal.

Running a fleet:

    harness start --agents 3           # detached; prints a session id and returns
    harness tui                        # watch and control it
    harness status                     # one-shot, scriptable
    harness scale 6                    # add or remove agents in flight
    harness agent pause a03
    harness stop                       # graceful: finish paid work, then wind up
    harness kill                       # flat: everything, now
    harness delete --session S --yes   # a finished fleet's directory and rows
    harness campaign start --rounds 4 --target 2.0 ...   # fleets in sequence, each on the last win
    harness campaign status | stop

Reading one, during or after:

    harness results --diff              # experiments, best first, with the diff
    harness timeline                    # who did what, when, and for how long
    harness calls -v                    # every model call, tokens and tools
    harness paper                       # the write-up per idea
    harness ask "which attempt touched the attention backend?"
    harness traces list | show <id> | export      # the debugging record

Filling the banks, and the tools an agent calls from its own shell:

    harness ideas seed                                      # the packaged bank
    harness skills list                                    # facts across runs
    harness tool recall "raise chunked prefill"             # what is known
    harness tool preflight --workspace agents/S/a01         # free checks
    harness tool roofline --batch 12                        # step time and cost
    harness tool gpu-run bench.py                           # H100, minutes, ~$1
    harness tool equivalence                                # still the same model?

`start` is asynchronous by default because a fleet runs for hours and must
outlive the terminal that launched it. Everything after it talks to the session
store, so the CLI and the TUI are just two clients of the same interface and
neither needs the other to exist.
"""
from __future__ import annotations

import argparse
import contextlib
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


def running_daemon(root: pathlib.Path) -> int:
    """The pid of a live daemon for this root, or 0."""
    f = root / "daemon.pid"
    if not f.is_file():
        return 0
    try:
        pid = int(f.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return 0


def runner_status(remote=None) -> tuple[str, str, str]:
    """(local digest, deployed digest, problem). The deployed digest comes
    from the app's `version` function; a deploy older than that function,
    or no deploy at all, reads as a problem too."""
    from simulator.runner.modal_runner import source_digest

    local = source_digest()
    try:
        if remote is None:
            import modal

            from simulator.api import APP_NAME
            remote = modal.Function.from_name(APP_NAME, "version").remote
        got = remote() or {}
        deployed = str(got.get("digest", ""))
    except Exception as e:
        return local, "", f"could not read the deployed runner ({type(e).__name__}: {e})"
    if deployed != local:
        return local, deployed, "the deployed runner is behind this checkout"
    return local, deployed, ""


def _require_current_runner(allow_stale: bool, remote=None) -> None:
    """Refuse to start a fleet against a runner that lags the source.
    The abort-at-deadline and startup-mark changes sat undeployed for a
    day while two fleets ran; nothing said so."""
    local, deployed, problem = runner_status(remote)
    if not problem:
        return
    msg = (f"{problem}\n  local  {local}\n  deployed  {deployed or '(none)'}\n"
           "  run `make deploy`, then start again (or --allow-stale-runner to run anyway)")
    if allow_stale:
        print("warning: " + msg, file=sys.stderr)
        return
    raise SystemExit(msg)


def _refuse_second_daemon(root: pathlib.Path, session_id: str, force: bool) -> None:
    pid = running_daemon(root)
    if pid and not force:
        # Two daemons on one root share the agent directories: each agent's
        # workspace is reset and re-seeded under the other's feet. The first
        # build run lost its kernel this way.
        raise SystemExit(f"a fleet is already running on {root} (pid {pid}); "
                         f"`harness --session {session_id} stop` first, or --force")


def _config_from_args(a, session_id: str, root: str) -> FleetConfig:
    """A `FleetConfig` from the `start` options (shared with `campaign start`)."""
    cfg = FleetConfig(
        session_id=session_id, root=root, agents=a.agents,
        eval_capacity=a.evals, budget_usd=a.budget, model=a.model,
        gpu=a.gpu, n_gpu=a.n_gpu,
        agent_max_attempts=a.max_attempts, agent_max_usd=a.agent_budget,
        stall_minutes=a.stall_minutes, auto_ablate=a.auto_ablate,
        dry_run=a.dry_run, fake_agents=a.fake_agents, note=a.note,
        bank=_bank_path(a.bank) if a.bank else "", manager=a.manager,
        profile_level=a.profile_level,
        seeds=tuple(s for s in (a.seed or [])),
        baseline=json.loads(a.baseline) if a.baseline else {})
    if a.base:
        cfg = cfg.with_base(a.base)          # refuses here, not in the daemon's log
    return cfg


def _spawn_daemon(root: pathlib.Path, argv: list[str], session_id: str) -> int:
    """Start a detached process under `caffeinate`, logging to
    `<root>/daemon.log`, its pid in `<root>/daemon.pid`."""
    log = (root / "daemon.log").open("a")
    # The daemon lives on this laptop because the agents are this laptop's
    # Claude Code login. If the laptop sleeps, so does the fleet: on
    # 2026-09-02 a closed lid froze three agents for five hours while the
    # GPUs they had rented kept running. `caffeinate -i -s` holds off idle
    # and AC sleep for as long as the daemon runs; it cannot override a
    # closed lid without an external display, so the TUI also reports gaps.
    caff = shutil.which("caffeinate")
    proc = subprocess.Popen(
        [caff, "-i", "-s", *argv] if caff else argv,
        stdout=log, stderr=subprocess.STDOUT,
        # Its own process group, so closing this terminal (or Ctrl-C here)
        # does not take down a fleet holding rented GPUs.
        start_new_session=True,
        env={**os.environ, "HARNESS_SESSION": session_id})
    (root / "daemon.pid").write_text(str(proc.pid))
    return proc.pid


def cmd_start(a) -> int:
    session_id = a.session or f"sess-{int(time.time())}"
    root = pathlib.Path(a.root or (pathlib.Path.cwd() / "agents" / session_id))
    root.mkdir(parents=True, exist_ok=True)
    _refuse_second_daemon(root, session_id, a.force)
    cfg = _config_from_args(a, session_id, str(root))
    if not (a.fake_agents or getattr(a, "dry_run", False)):
        _require_current_runner(getattr(a, "allow_stale_runner", False))
    cfg_path = root / "fleet.json"
    cfg.save(cfg_path)
    pid = _spawn_daemon(root, [sys.executable, "-m", "harness.daemon",
                               "--config", str(cfg_path)], session_id)
    print(f"session   {session_id}")
    fakes = [n for n, on in (("no GPUs", a.dry_run), ("scripted agents", a.fake_agents)) if on]
    print(f"agents    {a.agents}   eval capacity {a.evals}"
          + (f"   [{', '.join(fakes)}]" if fakes else ""))
    print(f"root      {root}")
    if cfg.base:
        print(f"base      {cfg.base_digest}  {cfg.base_label}  ({cfg.base})")
    print(f"pid       {pid}   log {root / 'daemon.log'}")
    print(f"\nwatch:    uv run harness --session {session_id} tui")
    print(f"stop:     uv run harness --session {session_id} stop")
    return 0


# ── campaign ─────────────────────────────────────────────────────────────

def _campaign_root(a, name: str) -> pathlib.Path:
    return pathlib.Path(a.root or (pathlib.Path.cwd() / "agents" / name))


def cmd_campaign(a) -> int:
    """Fleets in sequence, each starting from the last round's best
    publishable result (`harness.campaign`)."""
    from . import campaign as cp
    from .daemon import check

    if a.action == "start":
        name = a.session or f"camp-{int(time.time())}"
        root = _campaign_root(a, name)
        root.mkdir(parents=True, exist_ok=True)
        _refuse_second_daemon(root, name, a.force)
        (root / cp.STOP_FILE).unlink(missing_ok=True)
        template = _config_from_args(a, name, "")
        if not a.fake_agents:
            _require_current_runner(getattr(a, "allow_stale_runner", False))
        # Refused here, in the terminal: the first round's config is what
        # the daemon would check an hour from now.
        from dataclasses import asdict, replace
        check(replace(template, session_id=cp.round_session(name, 1),
                      root=str(cp.round_root(root, 1))))
        cfg = cp.CampaignConfig(name=name, root=str(root), rounds=a.rounds,
                                target=a.target, fleet=asdict(template))
        cfg_path = root / cp.CONFIG_FILE
        cfg.save(cfg_path)
        pid = _spawn_daemon(root, [sys.executable, "-m", "harness.campaign",
                                   "--config", str(cfg_path)], name)
        print(f"campaign  {name}   {a.rounds} rounds, target {a.target}x")
        print(f"rounds    sessions {cp.round_session(name, 1)} .. "
              f"{cp.round_session(name, a.rounds)} under {root}")
        if template.base:
            print(f"base      {template.base_digest}  {template.base_label}  ({template.base})")
        print(f"pid       {pid}   log {root / 'daemon.log'}")
        print(f"\nwatch:    uv run harness campaign status --root {root}   "
              f"| uv run harness --session {cp.round_session(name, 1)} tui")
        print(f"stop:     uv run harness campaign stop --root {root}   "
              "(or `harness --session <round> stop` to end after that round)")
        return 0

    name = a.session
    root = _campaign_root(a, name) if (name or a.root) else None
    if root is None or not (root / cp.STATE_FILE).is_file():
        print("no campaign found: pass --root, or --session <campaign name>", file=sys.stderr)
        return 1
    state = json.loads((root / cp.STATE_FILE).read_text())
    if a.action == "status":
        if a.json:
            print(json.dumps(state, indent=1, default=str))
        else:
            print(cp.status_text(state))
        return 0
    if a.action == "stop":
        # The marker ends the chain between rounds; the stop command ends
        # the round that is running now.
        (root / cp.STOP_FILE).write_text(str(time.time()))
        cur = state.get("current_session") or ""
        if not cur or state.get("status") != "running":
            print(f"campaign {state.get('name')} is {state.get('status')}; marked stopped")
            return 0
        a.session = cur
        print(f"stopping round session {cur}; no further rounds will start")
        return _send(a, "stop")
    print(f"unknown action {a.action}")
    return 2


# ── read ─────────────────────────────────────────────────────────────────

def daemon_alive(pid: int) -> bool:
    """The snapshot is the daemon's last word, not a heartbeat."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False


def _mark_dead(v):
    """A snapshot from a daemon that no longer exists must not read as
    running. Same rule as the TUI."""
    from dataclasses import replace

    if v.phase in ("running", "stopping", "starting", "paused") and not daemon_alive(v.pid):
        return replace(v, phase="dead",
                       agents=tuple(replace(a, status="lost",
                                            activity="daemon exited; last: " + (a.activity or "-"))
                                    for a in v.agents))
    return v


def _base_of(root: str) -> str:
    """`<digest> <label>` of the stack a fleet compounds on, from its
    fleet.json; empty for a fleet that started from stock."""
    if not root:
        return ""
    try:
        d = json.loads((pathlib.Path(root) / "fleet.json").read_text())
    except (OSError, ValueError):
        return ""
    if not d.get("base"):
        return ""
    return " ".join(t for t in (d.get("base_digest") or "?", d.get("base_label") or "") if t)


def cmd_status(a) -> int:
    v = _resolve(_store(a), a.session)
    if v is None:
        return 1
    if a.json:
        from dataclasses import asdict
        print(json.dumps(asdict(v), indent=1, default=str))
        return 0
    from . import results as rs

    age = time.time() - v.updated_at
    v = _mark_dead(v)
    # Dollars are what Modal bills: the fleet's figure plus the agents' own
    # GPU tool calls that only their directories know about.
    root = _root_for(a, v)
    unrep = rs.unreported_by_agent(root) if root else {}
    print(f"{v.session_id}   {v.phase}   {v.live_agents}/{v.target_agents} agents"
          f"   ${rs.fleet_modal_spend(root, v.cost_usd, unrep):.2f} of ${v.budget_usd:.0f}"
          f"   {v.tokens.total:,} tokens   updated {age:.0f}s ago")
    if v.note:
        print(f"note:  {v.note}")
    base = _base_of(root)
    if base:
        print(f"base:  {base}")
    from .billing import fetch as fetch_bill

    b = fetch_bill(timeout_s=8.0)
    print(b.line() if b else "modal this cycle  -  (Modal not reachable)")
    print(f"evals: {v.evals_running} running, {v.evals_queued} queued, "
          f"{v.evals_completed} done, {v.evals_deduped} deduped, "
          f"{v.gpu_utilisation:.0%} eval-slot utilisation")
    print()
    print(f"{'agent':<7}{'status':<12}{'idea':<26}{'att':>4}{'Δ%':>8}"
          f"{'$/1k':>8}{'rank':>6}{'share':>7}{'$':>8}{'tokens':>11}  activity")
    for ag in v.agents:
        d = "-" if ag.best_delta_pct is None else f"{ag.best_delta_pct:+.1f}"
        bill = "-" if ag.last_bill_per_1k is None else f"{ag.last_bill_per_1k:.2f}"
        rank = ag.last_rank or "-"
        share = "-" if ag.last_share_pct is None else f"{ag.last_share_pct:.2f}%"
        print(f"{ag.agent_id:<7}{ag.status:<12}{ag.idea_title[:24]:<26}"
              f"{ag.attempt:>4}{d:>8}{bill:>8}{rank:>6}{share:>7}"
              f"{ag.cost_usd + unrep.get(ag.agent_id, 0.0):>8.2f}{ag.tokens.total:>11,}"
              f"  {ag.activity[:40]}")
    return 0


def cmd_sessions(a) -> int:
    from . import results as rs

    for v in _store(a).sessions(limit=a.limit):
        root = _root_for(a, v)
        usd = rs.fleet_modal_spend(root, v.cost_usd) if root else v.cost_usd
        base = _base_of(root)
        print(f"{v.session_id:<22}{v.phase:<10}{len(v.agents):>3} agents"
              f"  ${usd:>8.2f}  {v.tokens.total:>12,} tok"
              + (f"  base {base.split()[0]}" if base else ""))
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
    root = getattr(v, "root", "") or _root_for(a, v)
    if root:
        from .inflight import cancel_pending

        done = cancel_pending(root)
        print(f"cancelled {len(done)} GPU call(s) still running for {v.session_id}")
    return 0


def _root_for(a, v) -> str:
    """The fleet root for a session: the snapshot's, else the conventional one."""
    for cand in (getattr(v, "root", None), getattr(a, "root", None),
                 f"agents/{v.session_id}"):
        if cand and pathlib.Path(cand).is_dir():
            return str(cand)
    return ""


def _pid_from_disk(a, v) -> int:
    root = pathlib.Path(a.root or v.root or "")
    f = root / "daemon.pid"
    return int(f.read_text().strip()) if f.is_file() else 0


def _dir_size(path: pathlib.Path) -> int:
    total = 0
    for p in path.rglob("*"):
        with contextlib.suppress(OSError):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
    return total


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def cmd_delete(a) -> int:
    """Wipe a finished fleet: its directory (`agents/<S>`: workspaces, runs,
    profiles, traces) and its rows in the session store. The directory is
    the run's record and the bulk of the space -- a night's profiles alone
    are gigabytes -- so this says what it will remove and how big it is,
    and needs `--yes` or a typed confirmation. A session whose daemon is
    still alive is refused: deleting workspaces under a running agent is
    how the first build run lost its kernel."""
    store = _store(a)
    session_id = a.del_session or a.session
    if not session_id:
        print("delete needs --session S", file=sys.stderr)
        return 2
    v = store.read(session_id)
    root = pathlib.Path(a.root or (v.root if v and v.root else "")
                        or (pathlib.Path.cwd() / "agents" / session_id))
    if v is None and not root.is_dir():
        print(f"no session {session_id!r} in {store.path} and no directory {root}",
              file=sys.stderr)
        return 1
    if v is not None and v.phase in ("running", "starting", "stopping", "paused"):
        if daemon_alive(v.pid):
            print(f"{session_id} is {v.phase} and its daemon (pid {v.pid}) is alive; "
                  f"`harness --session {session_id} stop` first", file=sys.stderr)
            return 1
        print(f"{session_id} reads {v.phase} but its daemon is gone; treating it as finished")
    if root.is_dir() and running_daemon(root):
        print(f"a daemon is running on {root} (pid {running_daemon(root)}); refusing",
              file=sys.stderr)
        return 1
    had_dir = root.is_dir()
    size = _dir_size(root) if had_dir else 0
    n_cmd = 0
    with contextlib.suppress(Exception):
        n_cmd = len(store._c.execute("SELECT id FROM commands WHERE session_id=?",
                                     (session_id,)).fetchall())
    print(f"will remove for session {session_id}:")
    if had_dir:
        print(f"  {root}  ({_human(size)})")
    else:
        print(f"  (no directory at {root})")
    if v is not None:
        print(f"  session row, {len(store.tokens(session_id))} token rows and "
              f"{n_cmd} command rows in {store.path}")
    if not a.yes:
        if not sys.stdin.isatty():
            print("not a terminal; pass --yes to confirm", file=sys.stderr)
            return 1
        if input(f"type {session_id} to confirm: ").strip() != session_id:
            print("not deleted")
            return 1
    if had_dir:
        shutil.rmtree(root)
    removed = store.delete_session(session_id)
    print(f"deleted {root if had_dir else 'no directory'}; rows: "
          + ", ".join(f"{k} {n}" for k, n in removed.items())
          + f"; freed {_human(size)}")
    return 0


def cmd_tool(a) -> int:
    from . import tools

    if a.action == "ablate":
        return _tool_ablate(tools, a)
    return tools.main(a.action, a)


def _tool_ablate(tools, a) -> int:
    """`harness tool ablate --env K=V --tier screen`: the workspace's stack
    swept once per env setting, so an agent can price a flag-gated change
    with the flag on and off. Dispatched here because `tools.ablate` is
    landing separately; until it does, say so instead of a traceback."""
    fn = getattr(tools, "ablate", None)
    if fn is None:
        print("harness tool ablate is not available in this build: "
              "harness.tools.ablate is missing", file=sys.stderr)
        return 2
    env = {}
    for kv in a.env or []:
        if "=" not in kv:
            print(f"--env expects KEY=VAL, got {kv!r}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        env[k.strip()] = v
    rep = fn(workspace=a.workspace, env=env, tier=a.tier)
    if a.json or not isinstance(rep, dict):
        print(json.dumps(rep, indent=1, default=str))
    else:
        for k, v in rep.items():
            print(f"{k}: {v if not isinstance(v, (dict, list)) else json.dumps(v, default=str)}")
    return 0 if not isinstance(rep, dict) or rep.get("ok", True) else 1


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
    if a.action == "seed":
        n = bank.seed(a.arg or "book")
        print(f"seeded {n} records from {a.arg or 'book'} -> {bank.path}")
        return 0
    if a.action == "claim":
        # What the fleet does for an agent, done by hand: the record least
        # like what is live, or with --seed the one most like the seed.
        r = bank.claim(a.agent or "operator", seed=a.seed)
        if r is None:
            print("nothing available to claim")
            return 1
        print(f"{r.id}  {r.title}")
        return 0
    if a.action == "related":
        for r in bank.related(a.arg, k=a.k):
            print(f"  {r.id}  {r.scale:12s} {r.status:9s}  {r.title[:70]}")
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


# ── skills ───────────────────────────────────────────────────────────────

def cmd_skills(a) -> int:
    """The skill bank: facts earlier runs established. Written by the
    manager during a run; `add` is for a person who learned something."""
    from .contracts import Fact
    from .skills import SqliteSkillBank, default_skills_path

    bank = SqliteSkillBank(a.bank or default_skills_path())
    if a.action == "list":
        rows = bank.list(topic=a.topic, status=None if a.all else "active")
        print(f"{bank.path}: {len(rows)} facts")
        for f in rows:
            flag = "" if f.status == "active" else f" [{f.status} -> {f.superseded_by}]"
            print(f"  {f.id}  {f.topic:18s} {f.confidence:.1f}  {f.claim[:80]}{flag}")
        return 0
    if a.action == "show":
        f = bank.get(a.arg)
        if f is None:
            print(f"no fact {a.arg}")
            return 1
        from dataclasses import asdict
        print(json.dumps(asdict(f), indent=1))
        return 0
    if a.action == "add":
        fid, losers = bank.add(Fact(claim=a.arg, topic=a.topic or "general",
                                    evidence=a.evidence, source="human",
                                    confidence=a.confidence))
        print(f"added {fid}" + (f"; superseded {', '.join(losers)}" if losers else ""))
        return 0
    if a.action == "retract":
        bank.retract(a.arg)
        print(f"retracted {a.arg}")
        return 0
    if a.action == "render":
        print(bank.render(query=a.arg) or "(no facts)")
        return 0
    print(f"unknown action {a.action}")
    return 2


# ── results and questions ────────────────────────────────────────────────

def _root_of(a) -> str:
    """The fleet root: `--root`, else the named or latest session's root."""
    if getattr(a, "root", ""):
        return a.root
    v = _store(a).read(a.session or "")
    if v is None or not v.root:
        raise SystemExit("no fleet root: pass --root or --session")
    return v.root


def cmd_results(a) -> int:
    from . import results as rs

    root = _root_of(a)
    rows = rs.leaderboard(root)
    if a.json:
        from dataclasses import asdict
        print(json.dumps([asdict(r) for r in rows], indent=1))
        return 0
    print(rs.summary_text(root, k=a.k))
    if a.diff and rows:
        best = next((r for r in rows if r.delta_pct is not None), rows[0])
        print(f"\ndiff for {best.experiment_id}:\n")
        print(rs.diff_for(root, best) or "(no diff recorded)")
    return 0


def cmd_calls(a) -> int:
    """Every model call an agent made, with per-message tokens and tool use:
    the most granular view there is of where an agent-hour went."""
    root = pathlib.Path(_root_of(a))
    agents = [a.agent] if a.agent else sorted(p.name for p in root.iterdir()
                                              if (p / "calls").is_dir())
    for ag in agents:
        d = root / ag / "calls"
        if not d.is_dir():
            continue
        print(ag)
        for f in sorted(d.glob("*.jsonl")):
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            msgs = [r for r in rows if r.get("type") == "assistant"]
            tot = {k: sum(r.get(k, 0) for r in msgs) for k in ("input", "output", "cache_read", "cache_write")}
            tools = {}
            for r in msgs:
                for t in r.get("tools") or ():
                    tools[t] = tools.get(t, 0) + 1
            res = next((r for r in rows if r.get("type") == "result"), {})
            span = (rows[-1]["ts"] - rows[0]["ts"]) / 60 if rows else 0
            print(f"  {f.stem:<22} {len(msgs):>4} msgs {span:>6.1f} min  "
                  f"in {tot['input']:>7,} out {tot['output']:>7,} cache {tot['cache_read']:>10,}"
                  f"  turns {res.get('num_turns', '-')}  tools: "
                  + ", ".join(f"{k}x{v}" for k, v in sorted(tools.items(), key=lambda kv: -kv[1])[:6]))
            if a.verbose:
                for r in rows:
                    if r.get("type") == "assistant":
                        print(f"      +{r['since_prev_s']:>6.1f}s  out {r['output']:>5} cache {r['cache_read']:>8}"
                              f"  {', '.join(r.get('tools') or []) or 'text'}")
    return 0


def cmd_spend(a) -> int:
    """Every dollar a fleet's directory can account for: evaluations from
    their `result.json`, tool calls from their workbench results, and the
    calls that never got a result. The fleet's own total counted only the
    first until 2026-09-03; this reads the disk, so it works for old runs."""
    import json

    from .agent.evaluator import sweep_cost

    root = pathlib.Path(_root_of(a))
    if not root.is_dir():
        print(f"no fleet directory at {root}", file=sys.stderr)
        return 1
    agents = sorted(p for p in root.glob("a*") if p.is_dir() and p.name[1:].isdigit())
    tot_ev = tot_wb = 0.0
    orphans = 0
    print(f"{'agent':<6} {'evals':>5} {'$ evals':>9} {'tools':>5} {'$ tools':>9} {'no result':>9}")
    for ag in agents:
        ev = wb = 0.0
        n_ev = n_wb = n_or = 0
        for r in ag.glob("runs/*/sweep.json"):
            with contextlib.suppress(Exception):
                ev += sweep_cost(json.loads(r.read_text()))
                n_ev += 1
        for r in ag.glob("workbench-*/result.json"):
            with contextlib.suppress(Exception):
                wb += float(json.loads(r.read_text()).get("cost_usd") or 0.0)
                n_wb += 1
        for c in list(ag.glob("workbench-*/call_id")) + list(ag.glob("runs/*/call_id")):
            if not (c.parent / "result.json").is_file():
                n_or += 1
        tot_ev += ev
        tot_wb += wb
        orphans += n_or
        print(f"{ag.name:<6} {n_ev:>5} {ev:>9.2f} {n_wb:>5} {wb:>9.2f} {n_or:>9}")
    print(f"{'total':<6} {'':>5} {tot_ev:>9.2f} {'':>5} {tot_wb:>9.2f} {orphans:>9}")
    print(f"\n${tot_ev + tot_wb:.2f} accounted for on disk. Calls with no result ran "
          "to completion unwatched (killed tools, killed daemons) and billed "
          "for their full length; that money is not in this total.")
    return 0


def cmd_paper(a) -> int:
    """List the run's papers, or compile one agent's .tex again."""
    from .paper import compile_tex, find_papers

    root = pathlib.Path(_root_of(a))
    if a.compile:
        tex = pathlib.Path(a.compile)
        out = compile_tex(tex)
        print(out or f"compile failed; see {tex.parent / 'paper.log'}")
        return 0 if out else 1
    # Listing compiles: a .tex left by a daemon without tectonic, or one
    # written after the PDF, becomes a PDF here rather than a surprise later.
    papers = find_papers(root, compile=True)
    if not papers:
        print("no papers yet (written at the end of an idea that reached a full sweep)")
        return 0
    for idea_id, p in sorted(papers.items()):
        note = "" if p.suffix == ".pdf" else f"   (not compiled; see {p.parent / 'paper.log'})"
        print(f"  {idea_id:<20} {p}{note}")
    return 0


def cmd_timeline(a) -> int:
    from . import timeline as tl

    root = _root_of(a)
    if a.html:
        pathlib.Path(a.html).write_text(tl.render_html(root))
        print(f"wrote {a.html}")
        return 0
    if a.md:
        print(f"wrote {tl.write_markdown(root)}")
        return 0
    print(tl.render_text(root, agent_id=a.agent))
    return 0


def cmd_ask(a) -> int:
    """One question about a run, answered from its data by Claude over the
    API. `--model` defaults to Claude Fable 5.1."""
    from .ask import Asker

    asker = Asker(_root_of(a), model=a.model)
    print(asker.ask(a.question))
    u = asker.last_usage
    if u:
        print(f"\n[{u.get('input', 0):,} in, {u.get('output', 0):,} out, "
              f"{u.get('cache_read', 0):,} cached]", file=sys.stderr)
    return 0


# ── traces ───────────────────────────────────────────────────────────────

def cmd_traces(a) -> int:
    from . import traces

    if a.action == "list":
        found = traces.find(a.root or None, session_id=a.session,
                            agent_id=a.agent, outcome=a.outcome, min_turns=a.min_turns)
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

def _add_start_options(s) -> None:
    """The `start` options; `campaign start` takes every one of them."""
    s.add_argument("--agents", type=int, default=4, help="agents running at once (default 4)")
    s.add_argument("--evals", type=int, default=2, help="concurrent GPU evaluations (default 2)")
    s.add_argument("--budget", type=float, default=200.0, help="$ ceiling for the fleet (default 200)")
    s.add_argument("--agent-budget", dest="agent_budget", type=float, default=40.0,
                   help="$ ceiling for one agent on one idea (default 40)")
    s.add_argument("--max-attempts", dest="max_attempts", type=int, default=6,
                   help="diffs one agent may evaluate on one idea (default 6)")
    s.add_argument("--stall-minutes", dest="stall_minutes", type=float, default=40.0,
                   help="cut a model call that has produced nothing for this long and "
                        "restart the attempt (default 20; 0 never)")
    s.add_argument("--no-auto-ablate", dest="auto_ablate", action="store_false",
                   help="do not price a replicated win with its ablation.env kill "
                        "switch set (default: one screen per replicated win)")
    s.add_argument("--model", default="sonnet", help="claude model: sonnet | opus (default sonnet)")
    s.add_argument("--gpu", default="H100", help="GPU every evaluation rents (default H100)")
    s.add_argument("--n-gpu", dest="n_gpu", type=int, default=1,
                   help="GPUs per evaluation")
    s.add_argument("--root", default="",
                   help="where agent workspaces go (default: agents/<session>)")
    s.add_argument("--force", action="store_true",
                   help="start even if a daemon for this root is alive")
    s.add_argument("--bank", nargs="?", const="__default__", default="",
                   help="claim ideas from the bank (default path when given no value)")
    s.add_argument("--profile-level", dest="profile_level", type=int, default=12,
                   help="capture a GPU profile at this concurrency on full sweeps "
                        "and serve it to agents over MCP (default 12; 0 = none)")
    s.add_argument("--manager", action="store_true",
                   help="review outcomes and stash reusable tools under <root>/tools/")
    s.add_argument("--seed", action="append", help="a starting hypothesis; repeatable")
    s.add_argument("--baseline", default="",
                   help='stock (or the base), measured on this grid; required. JSON: '
                        '{"bill_per_1k": 12.23, "quality": {"gsm8k": 0.69}, '
                        '"screen": {"bill_per_1k": 17.30}}')
    s.add_argument("--base", default="",
                   help="compound: every agent starts from this saved stack instead of "
                        "stock (a run directory, a stack.json, or a mirrored sglang/ "
                        "tree); --baseline is then that stack's own report")
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="fake the GPU evaluations (saves dollars, still runs "
                        "real Claude Code agents)")
    s.add_argument("--allow-stale-runner", dest="allow_stale_runner", action="store_true",
                   help="start even if the deployed Modal runner is behind this checkout")
    s.add_argument("--fake-agents", dest="fake_agents", action="store_true",
                   help="fake the agents too (saves subscription usage); "
                        "with --dry-run this exercises the whole fleet for free")
    s.add_argument("--note", default="", help="free text, recorded with the session")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description=__doc__.split("\n")[0])
    ap.add_argument("--store", default="",
                    help="session database (default ~/.auto-inference/sessions.db; "
                         "HARNESS_HOME moves it)")
    ap.add_argument("--session", default="", help="session id (default: most recent)")
    ap.add_argument("--wait", type=float, default=5.0,
                    help="seconds to wait for stop/scale/agent to be acknowledged")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="launch a fleet, detached")
    _add_start_options(s)
    s.set_defaults(fn=cmd_start)

    cp = sub.add_parser("campaign", help="fleets in sequence, each on the last round's win")
    cp.add_argument("action", choices=["start", "status", "stop"])
    cp.add_argument("--rounds", type=int, default=3, help="start: fleets to chain (default 3)")
    cp.add_argument("--target", type=float, default=2.0,
                    help="start: stop once bill_per_1k has improved this many times "
                         "over round one's baseline (default 2.0 = halved)")
    cp.add_argument("--json", action="store_true", help="status: machine-readable")
    _add_start_options(cp)
    cp.set_defaults(fn=cmd_campaign)

    st = sub.add_parser("status", help="one-shot snapshot")
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=cmd_status)

    ls = sub.add_parser("sessions", help="list recent sessions")
    ls.add_argument("--limit", type=int, default=10)
    ls.set_defaults(fn=cmd_sessions)

    sub.add_parser("tui", help="live dashboard").set_defaults(fn=cmd_tui)

    tl = sub.add_parser("tool", help="tools for agents (and for reading runs)")
    tl.add_argument("action", choices=["recall", "preflight", "roofline",
                                       "gpu-run", "equivalence", "ncu", "ablate"])
    tl.add_argument("intent", nargs="?", default="", metavar="ARG",
                    help="recall: what you are about to do; "
                         "gpu-run: the script to run")
    tl.add_argument("--kernel", default="", help="ncu: regex on kernel names to profile")
    tl.add_argument("--env", action="append", metavar="KEY=VAL",
                    help="ablate: a server environment variable; repeatable")
    tl.add_argument("--tier", default="screen", choices=["screen", "full"],
                    help="ablate: which sweep to price at (default screen)")
    tl.add_argument("--workspace", default=".",
                    help="the agent directory (preflight, gpu-run, equivalence, ablate)")
    tl.add_argument("--timeout", type=int, default=0,
                    help="gpu-run/equivalence: seconds the script itself gets; "
                         "0 takes the tool's own default (600s, 1800s), which "
                         "already allows for a 3-5 minute engine load")
    tl.add_argument("--context", type=int, default=20583,
                    help="roofline: input tokens per sequence")
    tl.add_argument("--batch", type=int, default=12,
                    help="roofline: sequences decoding together")
    tl.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8",
                    help="roofline: the model being served")
    tl.add_argument("--gpu", default="H100", help="roofline: the GPU")
    tl.add_argument("--n-gpu", dest="n_gpu", type=int, default=1,
                    help="roofline: GPUs it is served on")
    tl.add_argument("--root", default="", help="fleet root, for recall")
    tl.add_argument("-k", type=int, default=8, help="recall: hits to retrieve")
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(fn=cmd_tool)

    sk = sub.add_parser("skills", help="the skill bank: facts earlier runs established")
    sk.add_argument("action", choices=["list", "show", "add", "retract", "render"])
    sk.add_argument("arg", nargs="?", default="", help="id | claim | query")
    sk.add_argument("--bank", default="",
                    help="database path (default: ~/.auto-inference/skills.db, "
                         "shared by every run on this machine)")
    sk.add_argument("--topic", default="")
    sk.add_argument("--evidence", default="")
    sk.add_argument("--confidence", type=float, default=0.7)
    sk.add_argument("--all", action="store_true", help="list: include superseded")
    sk.set_defaults(fn=cmd_skills)

    rz = sub.add_parser("results", help="what the run found, best first")
    rz.add_argument("--root", default="", help="fleet root (default: the session's)")
    rz.add_argument("-k", type=int, default=12)
    rz.add_argument("--diff", action="store_true", help="print the best result's diff")
    rz.add_argument("--json", action="store_true")
    rz.set_defaults(fn=cmd_results)

    cl = sub.add_parser("calls", help="every model call, with per-message tokens and tools")
    cl.add_argument("--root", default="", help="fleet root (default: the session's)")
    cl.add_argument("--agent", default="", help="one agent only")
    cl.add_argument("-v", "--verbose", action="store_true", help="one line per message")
    cl.set_defaults(fn=cmd_calls)

    sp = sub.add_parser("spend", help="every dollar the fleet directory accounts for")
    sp.add_argument("--root", default="", help="fleet root (default: the session's)")
    sp.set_defaults(fn=cmd_spend)
    pp = sub.add_parser("paper", help="the write-up PDFs, one per idea")
    pp.add_argument("--root", default="", help="fleet root (default: the session's)")
    pp.add_argument("--compile", default="", help="compile this PAPER.tex again")
    pp.set_defaults(fn=cmd_paper)

    tl = sub.add_parser("timeline", help="the run as a readable timeline")
    tl.add_argument("--root", default="", help="fleet root (default: the session's)")
    tl.add_argument("--agent", default="", help="one agent only")
    tl.add_argument("--html", default="", help="write a single-file Gantt page here")
    tl.add_argument("--md", action="store_true", help="rewrite <root>/timeline.md")
    tl.set_defaults(fn=cmd_timeline)

    ask = sub.add_parser("ask", help="ask Claude a question about a run")
    ask.add_argument("question")
    ask.add_argument("--root", default="", help="fleet root (default: the session's)")
    ask.add_argument("--model", default="claude-fable-5-1")
    ask.set_defaults(fn=cmd_ask)

    ideas = sub.add_parser("ideas", help="the idea bank: fill it, inspect it")
    ideas.add_argument("action", choices=["list", "show", "search", "import", "seed",
                                          "claim", "related", "release",
                                          "extract-pdf", "arxiv"])
    ideas.add_argument("arg", nargs="?", default="",
                       help="id | query | jsonl path | pdf path | seed set (default book)")
    ideas.add_argument("--agent", default="",
                       help="claim: who holds the record (default operator)")
    ideas.add_argument("--seed", default="",
                       help="claim: steer toward this text instead of away from what is live")
    ideas.add_argument("--bank", default="",
                       help="database path (default: ~/.auto-inference/ideas.db, "
                            "shared by every run on this machine)")
    ideas.add_argument("--status", default="",
                       help="list: available | claimed | tried | retired")
    ideas.add_argument("--scale", default="",
                       help="list: kernel | architecture | memory | scheduler | "
                            "parallelism | numerics | other")
    ideas.add_argument("--source", default="", help="import/extract-pdf: source label")
    ideas.add_argument("--model", default="opus", help="extract-pdf/arxiv: claude model")
    ideas.add_argument("--pages", type=int, default=20, help="extract-pdf: window size")
    ideas.add_argument("-k", type=int, default=25,
                       help="search/related: hits; arxiv: per query")
    ideas.add_argument("--json", action="store_true")
    ideas.set_defaults(fn=cmd_ideas)

    tr = sub.add_parser("traces", help="list, read, or export agent traces")
    tr.add_argument("action", choices=["list", "show", "export"])
    tr.add_argument("trace_id", nargs="?", default="")
    tr.add_argument("--root", default="", help="fleet root (default: ./agents)")
    tr.add_argument("--agent", default="", help="list: only this agent")
    tr.add_argument("--outcome", default="", help="list: won | lost | neutral | diverged | error")
    tr.add_argument("--min-turns", dest="min_turns", type=int, default=4,
                    help="list: hide traces shorter than this (an API outage leaves "
                         "thousands of 3-line error traces); 0 shows all")
    tr.add_argument("--out", default="trace-export", help="export destination")
    tr.add_argument("--kind", default="", help="comma-separated turn kinds")
    tr.add_argument("--query", default="", help="substring filter")
    tr.add_argument("--limit", type=int, default=200)
    tr.add_argument("--width", type=int, default=140)
    tr.add_argument("--full", action="store_true", help="include the data blob")
    tr.set_defaults(fn=cmd_traces)
    sub.add_parser("stop", help="stop the fleet: model calls cancelled now, paid sweeps finish "
                                "(`kill <agent>` also drops the sweep)").set_defaults(fn=cmd_stop)

    k = sub.add_parser("kill", help="flat kill: everything, now")
    k.add_argument("--root", default="")
    k.set_defaults(fn=cmd_kill)

    de = sub.add_parser("delete", help="wipe a finished fleet's directory and store rows")
    # Its own --session so `harness delete --session S` reads naturally; the
    # global one (before the subcommand) still works.
    de.add_argument("--session", dest="del_session", default="", metavar="S",
                    help="session id")
    de.add_argument("--root", default="", help="fleet root (default: the session's)")
    de.add_argument("--yes", action="store_true", help="skip the confirmation")
    de.set_defaults(fn=cmd_delete)

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
