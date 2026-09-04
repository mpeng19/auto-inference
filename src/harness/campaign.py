"""The campaign: fleets in sequence, each building on the last one's win.

`harness start` runs one fleet from one base. The owner's goal is the chain:
stock, then one improvement, then a new -- ideally orthogonal -- idea on top
of it, and so on until the bill is two or three times smaller. Doing that by
hand means reading each fleet's results the next morning, picking a run
directory, working out its baseline at both tiers, and typing the next
`start`. This module is that morning routine as a daemon.

    harness campaign start --rounds 4 --target 2.0 <every start option>
    harness campaign status
    harness campaign stop

Round `n` is an ordinary fleet, session `<name>-r<n>` under
`<root>/r<n>/`, run in this process by `daemon.run` and controllable with
every existing command. When it ends -- budget, bank, wall clock, or the
operator -- the driver reads its leaderboard, takes the best **publishable**
result (`results.publishable`: a replicated win, gates held, ablation
explains it), or, saying so, the best replicated win when nothing is
publishable, and makes that result's run directory the next round's
`--base`. The next `--baseline` is the winner's own report: its full-tier
bill, its screen-tier bill from the screen attempt of the same stack (scaled
by the fleet's screen/full ratio when there was none), and its accuracy per
suite. Every idea the earlier rounds tried travels as `avoid` texts, and the
base's own idea seeds the claims (`FleetSpec.base_seed`), which is where
orthogonality comes from.

The chain ends when the cumulative gain against round one's baseline reaches
`target` (2.0 means the bill halved), the rounds run out, a round produces
no replicated win, or the operator stops it: `harness stop` on the current
round's session ends the campaign after that round, `harness campaign stop`
ends it now. `<root>/campaign.json` is rewritten after every round and is
what `harness campaign status` prints.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field, replace

from .daemon import FleetConfig

STATE_FILE = "campaign.json"
CONFIG_FILE = "campaign-config.json"
STOP_FILE = "STOP"


@dataclass
class CampaignConfig:
    """What `harness campaign start` writes and the driver reads."""
    name: str
    root: str
    rounds: int = 3
    target: float = 2.0
    fleet: dict = field(default_factory=dict)     # FleetConfig, as `asdict`

    @classmethod
    def load(cls, path) -> CampaignConfig:
        d = json.loads(pathlib.Path(path).read_text())
        return cls(name=d["name"], root=d["root"], rounds=int(d.get("rounds", 3)),
                   target=float(d.get("target", 2.0)), fleet=dict(d.get("fleet") or {}))

    def save(self, path) -> None:
        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=1, default=str))

    def template(self) -> FleetConfig:
        return _fleet_from_dict(self.fleet)


def _fleet_from_dict(d: dict) -> FleetConfig:
    from dataclasses import fields

    known = {f.name for f in fields(FleetConfig)}
    d = {k: v for k, v in d.items() if k in known}
    for k in ("levels", "screen_levels", "seeds", "avoid"):
        if k in d:
            d[k] = tuple(d[k] or ())
    return FleetConfig(**d)


def round_session(name: str, n: int) -> str:
    return f"{name}-r{n}"


def round_root(root: str | pathlib.Path, n: int) -> pathlib.Path:
    return pathlib.Path(root) / f"r{n}"


# ── reading a finished round ─────────────────────────────────────────────

def pick_best(round_root: str | pathlib.Path):
    """(result, fell_back): the best publishable result of a round, else
    the best replicated full-tier win with `fell_back` set, else None."""
    from . import results as rs

    rows = rs.leaderboard(round_root)
    pub = [r for r in rows if r.publishable and r.bill_per_1k is not None]
    if pub:
        return pub[0], False
    rep = [r for r in rows if r.verdict == "win" and r.tier == "full"
           and r.bill_per_1k is not None and r.evidence.get("replicated")]
    if rep:
        return rep[0], True
    return None, False


def run_dir_for(round_root: str | pathlib.Path, agent_id: str,
                stack_digest: str) -> pathlib.Path | None:
    """The run directory whose `stack.json` is this digest: what the next
    round loads as its base. The first measurement is preferred over the
    replicate (`-repN`), which holds the same stack."""
    from simulator import InferenceStack

    agent = pathlib.Path(round_root) / agent_id
    dirs = sorted(agent.glob("runs/attempt-*"), key=lambda d: ("-rep" in d.name, d.name))
    for d in dirs:
        try:
            st = InferenceStack.from_dict(json.loads((d / "stack.json").read_text()))
        except Exception:
            continue
        if st.digest == stack_digest:
            return d
    return None


def screen_bill_for(round_root: str | pathlib.Path, stack_digest: str,
                    full_bill: float | None, baseline: dict) -> float | None:
    """The winner's price at screen tier: the screen attempt of the same
    stack when the traces hold one, else its full bill scaled by the
    fleet's own screen/full ratio -- the screen carries warm-up a full
    sweep amortises, and a screen judged against a full number never
    promotes (see `loop._delta`)."""
    from . import traces as tr

    for tf in tr.find(round_root):
        try:
            for t in tr.read(tf.path, kinds=("eval_result",)):
                d = t.get("data") or {}
                if (t.get("name") == stack_digest and d.get("tier") == "screen"
                        and isinstance(d.get("bill_per_1k"), (int, float))):
                    return float(d["bill_per_1k"])
        except Exception:
            continue
    full0 = baseline.get("bill_per_1k")
    screen0 = (baseline.get("screen") or {}).get("bill_per_1k") \
        if isinstance(baseline.get("screen"), dict) else None
    if (isinstance(full_bill, (int, float)) and isinstance(full0, (int, float)) and full0
            and isinstance(screen0, (int, float))):
        return round(full_bill * screen0 / full0, 4)
    return None


def next_baseline(round_root: str | pathlib.Path, res, prev: dict) -> dict:
    """The next round's `--baseline`, from the winner's own report."""
    quality = {}
    for row in res.metrics.get("quality") or ():
        if isinstance(row, dict) and row.get("suite") and row.get("accuracy") is not None:
            quality[str(row["suite"])] = row["accuracy"]
    out = {"bill_per_1k": res.bill_per_1k,
           "quality": quality or dict(prev.get("quality") or {})}
    screen = screen_bill_for(round_root, res.stack_digest, res.bill_per_1k, prev)
    if screen is not None:
        out["screen"] = {"bill_per_1k": screen}
    return out


def tried_ideas(round_root: str | pathlib.Path) -> list[dict]:
    """Every idea a round tried, from its `summary.json` (memory.db as the
    fallback for a round that died before writing one)."""
    root = pathlib.Path(round_root)
    try:
        doc = json.loads((root / "summary.json").read_text())
        out = [{"idea_id": o.get("idea_id", ""), "title": o.get("title", ""),
                "hypothesis": o.get("hypothesis", ""), "stop": o.get("stop", "")}
               for o in doc.get("outcomes") or ()]
        if out:
            return out
    except (OSError, ValueError):
        pass
    db = root / "memory.db"
    if not db.is_file():
        return []
    import sqlite3

    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = c.execute("SELECT DISTINCT idea_id, hypothesis FROM experiments").fetchall()
        c.close()
    except sqlite3.Error:
        return []
    return [{"idea_id": r[0] or "", "title": "", "hypothesis": r[1] or "", "stop": ""}
            for r in rows]


# ── the driver ───────────────────────────────────────────────────────────

def _write_state(root: pathlib.Path, state: dict) -> pathlib.Path:
    state["updated_at"] = time.time()
    p = root / STATE_FILE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str))
    os.replace(tmp, p)
    return p


def drive(cfg: CampaignConfig, *, run_round=None, log=print) -> dict:
    """Run the rounds. `run_round(FleetConfig) -> reason` is `daemon.run`
    unless a test hands in a fake; the state returned is `campaign.json`."""
    if run_round is None:
        from .daemon import run as run_round
    root = pathlib.Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    template = cfg.template()
    base = template.base
    baseline = dict(template.baseline)
    avoid = list(template.avoid)
    bill0 = baseline.get("bill_per_1k")
    state = {"name": cfg.name, "root": str(root), "target": cfg.target,
             "rounds_planned": cfg.rounds, "status": "running",
             "started_at": time.time(), "baseline0": dict(baseline),
             "base0": base, "current_session": "", "rounds": [], "chain": [],
             "cumulative_gain": None, "stop_reason": ""}
    _write_state(root, state)
    stop_reason = ""
    for n in range(1, cfg.rounds + 1):
        if (root / STOP_FILE).exists():
            stop_reason = "campaign stop" + (f" after round {n - 1}" if n > 1 else "")
            break
        session, rroot = round_session(cfg.name, n), round_root(root, n)
        rroot.mkdir(parents=True, exist_ok=True)
        rcfg = replace(template, session_id=session, root=str(rroot),
                       baseline=dict(baseline), avoid=tuple(avoid))
        if base:
            rcfg = rcfg.with_base(base)
        rcfg.save(rroot / "fleet.json")
        # The round's directory names the process running it, so
        # `harness start` and `delete` on it refuse the way they would for
        # a fleet daemon.
        (rroot / "daemon.pid").write_text(str(os.getpid()))
        state["current_session"] = session
        _write_state(root, state)
        log(f"campaign {cfg.name}: round {n}/{cfg.rounds} as {session}"
            + (f" on base {rcfg.base_digest}" if rcfg.base_digest else " from stock"), flush=True)
        why = run_round(rcfg)

        entry = {"n": n, "session": session, "root": str(rroot),
                 "base": base, "base_digest": rcfg.base_digest, "base_idea": rcfg.base_idea,
                 "baseline": dict(baseline), "ended": why,
                 "tried": tried_ideas(rroot), "best": None, "gain_cumulative": None}
        for t in entry["tried"]:
            for text in (t.get("hypothesis"), t.get("idea_id")):
                if text and text not in avoid:
                    avoid.append(text)
        best, fell_back = pick_best(rroot)
        if best is None:
            state["rounds"].append(entry)
            stop_reason = f"round {n} produced no replicated win"
            break
        run_dir = run_dir_for(rroot, best.agent_id, best.stack_digest)
        gain = (round(bill0 / best.bill_per_1k, 4)
                if isinstance(bill0, (int, float)) and best.bill_per_1k else None)
        entry["best"] = {
            "experiment_id": best.experiment_id, "agent_id": best.agent_id,
            "hypothesis": best.hypothesis, "bill_per_1k": best.bill_per_1k,
            "delta_pct": best.delta_pct, "stack_digest": best.stack_digest,
            "publishable": best.publishable, "pub": best.pub, "fell_back": fell_back,
            "note": ("nothing publishable this round; fell back to the best "
                     "replicated win" if fell_back else ""),
            "run_dir": str(run_dir) if run_dir else ""}
        entry["gain_cumulative"] = gain
        state["chain"].append({"round": n, "hypothesis": best.hypothesis,
                               "experiment_id": best.experiment_id,
                               "bill_per_1k": best.bill_per_1k, "fell_back": fell_back})
        state["cumulative_gain"] = gain
        state["rounds"].append(entry)
        _write_state(root, state)
        log(f"campaign {cfg.name}: round {n} best ${best.bill_per_1k}/1k "
            f"({best.pub}{', fallback' if fell_back else ''}); gain {gain}x", flush=True)
        if why == "operator":
            stop_reason = f"operator stopped round {n}"
            break
        if gain is not None and gain >= cfg.target:
            stop_reason = f"target {cfg.target}x reached ({gain}x)"
            break
        if run_dir is None:
            stop_reason = f"round {n}: no run directory holds the winning stack"
            break
        if n == cfg.rounds:
            stop_reason = "rounds exhausted"
            break
        base = str(run_dir)
        baseline = next_baseline(rroot, best, baseline)
    state["status"] = "stopped"
    state["stop_reason"] = stop_reason or "rounds exhausted"
    state["current_session"] = ""
    _write_state(root, state)
    log(f"campaign {cfg.name}: {state['stop_reason']}", flush=True)
    return state


def status_text(state: dict) -> str:
    """`harness campaign status`."""
    out = [f"{state.get('name')}   {state.get('status')}   "
           f"{len(state.get('rounds') or ())}/{state.get('rounds_planned')} rounds   "
           f"target {state.get('target')}x   gain "
           + (f"{state['cumulative_gain']}x" if state.get("cumulative_gain") else "-")]
    b0 = state.get("baseline0") or {}
    if b0.get("bill_per_1k") is not None:
        out.append(f"baseline0  ${b0['bill_per_1k']}/1k"
                   + (f"   base {state['base0']}" if state.get("base0") else "   stock"))
    if state.get("current_session"):
        out.append(f"running    {state['current_session']}")
    for r in state.get("rounds") or ():
        best = r.get("best")
        line = (f"  r{r['n']}  {r['session']:<18} ended {r.get('ended') or '-':<9} "
                f"base {r.get('base_digest') or 'stock':<12} "
                f"baseline ${(r.get('baseline') or {}).get('bill_per_1k')}/1k  ")
        if best:
            line += (f"best ${best['bill_per_1k']}/1k ({best['pub']}"
                     + (", fallback" if best.get("fell_back") else "") + f")  "
                     f"gain {r.get('gain_cumulative')}x  {best['hypothesis'][:60]}")
        else:
            line += f"no replicated win ({len(r.get('tried') or ())} ideas tried)"
        out.append(line)
        if best and best.get("note"):
            out.append(f"      {best['note']}")
    if state.get("stop_reason"):
        out.append(f"stopped    {state['stop_reason']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="harness-campaign")
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    drive(CampaignConfig.load(a.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
