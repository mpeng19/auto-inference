"""The fleet process. Started detached by `harness start`, watched by the TUI.

Separate from the CLI on purpose. A fleet runs for hours and must survive the
terminal that launched it -- the same reason `simulator` spawns sweeps rather
than calling them -- so `start` writes a config, spawns this, and returns a
session id. Everything after that is done through the session store.

Assembling the services is all this file does. That assembly is the only place
that knows which implementation of each contract is in use, which is what makes
swapping one a single-line change.
"""
from __future__ import annotations

import json
import pathlib
import signal
import sys
import time
from dataclasses import asdict, dataclass, field

from .agent import ClaudeCodeProposer, IterativeAgent, SimulatorEvaluator, Workspace
from .context import JsonlContext
from .contracts import AgentBudget, FleetBudget, FleetSpec, Idea
from .ideas import SqliteIdeaBank
from .memory import SqliteMemory
from .orchestration import EvalBroker, Fleet
from .session import SqliteSessionStore, default_store_path
from .skills import SqliteSkillBank, default_skills_path


@dataclass
class FleetConfig:
    """Everything a fleet needs, serialisable so a daemon can be restarted."""
    session_id: str = ""
    root: str = ""                  # per-agent working directories
    agents: int = 4
    eval_capacity: int = 2
    budget_usd: float = 200.0
    max_wall_s: float = 24 * 3600
    agent_max_attempts: int = 6
    agent_max_usd: float = 40.0
    patience: int = 3
    screen_first: bool = True
    model: str = "sonnet"           # `claude --model`; prefer opus/sonnet here
    seed_model: str = ""
    # simulator settings for a real evaluation
    gpu: str = "H100"
    n_gpu: int = 1
    sim_model: str = "Qwen/Qwen3.8-27B-FP8"
    levels: tuple[int, ...] = (4, 8, 12, 16, 24)
    screen_levels: tuple[int, ...] = (8, 12)
    seconds_per_level: float = 120.0
    screen_seconds: float = 60.0
    baseline: dict = field(default_factory=dict)
    seeds: tuple[str, ...] = ()     # free-text hypotheses to start from
    # Where ideas come from once the seeds run out. Empty: the agents seed
    # themselves, which produced one-line knob tweaks. A path: records are
    # claimed from the bank, one per agent, least similar first.
    bank: str = ""
    # "tune" asks for the smallest edit; "build" hands over a mechanism and
    # expects a kernel-scale change with a design note and workbench checks.
    mode: str = "tune"
    # A reviewer that turns what agents keep re-deriving into shared tools
    # under <root>/tools/. A few model calls a night.
    manager: bool = False
    # Capture a GPU profile at this level on full sweeps (0 = none) and hand
    # it to the agent as tracedb tools. 12 is the priced level on stock.
    profile_level: int = 12
    # Two separate fakes, because they cost different things. `dry_run` skips
    # the GPU (dollars); `fake_agents` skips Claude Code (subscription usage).
    # A flag named "dry run" that still spawns ten real agents is a trap.
    dry_run: bool = False
    fake_agents: bool = False
    note: str = ""

    @classmethod
    def load(cls, path) -> FleetConfig:
        d = json.loads(pathlib.Path(path).read_text())
        d["levels"] = tuple(d.get("levels") or (4, 8, 12, 16, 24))
        d["screen_levels"] = tuple(d.get("screen_levels") or (8, 12))
        d["seeds"] = tuple(d.get("seeds") or ())
        return cls(**d)

    def save(self, path) -> None:
        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=1, default=str))


class _ScriptedProposer:
    """Stands in for Claude Code so the fleet can be exercised for free."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._n = 0

    def seed(self, live_ideas, brief):
        self._n += 1
        topics = ["prefill chunking", "radix eviction", "batch admission",
                  "decode scheduling", "kv layout", "queue ordering",
                  "cuda graph capture", "token budgeting"]
        # Offset by agent so a fleet of these does not all propose the same
        # thing and spend its time being rejected for duplication.
        idx = (abs(hash(self.agent_id)) + self._n) % len(topics)
        topic = topics[idx]
        return Idea(title=f"{topic} {self._n}",
                    hypothesis=f"changing {topic} lowers cost per output token",
                    targets=("srt/managers/schedule_policy.py",))

    def edit(self, ws, idea, brief, attempt, history):
        ws.materialise("srt/managers/schedule_policy.py")
        cur = ws.read("srt/managers/schedule_policy.py")
        ws.edit("srt/managers/schedule_policy.py",
                f"# {self.agent_id} {idea.id} attempt {attempt}\n" + cur)
        return f"scripted edit for {idea.title}"

    def study(self, ws, idea, brief, history):
        return "scripted study"


def _fake_runner(req):
    """Deterministic stand-in so the whole pipeline can be exercised free."""
    import random
    rng = random.Random(req.stack.digest)
    time.sleep(2.0 if req.tier == "screen" else 5.0)
    delta = rng.uniform(-12, 8)
    base = 12.23
    return True, {"bill_per_1k": round(base * (1 + delta / 100), 3),
                  "effective_in_per_m": 0.0294, "out_per_m": 5.60,
                  "n_star": rng.choice([8, 12, 16]), "batch": 5.0,
                  "hit_rate": 0.75, "gpu_s_per_request": 7.34,
                  "share_per_node": 0.0042,
                  "cost_usd": 0.4 if req.tier == "screen" else 2.2}, ""


def evaluator_for(cfg: FleetConfig, tier: str) -> SimulatorEvaluator:
    """The real evaluator for one tier. Module-level so a test can check what
    a fleet would rent without renting it.

    The quality gate only fires against a baseline, and the baseline lives in
    `cfg.baseline["quality"]` as `{suite: accuracy}` -- the same map the
    stock sweep prints. Leaving it out does not weaken the gate, it removes
    it: every attempt scores `regressed=False` and a stack that serves worse
    answers faster is recorded as a win.
    """
    screen = tier == "screen"
    quality = dict((cfg.baseline or {}).get("quality") or {})
    root = pathlib.Path(cfg.root or (pathlib.Path.cwd() / "agents"))
    return SimulatorEvaluator(
        n_gpu=cfg.n_gpu, gpu=cfg.gpu, model=cfg.sim_model,
        levels=cfg.screen_levels if screen else cfg.levels,
        seconds_per_level=cfg.screen_seconds if screen else cfg.seconds_per_level,
        profile_level=0 if screen else cfg.profile_level,
        profiles_root=str(root / "profiles"),
        extra={"quality_baseline": quality})


def build(cfg: FleetConfig, store=None) -> tuple[Fleet, EvalBroker]:
    """Assemble every service. The one place implementations are chosen."""
    root = pathlib.Path(cfg.root or (pathlib.Path.cwd() / "agents"))
    root.mkdir(parents=True, exist_ok=True)
    store = store or SqliteSessionStore(default_store_path())
    memory = SqliteMemory(root / "memory.db")
    context = JsonlContext(root / "traces", session_id=cfg.session_id)
    bank = SqliteIdeaBank(cfg.bank) if cfg.bank else None

    if cfg.dry_run:
        runner = _fake_runner
    else:
        def runner(req):
            return evaluator_for(cfg, req.tier).evaluate(req.stack, req.run_dir)

    broker = EvalBroker(runner, capacity=cfg.eval_capacity)
    fleet = Fleet(None, broker, store=store, session_id=cfg.session_id,
                  root=str(root))
    fleet.bank = bank
    skills = SqliteSkillBank(default_skills_path())
    fleet.skills = skills
    if cfg.manager and not cfg.fake_agents:
        from .ideas.llm import ask_with
        from .manager import Manager
        fleet.manager = Manager(root, ask_with(cfg.model, cwd=root), skills=skills,
                                session_id=cfg.session_id)

    def make_agent(agent_id: str, fl: Fleet):
        ws = Workspace(root / agent_id, agent_id=agent_id)
        if cfg.fake_agents:
            prop = _ScriptedProposer(agent_id)
        else:
            prop = ClaudeCodeProposer(
                model=cfg.model, seed_model=cfg.seed_model, mode=cfg.mode,
                session_tools=(fl.manager.tools_index if fl.manager is not None else None),
                session_skills=(fl.skills.render if getattr(fl, "skills", None) else None))
            # Token use is attributed the moment it is spent, which is what
            # makes the dashboard's per-agent cost real rather than
            # apportioned after the fact.
            prop.on_tokens = lambda use, _a=agent_id: fl.report(_a, tokens=use)
        return IterativeAgent(
            agent_id=agent_id, workspace=ws, memory=memory, context=context,
            proposer=prop, evals=fl.evals, control=fl, baseline=cfg.baseline)

    fleet.make_agent = make_agent
    return fleet, broker


def check(cfg: FleetConfig) -> None:
    """Refuse configurations that run but cannot work."""
    if cfg.dry_run:
        return
    base = cfg.baseline or {}
    if not isinstance(base.get("bill_per_1k"), (int, float)):
        raise SystemExit("--baseline needs bill_per_1k from a stock sweep on the "
                         "same grid (docs/NEXT.md step 1)")
    if not (base.get("quality") or {}):
        raise SystemExit('--baseline needs "quality": {suite: accuracy}; without '
                         "it the quality gate never fires")
    if cfg.mode == "build" and not cfg.bank:
        raise SystemExit("--mode build needs --bank: build-mode agents implement a "
                         "recorded mechanism; without the bank they self-seed and "
                         "produce the one-line tweaks build mode exists to replace")
    if cfg.screen_first and not isinstance(base.get("screen"), dict):
        raise SystemExit('--baseline needs "screen": {"bill_per_1k": ...} from a '
                         "stock sweep at screen tier, or screens are compared "
                         "with a full sweep and can never be promoted")


def run(cfg: FleetConfig) -> None:
    check(cfg)
    store = SqliteSessionStore(default_store_path())
    fleet, broker = build(cfg, store=store)
    spec = FleetSpec(
        baseline_metrics=cfg.baseline,
        seeds=tuple(Idea(title=s[:40], hypothesis=s) for s in cfg.seeds),
        agent_budget=AgentBudget(max_attempts=cfg.agent_max_attempts,
                                 max_usd=cfg.agent_max_usd,
                                 patience=cfg.patience,
                                 screen_first=cfg.screen_first),
        fleet_budget=FleetBudget(max_agents=cfg.agents,
                                 max_concurrent_evals=cfg.eval_capacity,
                                 max_usd_total=cfg.budget_usd,
                                 max_wall_s=cfg.max_wall_s),
        root=cfg.root, note=cfg.note)

    stopping = {"flag": False}

    def _sig(_s, _f):
        stopping["flag"] = True
        fleet.stop("signal")

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    fleet.start(spec)
    try:
        while not stopping["flag"] and fleet.state().running:
            time.sleep(1.0)
            snap = store.read(cfg.session_id)
            if snap is not None and snap.phase == "stopping":
                break
    finally:
        fleet.stop("finished")
        broker.shutdown(wait=False)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="harness-daemon")
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    run(FleetConfig.load(a.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
