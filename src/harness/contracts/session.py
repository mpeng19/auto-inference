"""SessionStore: the seam between a running fleet and everything watching it.

A fleet runs detached for hours. The thing that watches it -- a TUI, a status
command, a script -- is a **different process**, started and stopped whenever
someone feels like it. So the two cannot share objects, and the fleet cannot
push to a viewer that may not exist.

Both sides therefore talk to a store instead:

    fleet  --writes-->  [ session store ]  <--reads--   TUI / CLI
           <--reads---                     --writes-->  commands

That inversion is what makes control possible at all. A viewer issues
`pause a03` by *writing a command row*; the fleet picks it up on its next tick.
Nothing blocks, nothing needs the two processes alive at the same instant, and
a crashed viewer leaves the fleet untouched.

Two things the contract insists on:

**Cost and tokens are per agent, always.** "The fleet has spent $200" is not
actionable. "a03 has spent $80 across 4 attempts on an idea that has not
improved" is. Every counter is attributed.

**Commands are durable and acknowledged.** A `kill` that is delivered but not
recorded leaves a viewer showing an agent that is already gone, and a viewer
that cannot tell the difference between "not yet applied" and "ignored" is
worse than no viewer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .common import new_id, now

AgentStatus = Literal["idle", "thinking", "queued", "evaluating", "paused",
                      "stopping", "done", "failed"]
CommandKind = Literal["pause", "resume", "kill", "scale", "stop", "note"]
SessionPhase = Literal["starting", "running", "paused", "stopping", "stopped"]


@dataclass(frozen=True)
class TokenUse:
    """Token accounting. Cache reads are separated because they are ~10x cheaper
    and a fleet sharing a prefix should be able to see that working."""
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def __sub__(self, o: TokenUse) -> TokenUse:
        return TokenUse(max(0, self.input - o.input), max(0, self.output - o.output),
                        max(0, self.cache_read - o.cache_read),
                        max(0, self.cache_write - o.cache_write))

    def __add__(self, o: TokenUse) -> TokenUse:
        return TokenUse(self.input + o.input, self.output + o.output,
                        self.cache_read + o.cache_read,
                        self.cache_write + o.cache_write)

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


@dataclass(frozen=True)
class AgentView:
    """Everything a watcher needs about one agent, in one row."""
    agent_id: str
    status: AgentStatus = "idle"
    idea_title: str = ""
    idea_hypothesis: str = ""
    activity: str = ""              # what it is doing *right now*, one line
    attempt: int = 0
    attempts_total: int = 0
    best_delta_pct: float | None = None    # vs baseline; None = nothing yet
    # The last priced result, as a watcher reads it: the bill, where it
    # ranks on the OpenRouter board ("9/12"), and the market share one node
    # serves at that price.
    last_bill_per_1k: float | None = None
    last_rank: str = ""
    last_share_pct: float | None = None
    # Seconds by phase: edit, study, wait (on a GPU), recall, other. What an
    # agent-hour actually went on, and the number to look at when a run is
    # slow -- a fleet spending its time in `wait` needs GPUs, one in `edit`
    # needs a faster model or smaller ideas.
    phase_s: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens: TokenUse = field(default_factory=TokenUse)
    eval_ticket: str = ""
    queued_s: float = 0.0
    idle_s: float = 0.0
    # Model calls the fleet cut for producing nothing (`Fleet.stall_s`) and
    # restarted. A row that keeps stalling is a model or network problem,
    # not a slow idea.
    stalls: int = 0
    updated_at: float = field(default_factory=now)
    note: str = ""


@dataclass(frozen=True)
class SessionView:
    """The whole fleet at one instant."""
    session_id: str
    phase: SessionPhase = "starting"
    started_at: float = 0.0
    updated_at: float = field(default_factory=now)
    agents: tuple[AgentView, ...] = ()
    target_agents: int = 0
    cost_usd: float = 0.0
    tokens: TokenUse = field(default_factory=TokenUse)
    evals_queued: int = 0
    evals_running: int = 0
    evals_capacity: int = 0
    evals_completed: int = 0
    evals_deduped: int = 0
    gpu_utilisation: float = 0.0
    budget_usd: float = 0.0
    # Carried so a watcher can find the process without guessing a path. A
    # `kill` that has to reconstruct the working directory from conventions
    # fails exactly when it is most needed.
    pid: int = 0
    root: str = ""
    note: str = ""

    @property
    def live_agents(self) -> int:
        return sum(1 for a in self.agents
                   if a.status not in ("done", "failed", "paused"))


@dataclass(frozen=True)
class Command:
    """An instruction from a watcher. Durable until the fleet acknowledges it."""
    id: str = field(default_factory=lambda: new_id("cmd"))
    kind: CommandKind = "note"
    agent_id: str = ""              # "" means the whole session
    value: str = ""                 # scale target, note text, ...
    issued_at: float = field(default_factory=now)
    applied_at: float = 0.0
    result: str = ""


@runtime_checkable
class SessionStore(Protocol):
    # ── the fleet writes ──
    def create(self, view: SessionView) -> str: ...
    def publish(self, view: SessionView) -> None:
        """Overwrite the current snapshot. Called on every fleet tick."""
        ...

    def add_tokens(self, session_id: str, agent_id: str, use: TokenUse) -> None: ...

    def take_commands(self, session_id: str) -> tuple[Command, ...]:
        """Return unapplied commands. The fleet acknowledges each afterwards."""
        ...

    def acknowledge(self, command_id: str, result: str = "") -> None: ...

    # ── watchers read and command ──
    def read(self, session_id: str = "") -> SessionView | None:
        """Latest snapshot. Empty id means the most recent session."""
        ...

    def sessions(self, limit: int = 20) -> tuple[SessionView, ...]: ...

    def send(self, cmd: Command) -> str:
        """Issue a command. Returns its id; poll `command_status` for the result."""
        ...

    def command_status(self, command_id: str) -> Command | None: ...
