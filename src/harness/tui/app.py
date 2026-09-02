"""A deliberately small dashboard: one table, one detail pane, six keys.

The temptation with a fleet is to show everything -- every trace, every metric,
a graph per agent. That produces a screen nobody reads while a bill runs. So
this answers four questions and stops:

    who is running, and what is each one doing right now
    what has it cost, in dollars and in tokens, per agent and in total
    are the GPUs busy, or is the fleet stalled behind them
    which agent do I want to pause or kill

It talks only to the session store, so it can be started and stopped at will
and a crash here cannot touch a running fleet. Commands are written as rows;
the fleet applies them on its next tick, which is why an action shows
`pending` briefly before the status changes.
"""
from __future__ import annotations

import time
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

from ..contracts.session import Command

REFRESH_S = 1.0

STATUS_STYLE = {
    "evaluating": "bold cyan",
    "queued": "cyan",
    "thinking": "green",
    "paused": "yellow",
    "stopping": "yellow",
    "failed": "bold red",
    "done": "dim",
    "idle": "dim",
}


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


class FleetApp(App):
    """One table of agents, one summary, one detail pane."""

    CSS = """
    Screen { layout: vertical; }
    #summary { height: 4; padding: 0 1; }
    #body { height: 1fr; }
    #agents { width: 2fr; }
    #detail { width: 1fr; padding: 0 1; border-left: solid $panel; }
    """

    BINDINGS: ClassVar = [
        ("p", "pause", "Pause agent"),
        ("r", "resume", "Resume agent"),
        ("k", "kill_agent", "Kill agent"),
        ("+", "scale_up", "Add agent"),
        ("-", "scale_down", "Remove agent"),
        ("s", "stop_fleet", "Stop fleet"),
        ("q", "quit", "Quit (fleet keeps running)"),
    ]

    def __init__(self, store, session_id: str = ""):
        super().__init__()
        self.store = store
        self.session_id = session_id
        self.view = None
        self._pending: dict[str, str] = {}
        # What was last rendered, as plain text. Kept because a widget's
        # rendered content is not reliably readable across textual versions,
        # and a dashboard nobody can assert on is a dashboard that silently
        # stops updating.
        self.summary_text = ""
        self.detail_text = ""

    # ── layout ───────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        with Horizontal(id="body"):
            with Vertical(id="agents"):
                yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            yield Static(id="detail")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#table", DataTable)
        for col, w in (("agent", 6), ("status", 11), ("idea", 22), ("att", 4),
                       ("Δ%", 7), ("cost", 9), ("tokens", 8)):
            t.add_column(col, width=w)
        self.set_interval(REFRESH_S, self.refresh_view)
        self.refresh_view()

    # ── data ─────────────────────────────────────────────────────────────
    @work(thread=True, exclusive=True)
    def _load(self) -> None:
        try:
            v = self.store.read(self.session_id)
        except Exception:
            v = None
        self.call_from_thread(self._apply, v)

    def refresh_view(self) -> None:
        self._load()

    def _apply(self, v) -> None:
        self.view = v
        self._render_summary()
        self._render_table()
        self._render_detail()

    def _render_summary(self) -> None:
        s = self.query_one("#summary", Static)
        v = self.view
        if v is None:
            self.summary_text = "no session found — run `harness start`"
            s.update(Text(self.summary_text, style="yellow"))
            return
        age = time.time() - v.updated_at
        stale = "  [stale]" if age > 10 else ""
        bar = _bar(v.cost_usd, v.budget_usd)
        t = Text()
        t.append(f"{v.session_id}  ", style="bold")
        t.append(f"{v.phase}{stale}\n",
                 style="bold green" if v.phase == "running" else "yellow")
        t.append(f"agents {v.live_agents}/{v.target_agents}    "
                 f"spend {_money(v.cost_usd)} / {_money(v.budget_usd)} {bar}    "
                 f"tokens {_tokens(v.tokens.total)}"
                 f" (cache {_tokens(v.tokens.cache_read)})\n")
        t.append(f"GPUs {v.evals_running}/{v.evals_capacity} busy  "
                 f"{v.evals_queued} queued  {v.evals_completed} done  "
                 f"{v.evals_deduped} deduped  {v.gpu_utilisation:.0%} utilisation",
                 style="cyan")
        if v.note:
            t.append(f"\n{v.note}", style="bold red")
        self.summary_text = t.plain
        s.update(t)

    def _render_table(self) -> None:
        t = self.query_one("#table", DataTable)
        row = t.cursor_row
        t.clear()
        for a in (self.view.agents if self.view else ()):
            pend = self._pending.get(a.agent_id, "")
            status = f"{a.status}{'*' if pend else ''}"
            d = "-" if a.best_delta_pct is None else f"{a.best_delta_pct:+.1f}"
            t.add_row(a.agent_id,
                      Text(status, style=STATUS_STYLE.get(a.status, "")),
                      a.idea_title[:22] or "-", str(a.attempt),
                      Text(d, style="green" if d.startswith("-") else ""),
                      _money(a.cost_usd), _tokens(a.tokens.total),
                      key=a.agent_id)
        if row is not None and t.row_count:
            t.move_cursor(row=min(row, t.row_count - 1))

    def _render_detail(self) -> None:
        d = self.query_one("#detail", Static)
        a = self._selected()
        if a is None:
            self.detail_text = "select an agent"
            d.update(Text(self.detail_text, style="dim"))
            return
        t = Text()
        t.append(f"{a.agent_id}\n", style="bold")
        t.append(f"{a.status}\n\n", style=STATUS_STYLE.get(a.status, ""))
        t.append("idea\n", style="bold")
        t.append(f"{a.idea_title or '-'}\n")
        t.append(f"{a.idea_hypothesis or ''}\n\n", style="dim")
        t.append("doing now\n", style="bold")
        t.append(f"{a.activity or '-'}\n\n")
        t.append("cost\n", style="bold")
        t.append(f"{_money(a.cost_usd)}   {a.attempts_total} attempts\n")
        t.append(f"in {_tokens(a.tokens.input)}  out {_tokens(a.tokens.output)}\n")
        t.append(f"cache read {_tokens(a.tokens.cache_read)}\n\n", style="dim")
        if a.queued_s:
            t.append(f"last wait for a GPU: {a.queued_s:.0f}s\n", style="dim")
        if a.idle_s:
            t.append(f"idle: {a.idle_s:.0f}s\n", style="dim")
        if a.note:
            t.append(f"\n{a.note}\n", style="yellow")
        pend = self._pending.get(a.agent_id)
        if pend:
            t.append(f"\n{pend} pending…\n", style="yellow")
        self.detail_text = t.plain
        d.update(t)

    def _selected(self):
        if not self.view or not self.view.agents:
            return None
        t = self.query_one("#table", DataTable)
        i = t.cursor_row
        if i is None or i >= len(self.view.agents):
            return None
        return self.view.agents[i]

    # ── commands ─────────────────────────────────────────────────────────
    def _send(self, kind: str, agent_id: str = "", value: str = "") -> None:
        if self.view is None:
            return
        self.store.send_to(self.view.session_id,
                           Command(kind=kind, agent_id=agent_id, value=value))
        if agent_id:
            self._pending[agent_id] = kind
        self.notify(f"{kind} {agent_id or value}".strip(), timeout=2)

    def action_pause(self) -> None:
        a = self._selected()
        if a:
            self._send("pause", a.agent_id)

    def action_resume(self) -> None:
        a = self._selected()
        if a:
            self._send("resume", a.agent_id)

    def action_kill_agent(self) -> None:
        a = self._selected()
        if a:
            self._send("kill", a.agent_id)

    def action_scale_up(self) -> None:
        if self.view:
            self._send("scale", value=str(self.view.target_agents + 1))

    def action_scale_down(self) -> None:
        if self.view:
            self._send("scale", value=str(max(0, self.view.target_agents - 1)))

    def action_stop_fleet(self) -> None:
        self._send("stop")


def _bar(x: float, total: float, width: int = 16) -> str:
    if total <= 0:
        return ""
    n = min(width, int(width * x / total))
    return "[" + "#" * n + "." * (width - n) + "]"


def run_tui(store, session_id: str = "") -> int:
    FleetApp(store, session_id).run()
    return 0
