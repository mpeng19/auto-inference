"""The dashboard: a fleet tab, a results tab, and an ask box.

The temptation with a fleet is to show everything -- every trace, every metric,
a graph per agent. That produces a screen nobody reads while a bill runs. So
the fleet tab answers four questions and stops: who is running and what each
is doing now, what it has cost in dollars and tokens (and minutes by phase),
whether the GPUs are busy or the fleet is stalled behind them, and which agent
to pause or kill. The results tab is the morning view: every experiment best
first, its artifacts, and a box to ask Claude about the run. It is a viewer
of saved logs -- nothing on it controls the fleet, and its keys open files.

It talks only to the session store, so it can be started and stopped at will
and a crash here cannot touch a running fleet. Commands are written as rows;
the fleet applies them on its next checkpoint. A snapshot is the daemon's
last word, not a heartbeat, so the dashboard also asks the OS whether the
daemon is still alive before believing "running".

**What "pending" means.** A command is pending exactly while its row has no
`applied_at`. The fleet sets that only after it has applied the command and
published the snapshot that shows it, so the mark and the status never
disagree for more than one refresh, and the mark is never cleared by
guessing from a status. If the daemon is dead the row cannot be delivered,
and the dashboard says so instead of showing a spinner that will not end.

**Dollars are Modal.** The only money on this screen is GPU spend
(evaluations and the agent's own GPU tools). Claude runs on the
subscription; its use is shown as tokens and never priced. The figure is
what Modal bills, not what the daemon has counted: the agents' own
`gpu-run` / `ncu` / `equivalence` calls are read from their directories
and added to the snapshot's number (`results.unreported_tool_spend`), so
a fleet whose daemon predates the spend ledger still shows its real bill.

**Three scroll regions.** The header (session, spend, evals, baseline) is
pinned. Everything under it -- both tabs, tables and detail panes -- is
one scrolling panel, `#main`. The ask panel is docked below `#main` as its
sibling, never its child, and its containers end every wheel event they
receive: at its top or bottom edge, the wheel stops there instead of
carrying on into the main panel, which is what textual does by default.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import time
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import MouseScrollDown, MouseScrollUp
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Static,
    TabbedContent,
    TabPane,
)

from ..contracts.session import Command

REFRESH_S = 0.5
# A live fleet acknowledges within a tick or two. A command still unapplied
# after this long is on a daemon that is not ticking (asleep, hung), which
# is a different thing from "pending" and is labelled as such.
UNACKNOWLEDGED_S = 15.0

STATUS_STYLE = {
    "evaluating": "bold cyan",
    "queued": "cyan",
    "thinking": "green",
    "paused": "yellow",
    "stopping": "yellow",
    "failed": "bold red",
    "done": "dim",
    "idle": "dim",
    "lost": "bold red",
}

LIVE_PHASES = ("running", "stopping", "starting", "paused")

FLEET_ACTIONS = ("pause", "resume", "kill_agent", "scale_up", "scale_down", "stop_fleet")
RESULT_ACTIONS = ("open_artifact", "open_report")
ASK_ACTIONS = ("answer_grow", "answer_shrink", "close_ask")


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


class IsolatedScroll(VerticalScroll):
    """A scroll region the wheel cannot leave.

    Textual's own handler stops a wheel event only when it managed to
    scroll, so at an edge the event bubbles to the next scroller up -- for
    the ask panel that would be the main panel, which then moves under a
    reader who was only trying to reach the top of an answer. Here the
    event is handled and stopped whichever way it turns, and the default
    handler is skipped so nothing scrolls twice."""

    def _wheel(self, event, direction: int) -> None:
        event.prevent_default()
        event.stop()
        self.scroll_relative(y=direction * self.app.scroll_sensitivity_y, animate=False)

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._wheel(event, -1)

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._wheel(event, +1)


class _WheelToAnswer:
    """Mixin for the ask panel and the answer log: a wheel anywhere on
    them scrolls the answer box, and goes no further."""

    def _wheel(self, event, direction: int) -> None:
        event.prevent_default()
        event.stop()
        try:
            box = self.screen.query_one("#answer_box", IsolatedScroll)
        except Exception:
            return
        box.scroll_relative(y=direction * self.app.scroll_sensitivity_y, animate=False)

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._wheel(event, -1)

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._wheel(event, +1)


class AnswerLog(_WheelToAnswer, Static):
    pass


class AskPanel(_WheelToAnswer, Vertical):
    pass


class FleetApp(App):
    """One table of agents, one summary, one detail pane."""

    # Header widgets are fixed-height and outside every scroller, so they
    # stay put. `#main` is the one scroller for both tabs: every pane in it
    # is `height: auto`, so the panel grows with its content and `#main`
    # scrolls it. The ask panel is docked at the bottom of the screen, a
    # sibling of `#main`.
    CSS = """
    Screen { layout: vertical; overflow: hidden; }
    #summary { height: 4; padding: 0 1; }
    #baseline { height: 1; padding: 0 1; color: $text-muted; }
    #bill { height: 1; padding: 0 1; color: $text-muted; }
    #main { height: 1fr; }
    #tabs, #fleet_pane, #results_pane, #body, #results_body, #agents { height: auto; }
    DataTable { max-height: 100vh; }
    #agents { width: 2fr; }
    #detail_pane { width: 1fr; height: auto; border-left: solid $panel; }
    #detail { padding: 0 1; height: auto; }
    #results { width: 2fr; }
    #result_detail_pane { width: 1fr; height: auto; border-left: solid $panel; }
    #result_detail { padding: 0 1; height: auto; }
    #ask_panel { dock: bottom; height: auto; border-top: solid $panel; display: none; }
    #answer_box { height: 12; padding: 0 1; }
    #answer { height: auto; }
    """

    BINDINGS: ClassVar = [
        ("p", "pause", "Pause agent"),
        ("r", "resume", "Resume agent"),
        ("k", "kill_agent", "Kill agent"),
        ("+", "scale_up", "Add agent"),
        ("-", "scale_down", "Remove agent"),
        ("s", "stop_fleet", "Stop fleet"),
        ("a", "ask", "Ask about the run"),
        ("escape", "close_ask", "Close the ask box"),
        ("o", "open_artifact", "Open result's files"),
        ("b", "open_report", "Open report/paper in browser"),
        ("t", "open_timeline", "Open timeline (HTML)"),
        ("ctrl+up", "answer_grow", "Bigger answer box"),
        ("ctrl+down", "answer_shrink", "Smaller answer box"),
        # priority: the screen otherwise takes Tab for focus-cycling first
        Binding("tab", "next_tab", "Fleet / results", priority=True, show=False),
        ("q", "quit", "Quit (fleet keeps running)"),
    ]

    def __init__(self, store, session_id: str = ""):
        super().__init__()
        self.store = store
        self.session_id = session_id
        self.view = None
        # agent id -> its oldest unacknowledged command; fleet-level ones
        # (scale, stop) in the list. Both come from the store on every
        # refresh, so a dashboard restarted mid-flight shows the same marks.
        self._pending: dict[str, Command] = {}
        self._pending_fleet: list[Command] = []
        # Commands this dashboard sent and has not yet seen acknowledged:
        # the result ("paused", "a03 is already done") is shown once.
        self._awaiting: dict[str, str] = {}
        # What was last rendered, as plain text. Kept because a widget's
        # rendered content is not reliably readable across textual versions,
        # and a dashboard nobody can assert on is a dashboard that silently
        # stops updating.
        self.summary_text = ""
        self.baseline_text = ""
        self.detail_text = ""
        self.results_text = ""
        self.answer_text = ""
        self._asker = None
        self._results = []
        self._results_at = 0.0
        self._baseline_root = None
        self.result_artifacts: list[tuple[str, str]] = []
        # agent id -> Modal dollars its directory holds that the snapshot's
        # cost_usd does not; read off the UI thread with the snapshot.
        self._unreported: dict[str, float] = {}

    # ── layout ───────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        yield Static(id="baseline")
        yield Static(id="bill")
        with VerticalScroll(id="main"), TabbedContent(id="tabs"):
            with TabPane("fleet", id="tab_fleet"), Vertical(id="fleet_pane"), Horizontal(id="body"):
                with Vertical(id="agents"):
                    yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
                with Vertical(id="detail_pane"):
                    yield Static(id="detail")
            with TabPane("results", id="tab_results"), Vertical(id="results_pane"), \
                    Horizontal(id="results_body"):
                yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
                with Vertical(id="result_detail_pane"):
                    yield Static(id="result_detail")
        with AskPanel(id="ask_panel"):
            with IsolatedScroll(id="answer_box"):
                yield AnswerLog(id="answer")
            yield Input(placeholder="ask about this run (Enter to send; ctrl+up/down resizes)", id="ask")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#table", DataTable)
        for col, w in (("agent", 6), ("status", 12), ("idea", 22), ("att", 4),
                       ("Δ%", 7), ("$/1k", 7), ("rank", 6), ("share", 7),
                       ("$ modal", 9), ("tokens", 8)):
            t.add_column(col, width=w)
        r = self.query_one("#results", DataTable)
        for col, w in (("verdict", 8), ("tier", 6), ("Δ%", 7), ("$/1k", 7), ("rank", 6),
                       ("share", 7), ("N*", 4), ("agent", 6), ("hypothesis", 40)):
            r.add_column(col, width=w)
        self.set_interval(REFRESH_S, self.refresh_view)
        # Modal's own bill, so the dollars on this screen can be read against
        # the one Modal will charge. Fetched in the background every five
        # minutes; the line shows a dash until the first reading lands.
        from ..billing import Cached

        self._bill = Cached()
        self._show_bill()
        self.set_interval(30.0, self._show_bill)
        self.refresh_view()

    # ── data ─────────────────────────────────────────────────────────────
    @work(thread=True, exclusive=True)
    def _load(self) -> None:
        """Read the snapshot and the unacknowledged commands in one go, off
        the UI thread; the store is a file and must never stall a keypress."""
        try:
            v = self.store.read(self.session_id)
        except Exception:
            v = None
        pend: tuple[Command, ...] = ()
        acked: list[tuple[str, Command]] = []
        unrep: dict[str, float] = {}
        if v is not None:
            try:
                pend = self.store.pending(v.session_id)
                for cid, label in list(self._awaiting.items()):
                    c = self.store.command_status(cid)
                    if c is not None and c.applied_at:
                        acked.append((label, c))
            except Exception:
                pend = ()
            if v.root:
                from ..results import unreported_by_agent

                try:
                    unrep = unreported_by_agent(v.root)
                except Exception:
                    unrep = {}
        self.call_from_thread(self._apply, v, pend, acked, unrep)

    def refresh_view(self) -> None:
        self._load()

    @staticmethod
    def _daemon_alive(v) -> bool:
        """Is the process that published this snapshot still running?
        The store is the last thing a daemon wrote, not a heartbeat: a fleet
        that died at 04:00 still reads "running" at 09:00 unless someone
        asks the OS."""
        import os

        if not v or not v.pid:
            return False
        try:
            os.kill(v.pid, 0)
            return True
        except PermissionError:
            return True                      # exists, owned by someone else
        except (ProcessLookupError, OSError):
            return False

    @property
    def _alive(self) -> bool:
        """Can a command written now be delivered?"""
        return self.view is not None and self.view.phase in LIVE_PHASES

    def _apply(self, v, pend=(), acked=(), unrep=None) -> None:
        if unrep is not None:
            self._unreported = unrep
        if v is not None and v.phase in LIVE_PHASES and not self._daemon_alive(v):
            # Dead daemon: say so, and stop showing agents as busy.
            from dataclasses import replace
            v = replace(v, phase="dead",
                        agents=tuple(replace(a, status="lost",
                                             activity="daemon exited; last: " + (a.activity or "-"))
                                     for a in v.agents))
        self.view = v
        self._pending = {}
        self._pending_fleet = []
        for c in pend:                       # oldest first; keep the oldest per agent
            if c.agent_id:
                self._pending.setdefault(c.agent_id, c)
            else:
                self._pending_fleet.append(c)
        for label, c in acked:
            self._awaiting.pop(c.id, None)
            self.notify(f"{label}: {c.result}", timeout=3)
        self._render_summary()
        self._render_baseline()
        self._render_table()
        self._render_detail()
        self._render_results()
        self.refresh_bindings()

    # ── pending marks ───────────────────────────────────────────────────
    def _pending_label(self, c: Command) -> tuple[str, str]:
        """What to say about an unacknowledged command, and in what colour.
        Three states, none of them a spinner: pending (a live daemon will
        take it within a tick), unacknowledged (the daemon exists but is
        not ticking -- asleep or hung), undeliverable (the daemon is gone)."""
        what = f"{c.kind}{' ' + c.value if c.value else ''}"
        if not self._alive:
            return f"{what} undeliverable: the daemon is gone", "bold red"
        age = max(0.0, time.time() - c.issued_at)
        if age > UNACKNOWLEDGED_S:
            return f"{what} unacknowledged for {age:.0f}s (daemon not ticking?)", "bold red"
        return f"{what} pending ({age:.0f}s)", "yellow"

    def _render_summary(self) -> None:
        s = self.query_one("#summary", Static)
        v = self.view
        if v is None:
            self.summary_text = "no session found — run `harness start`"
            s.update(Text(self.summary_text, style="yellow"))
            return
        age = time.time() - v.updated_at
        stale = "  [stale]" if age > 10 else ""
        spend = self._fleet_spend()
        bar = _bar(spend, v.budget_usd)
        t = Text()
        t.append(f"{v.session_id}  ", style="bold")
        t.append(f"{v.phase}{stale}",
                 style="bold green" if v.phase == "running"
                 else "bold red" if v.phase == "dead" else "yellow")
        for c in self._pending_fleet:
            label, style = self._pending_label(c)
            t.append(f"   {label}", style=style)
        t.append("\n")
        if v.phase == "dead":
            t.append("the daemon is gone; this is its last snapshot\n", style="red")
        t.append(f"agents {v.live_agents}/{v.target_agents}    "
                 f"Modal spend {_money(spend)} / {_money(v.budget_usd)} {bar}    "
                 f"tokens {_tokens(v.tokens.total)}"
                 f" (cache {_tokens(v.tokens.cache_read)})\n")
        # Evaluation slots (`--evals`), each its own Modal container: not
        # GPUs the fleet owns.
        t.append(f"evals {v.evals_running}/{v.evals_capacity} running  "
                 f"{v.evals_queued} queued  {v.evals_completed} done  "
                 f"{v.evals_deduped} deduped  {v.gpu_utilisation:.0%} slot utilisation",
                 style="cyan")
        if v.note:
            t.append(f"\n{v.note}", style="bold red")
        self.summary_text = t.plain
        s.update(t)

    def _fleet_spend(self) -> float:
        """What Modal bills for the fleet: the snapshot's figure plus every
        agent directory's unreported tool spend."""
        return (self.view.cost_usd if self.view else 0.0) + sum(self._unreported.values())

    def _agent_spend(self, a) -> float:
        return a.cost_usd + self._unreported.get(a.agent_id, 0.0)

    def _render_baseline(self) -> None:
        """The numbers every delta on this screen is against, from the
        fleet's own `fleet.json`. Read once per root: it does not change
        while a fleet runs."""
        root = self._root()
        if root == self._baseline_root:
            return
        self._baseline_root = root
        base = {}
        if root:
            try:
                base = json.loads((pathlib.Path(root) / "fleet.json").read_text()).get("baseline") or {}
            except (OSError, ValueError):
                base = {}
        if not base:
            self.baseline_text = "baseline: none recorded (deltas cannot be judged)"
            self.query_one("#baseline", Static).update(Text(self.baseline_text, style="yellow"))
            return
        bits = []
        if isinstance(base.get("bill_per_1k"), (int, float)):
            bits.append(f"full ${base['bill_per_1k']:.2f}/1k")
        screen = base.get("screen") if isinstance(base.get("screen"), dict) else {}
        if isinstance(screen.get("bill_per_1k"), (int, float)):
            bits.append(f"screen ${screen['bill_per_1k']:.2f}/1k")
        for suite, acc in sorted((base.get("quality") or {}).items()):
            if isinstance(acc, (int, float)):
                bits.append(f"{suite} {acc:.0%}")
        self.baseline_text = "baseline (stock)   " + "   ".join(bits)
        self.query_one("#baseline", Static).update(Text(self.baseline_text))

    def _render_table(self) -> None:
        t = self.query_one("#table", DataTable)
        row = t.cursor_row
        t.clear()
        for a in (self.view.agents if self.view else ()):
            pend = self._pending.get(a.agent_id)
            mark = ""
            if pend is not None:
                mark = "*" if self._alive else "!"
            d = "-" if a.best_delta_pct is None else f"{a.best_delta_pct:+.1f}"
            bill = "-" if a.last_bill_per_1k is None else f"{a.last_bill_per_1k:.2f}"
            share = "-" if a.last_share_pct is None else f"{a.last_share_pct:.2f}%"
            t.add_row(a.agent_id,
                      Text(f"{a.status}{mark}", style=STATUS_STYLE.get(a.status, "")),
                      a.idea_title[:22] or "-", str(a.attempt),
                      Text(d, style="green" if d.startswith("-") else ""),
                      bill, a.last_rank or "-", share,
                      _money(self._agent_spend(a)), _tokens(a.tokens.total),
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
        t.append(f"{a.status}\n", style=STATUS_STYLE.get(a.status, ""))
        pend = self._pending.get(a.agent_id)
        if pend is not None:
            label, style = self._pending_label(pend)
            t.append(f"{label}\n", style=style)
        t.append("\nidea\n", style="bold")
        t.append(f"{a.idea_title or '-'}\n")
        t.append(f"{a.idea_hypothesis or ''}\n\n", style="dim")
        t.append("doing now\n", style="bold")
        t.append(f"{a.activity or '-'}\n\n")
        t.append("Modal spend\n", style="bold")
        t.append(f"{_money(self._agent_spend(a))}   {a.attempts_total} attempts\n")
        unrep = self._unreported.get(a.agent_id, 0.0)
        if unrep > 0:
            t.append(f"of which {_money(unrep)} from its own GPU tool calls, "
                     "not in the fleet's count\n", style="dim")
        t.append("\n")
        t.append("Claude tokens (subscription; not billed here)\n", style="bold")
        t.append(f"in {_tokens(a.tokens.input)}  out {_tokens(a.tokens.output)}  "
                 f"cache read {_tokens(a.tokens.cache_read)}  "
                 f"cache write {_tokens(a.tokens.cache_write)}\n\n")
        if a.phase_s:
            t.append("time by phase\n", style="bold")
            t.append("  ".join(f"{k} {_dur(v)}" for k, v in sorted(
                a.phase_s.items(), key=lambda kv: -kv[1])) + "\n")
        calls = self._recent_calls(a.agent_id)
        if calls:
            t.append("\nrecent calls\n", style="bold")
            for c in calls:
                t.append(f"  {c['phase']:<6} {c['min']:>5.1f}m {c['msgs']:>3} msgs "
                         f"out {_tokens(c['out']):>5} cache {_tokens(c['cache']):>6}"
                         f"{'  ' + c['tools'] if c['tools'] else ''}\n", style="dim")
        if a.queued_s:
            t.append(f"last wait for a GPU: {a.queued_s:.0f}s\n", style="dim")
        if a.idle_s:
            t.append(f"idle: {a.idle_s:.0f}s\n", style="dim")
        if a.note:
            t.append(f"\n{a.note}\n", style="yellow")
        self.detail_text = t.plain
        d.update(t)

    # ── results tab ──────────────────────────────────────────────────────
    def _root(self) -> str:
        return (self.view.root if self.view and self.view.root else "") or ""

    def _render_results(self, force: bool = False) -> None:
        """The run's experiments, best first. Read from memory.db every few
        seconds rather than every tick: it is a file on disk, not the store."""
        root = self._root()
        if not root:
            return
        if not force and time.time() - self._results_at < 5.0:
            return
        self._results_at = time.time()
        from ..results import leaderboard

        try:
            rows = leaderboard(root)
        except Exception:
            rows = []
        self._results = rows
        t = self.query_one("#results", DataTable)
        cur = t.cursor_row
        t.clear()
        for r in rows:
            d = "-" if r.delta_pct is None else f"{r.delta_pct:+.1f}"
            bill = "-" if r.bill_per_1k is None else f"{r.bill_per_1k:.2f}"
            share = "-" if r.share_pct is None else f"{r.share_pct:.2f}%"
            style = {"win": "bold green", "loss": "red", "neutral": "", "invalid": "dim"}.get(r.verdict, "")
            # Colour follows the verdict, not the sign: a -2% inside the
            # noise floor is neutral and must not look like a win.
            t.add_row(Text(r.verdict, style=style), r.tier,
                      Text(d, style=style),
                      bill, r.rank or "-", share, str(r.n_star or "-"),
                      r.agent_id, r.title, key=r.experiment_id)
        if cur is not None and t.row_count:
            t.move_cursor(row=min(cur, t.row_count - 1))
        self.results_text = "\n".join(f"{r.verdict} {r.delta_pct} {r.title}" for r in rows)
        self._render_result_detail()

    def _selected_result(self):
        rows = self._results
        t = self.query_one("#results", DataTable)
        i = t.cursor_row
        if not rows or i is None or i >= len(rows):
            return None
        return rows[i]

    def _render_result_detail(self) -> None:
        d = self.query_one("#result_detail", Static)
        r = self._selected_result()
        if r is None:
            self.result_artifacts = []
            d.update(Text("select a result", style="dim"))
            return
        x = Text()
        x.append(f"{r.experiment_id}  {r.agent_id}\n", style="bold")
        x.append(f"{r.verdict}\n\n", style="bold green" if r.verdict == "win" else "")
        x.append(f"{r.hypothesis}\n\n")
        x.append(f"{r.summary or '-'}\n", style="dim")
        if r.quality:
            x.append(f"quality {r.quality}\n", style="dim")
        if r.rank:
            x.append(f"rank {r.rank} on the OpenRouter board; one node serves "
                     f"{r.share_pct:.2f}% of the market\n", style="dim")
        arts = self._artifacts_for(r)
        if arts:
            x.append("\nartifacts  (o opens the run folder, b opens the paper or report)\n",
                     style="bold")
            for label, p in arts:
                x.append(f"  {label:<9} {p}\n", style="dim")
        else:
            x.append("\nno files on disk for this result (not a completed run)\n", style="dim")
        self.result_artifacts = arts
        d.update(x)

    def on_data_table_row_highlighted(self, event) -> None:
        if getattr(event.data_table, "id", "") == "results":
            self._render_result_detail()
        else:
            self._render_detail()
        self.refresh_bindings()

    @property
    def _tab(self) -> str:
        try:
            return self.query_one("#tabs", TabbedContent).active
        except Exception:
            return "tab_fleet"

    def action_next_tab(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab_results" if tabs.active == "tab_fleet" else "tab_fleet"
        if tabs.active == "tab_results":
            self._render_results(force=True)
        self.refresh_bindings()

    def on_tabbed_content_tab_activated(self, event) -> None:
        self.refresh_bindings()

    @property
    def _ask_open(self) -> bool:
        try:
            return self.query_one("#ask_panel", AskPanel).display
        except Exception:
            return False

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Which keys the footer lists, per tab. The fleet tab controls the
        fleet; the results tab is a viewer of saved logs and has no pause,
        kill or scale -- a footer full of keys that do nothing is noise, and
        a kill key on a page of old results is a trap. Opening a result's
        files needs files: only a completed run has them."""
        if action in ASK_ACTIONS:
            return self._ask_open
        if action in FLEET_ACTIONS:
            if self._tab != "tab_fleet":
                return False
            return True if self._alive else None       # shown dimmed when dead
        if action in RESULT_ACTIONS:
            if self._tab != "tab_results":
                return False
            kinds = {k for k, _ in self.result_artifacts}
            need = {"run"} if action == "open_artifact" else {"paper", "report"}
            return bool(kinds & need)
        return True

    def action_ask(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab_results"
        self._render_results(force=True)
        self.query_one("#ask_panel", AskPanel).display = True
        self.query_one("#ask", Input).focus()
        self.refresh_bindings()

    def action_close_ask(self) -> None:
        """Escape: hide the ask box and the answer, hand focus back to the
        results table. The conversation is kept; `a` reopens it."""
        self.query_one("#ask_panel", AskPanel).display = False
        self.query_one("#results", DataTable).focus()
        self.refresh_bindings()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        q = event.value.strip()
        if not q or not self._root():
            return
        event.input.value = ""
        self._set_answer(f"you: {q}\n\nthinking…", "dim")
        self._answer(q)

    def _live_view(self):
        """The snapshot for the ask context: the store's while the daemon is
        alive, nothing once it is not -- a finished run is answered from its
        directory, which is what outlives the store."""
        return self.view if self._alive else None

    @work(thread=True, exclusive=True, group="ask")
    def _answer(self, question: str) -> None:
        try:
            if self._asker is None:
                from ..ask import Asker
                self._asker = Asker(self._root(), view_source=self._live_view)
            text = self._asker.ask(question)
            u = self._asker.last_usage
            foot = (f"\n\n[{u.get('input', 0):,} in, {u.get('output', 0):,} out, "
                    f"{u.get('cache_read', 0):,} cached]") if u else ""
            self.call_from_thread(self._set_answer, f"you: {question}\n\n{text}{foot}", "")
        except Exception as e:
            self.call_from_thread(self._set_answer, f"ask failed: {type(e).__name__}: {e}", "red")

    def _set_answer(self, text: str, style: str = "") -> None:
        self.answer_text = text
        self.query_one("#answer", Static).update(Text(text, style=style))
        box = self.query_one("#answer_box", VerticalScroll)
        box.scroll_end(animate=False)

    def _resize_answer(self, delta: int) -> None:
        box = self.query_one("#answer_box", VerticalScroll)
        cur = int(getattr(box.styles.height, "value", 12) or 12)
        box.styles.height = max(4, min(40, cur + delta))

    def action_answer_grow(self) -> None:
        self._resize_answer(4)

    def action_answer_shrink(self) -> None:
        self._resize_answer(-4)

    def _artifacts_for(self, r) -> list[tuple[str, str]]:
        """Everything on disk behind one result: the run directory of its
        attempts, the report and plots, the paper if written, its trace, and
        the profile if captured. Paths, so a person can open them."""
        from ..paper import find_papers

        root = self._root()
        if not root:
            return []
        root = pathlib.Path(root)
        out: list[tuple[str, str]] = []
        agent = root / r.agent_id
        runs = sorted((agent / "runs").glob("attempt-*")) if (agent / "runs").is_dir() else []
        # the attempt whose report names this stack digest
        for d in reversed(runs):
            rep = d / "report.txt"
            try:
                if r.stack_digest and r.stack_digest in rep.read_text():
                    out.append(("run", str(d)))
                    out.append(("report", str(rep)))
                    if (d / "stack.json").is_file():
                        out.append(("stack", str(d / "stack.json")))
                    for png in sorted(d.glob("*.png")):
                        out.append(("plot", str(png)))
                    break
            except OSError:
                continue
        papers = find_papers(root)
        for idea_id, p in papers.items():
            if idea_id and (r.metrics.get("idea_id") == idea_id or idea_id in r.trace_ref):
                out.append(("paper", str(p)))
        if not any(k == "paper" for k, _ in out) and papers:
            # fall back: a paper by this agent whose directory is newest
            mine = [p for p in papers.values() if p.parts and r.agent_id in p.parts]
            if mine:
                out.append(("paper", str(sorted(mine, key=lambda p: p.stat().st_mtime)[-1])))
        if r.trace_ref:
            t = root / "traces" / f"{r.trace_ref}.jsonl"
            if t.is_file():
                out.append(("trace", str(t)))
        prof = r.metrics.get("profile_db")
        if prof and pathlib.Path(prof).is_file():
            out.append(("profile", str(prof)))
        return out

    @staticmethod
    def _open(path: str, kind: str = "file") -> None:
        """Hand a path to the right program: a folder to the IDE
        (`HARNESS_IDE`, else `cursor`, else `code`, else the OS opener); a
        document to the default browser, which renders PDFs and text alike."""
        import os
        import shutil
        import subprocess
        import webbrowser

        if kind == "dir":
            ide = os.environ.get("HARNESS_IDE") or shutil.which("cursor") or shutil.which("code")
            if ide:
                subprocess.Popen([ide, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        if kind == "browser":
            # An absolute file URI. `file://agents/build-4/timeline.html` is a
            # host named "agents", which is the ERR_INVALID_URL the browser
            # showed for every relative fleet root.
            webbrowser.open(pathlib.Path(path).resolve().as_uri())
            return
        opener = shutil.which("open") or shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def action_open_artifact(self) -> None:
        run = next((p for k, p in self.result_artifacts if k == "run"), None)
        if run:
            self._open(run, kind="dir")
            self.notify(f"opened {run} in the IDE", timeout=2)

    def action_open_report(self) -> None:
        arts = self.result_artifacts
        target = next((p for k, p in arts if k == "paper"), None) or \
            next((p for k, p in arts if k == "report"), None)
        if target:
            self._open(target, kind="browser")
            self.notify(f"opened {target} in the browser", timeout=2)

    def on_data_table_row_selected(self, event) -> None:
        """Enter on a result row does what `b` does."""
        if getattr(event.data_table, "id", "") == "results":
            self.action_open_report()

    def _show_bill(self) -> None:
        b = self._bill.get()
        text = b.line() if b else "modal this cycle  -  (bill not fetched yet, or no Modal token)"
        with contextlib.suppress(Exception):
            self.query_one("#bill", Static).update(Text(text))

    def action_open_timeline(self) -> None:
        from ..timeline import render_html

        root = self._root()
        if not root:
            return
        p = pathlib.Path(root) / "timeline.html"
        p.write_text(render_html(root))
        self._open(str(p), kind="browser")
        self.notify(f"opened {p}", timeout=2)

    def _recent_calls(self, agent_id: str, k: int = 5) -> list[dict]:
        """The agent's last k model calls, from `results.recent_calls`."""
        from ..results import recent_calls

        root = self._root()
        return recent_calls(root, agent_id, k=k) if root else []

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
        """Write a command row, or refuse. A row for a daemon that is gone
        is not "pending", it is a mark that never clears; say so instead."""
        if self.view is None:
            return
        what = f"{kind} {agent_id or value}".strip()
        if not self._alive:
            self.notify(f"{what}: not sent, the daemon is {self.view.phase}",
                        severity="warning", timeout=4)
            return
        cmd = Command(kind=kind, agent_id=agent_id, value=value)
        self.store.send_to(self.view.session_id, cmd)
        self._awaiting[cmd.id] = what
        # Show the mark now rather than after the next refresh; the store
        # will confirm it on the next read.
        if agent_id:
            self._pending.setdefault(agent_id, cmd)
        else:
            self._pending_fleet.append(cmd)
        self._render_summary()
        self._render_table()
        self._render_detail()
        self.notify(what, timeout=2)

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


def _dur(s: float) -> str:
    return f"{s/3600:.1f}h" if s >= 3600 else f"{s/60:.0f}m" if s >= 90 else f"{s:.0f}s"


def _bar(x: float, total: float, width: int = 16) -> str:
    if total <= 0:
        return ""
    n = min(width, int(width * x / total))
    return "[" + "#" * n + "." * (width - n) + "]"


def run_tui(store, session_id: str = "") -> int:
    FleetApp(store, session_id).run()
    return 0
