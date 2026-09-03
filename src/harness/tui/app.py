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
from textual.containers import Horizontal, Vertical, VerticalScroll
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
    #results_body { height: 1fr; }
    #results { width: 2fr; }
    #result_detail { width: 1fr; padding: 0 1; border-left: solid $panel; }
    #ask { dock: bottom; }
    #answer_box { height: 12; padding: 0 1; border-top: solid $panel; }
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
        ("o", "open_artifact", "Open selected result's files"),
        ("t", "open_timeline", "Open timeline (HTML)"),
        ("ctrl+up", "answer_grow", "Bigger answer box"),
        ("ctrl+down", "answer_shrink", "Smaller answer box"),
        ("tab", "next_tab", "Fleet / results"),
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
        self.results_text = ""
        self.answer_text = ""
        self._asker = None
        self._results_at = 0.0

    # ── layout ───────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        with TabbedContent(id="tabs"):
            with TabPane("fleet", id="tab_fleet"), Horizontal(id="body"):
                with Vertical(id="agents"):
                    yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
                yield Static(id="detail")
            with TabPane("results", id="tab_results"), Vertical():
                with Horizontal(id="results_body"):
                    yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
                    yield Static(id="result_detail")
                with VerticalScroll(id="answer_box"):
                    yield Static(id="answer")
                yield Input(placeholder="ask about this run (Enter to send; ctrl+up/down resizes)", id="ask")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#table", DataTable)
        for col, w in (("agent", 6), ("status", 11), ("idea", 22), ("att", 4),
                       ("Δ%", 7), ("$/1k", 7), ("rank", 6), ("share", 7),
                       ("cost", 9), ("tokens", 8)):
            t.add_column(col, width=w)
        r = self.query_one("#results", DataTable)
        for col, w in (("verdict", 8), ("tier", 6), ("Δ%", 7), ("$/1k", 7), ("rank", 6),
                       ("share", 7), ("N*", 4), ("agent", 6), ("hypothesis", 40)):
            r.add_column(col, width=w)
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
        self._settle_pending()
        self._render_summary()
        self._render_table()
        self._render_detail()
        self._render_results()

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
            bill = "-" if a.last_bill_per_1k is None else f"{a.last_bill_per_1k:.2f}"
            share = "-" if a.last_share_pct is None else f"{a.last_share_pct:.2f}%"
            t.add_row(a.agent_id,
                      Text(status, style=STATUS_STYLE.get(a.status, "")),
                      a.idea_title[:22] or "-", str(a.attempt),
                      Text(d, style="green" if d.startswith("-") else ""),
                      bill, a.last_rank or "-", share,
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
        if a.phase_s:
            t.append("time\n", style="bold")
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
        pend = self._pending.get(a.agent_id)
        if pend:
            t.append(f"\n{pend} pending…\n", style="yellow")
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

    def _render_result_detail(self) -> None:
        d = self.query_one("#result_detail", Static)
        rows = getattr(self, "_results", [])
        t = self.query_one("#results", DataTable)
        i = t.cursor_row
        if not rows or i is None or i >= len(rows):
            d.update(Text("select a result", style="dim"))
            return
        r = rows[i]
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
            x.append("\nartifacts  (o opens the folder, Enter opens the paper or report)\n", style="bold")
            for label, p in arts:
                x.append(f"  {label:<9} {p}\n", style="dim")
        self.result_artifacts = arts
        d.update(x)

    def on_data_table_row_highlighted(self, event) -> None:
        if getattr(event.data_table, "id", "") == "results":
            self._render_result_detail()
        else:
            self._render_detail()

    def action_next_tab(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab_results" if tabs.active == "tab_fleet" else "tab_fleet"
        if tabs.active == "tab_results":
            self._render_results(force=True)

    def action_ask(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab_results"
        self._render_results(force=True)
        self.query_one("#ask", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        q = event.value.strip()
        if not q or not self._root():
            return
        event.input.value = ""
        self._set_answer(f"you: {q}\n\nthinking…", "dim")
        self._answer(q)

    @work(thread=True, exclusive=True, group="ask")
    def _answer(self, question: str) -> None:
        try:
            if self._asker is None:
                from ..ask import Asker
                self._asker = Asker(self._root())
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

    def _settle_pending(self) -> None:
        """Drop a pending mark once the fleet's view reflects the command.
        Pause is cooperative, so `paused` appears at once but the agent
        finishes its current call first; the mark says "sent", the status
        says "applied", and neither should outlive the other by more than
        one refresh."""
        if not self.view:
            return
        status = {a.agent_id: a.status for a in self.view.agents}
        done = {"pause": ("paused", "done", "failed"), "resume": ("thinking", "evaluating",
                                                                   "queued", "idle", "done"),
                "kill": ("done", "failed", "stopping")}
        for agent_id, kind in list(self._pending.items()):
            if status.get(agent_id) in done.get(kind, ()):
                self._pending.pop(agent_id, None)
        # scale/stop are fleet-level; they never show per agent
        self._pending.pop("", None)

    def _artifacts_for(self, r) -> list[tuple[str, str]]:
        """Everything on disk behind one result: the run directory of its
        attempts, the report and plots, the paper if written, its trace, and
        the profile if captured. Paths, so a person can open them."""
        import pathlib

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
    def _open(path: str) -> None:
        import shutil
        import subprocess

        opener = shutil.which("open") or shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def action_open_artifact(self) -> None:
        arts = getattr(self, "result_artifacts", [])
        run = next((p for k, p in arts if k == "run"), None)
        if run:
            self._open(run)
            self.notify(f"opened {run}", timeout=2)

    def on_data_table_row_selected(self, event) -> None:
        if getattr(event.data_table, "id", "") != "results":
            return
        arts = getattr(self, "result_artifacts", [])
        target = next((p for k, p in arts if k == "paper"), None) or \
            next((p for k, p in arts if k == "report"), None)
        if target:
            self._open(target)
            self.notify(f"opened {target}", timeout=2)

    def action_open_timeline(self) -> None:
        import pathlib

        from ..timeline import render_html

        root = self._root()
        if not root:
            return
        p = pathlib.Path(root) / "timeline.html"
        p.write_text(render_html(root))
        self._open(str(p))
        self.notify(f"opened {p}", timeout=2)

    def _recent_calls(self, agent_id: str, k: int = 5) -> list[dict]:
        """The agent's last k model calls from its call log, summarised.
        Read from disk, throttled by the caller's refresh; a running call
        shows its messages so far."""
        import json
        import pathlib

        root = self._root()
        if not root:
            return []
        d = pathlib.Path(root) / agent_id / "calls"
        if not d.is_dir():
            return []
        out = []
        for f in sorted(d.glob("*.jsonl"))[-k:]:
            try:
                rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            except (OSError, ValueError):
                continue
            msgs = [r for r in rows if r.get("type") == "assistant"]
            tools: dict[str, int] = {}
            for r in msgs:
                for name in r.get("tools") or ():
                    tools[name] = tools.get(name, 0) + 1
            span = (rows[-1]["ts"] - rows[0]["ts"]) / 60 if len(rows) > 1 else 0.0
            out.append({"phase": f.stem.split("-")[0], "min": span, "msgs": len(msgs),
                        "out": sum(r.get("output", 0) for r in msgs),
                        "cache": sum(r.get("cache_read", 0) for r in msgs),
                        "tools": " ".join(f"{n}x{c}" for n, c in
                                          sorted(tools.items(), key=lambda kv: -kv[1])[:3])})
        return out

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
            self._render_detail()
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
