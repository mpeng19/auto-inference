"""What an agent call costs in time, and what happens when it should stop.

Three failures from the night of 2026-09-02, one test group each.

**The tools were refused.** Every agent write-up contained "This command needs
your approval to run (it's a harness tool...)": `--permission-mode acceptEdits`
auto-approves edits and nothing else, so in headless mode every `harness tool`
call was denied and the agents guessed instead of measuring.

**The study outlived its result.** A study started when the sweep was
submitted and ran to completion whatever happened, so an agent whose screen
came back in five minutes spent another fifteen answering a question that had
been answered.

**A five-hour host sleep was invisible.** Nothing timed a phase, and
`subprocess.run(timeout=...)` measures a monotonic clock that stops while the
host sleeps -- so the calls neither timed out nor looked slow.
"""
import json
import pathlib
import threading
import time
from dataclasses import dataclass, field

import pytest

from harness import EvalBroker, IterativeAgent, Workspace
from harness.agent.claude_code import (
    DEFAULT_EDIT_TIMEOUT_S,
    DEFAULT_TARGETS,
    CallStats,
    ClaudeCodeProposer,
)
from harness.contracts import AgentBudget, Brief, Idea

from .test_workspace import FakeStock

P = "srt/managers/schedule_policy.py"


def _fake_claude(tmp_path: pathlib.Path, body: str, name: str = "claude") -> str:
    """A `claude` that is a shell script.

    The real binary is never invoked by the suite: a test that spends
    subscription usage is a test nobody runs. Everything here exercises the
    subprocess machinery -- process groups, pipes, the JSON envelope -- which
    is where the bugs were.
    """
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(0o755)
    return str(p)


ENVELOPE = json.dumps({
    "type": "result", "result": "I fused the two kernels.",
    "duration_ms": 812345, "duration_api_ms": 790000, "num_turns": 41,
    "is_error": False, "permission_denials": [],
    "usage": {"input_tokens": 900, "output_tokens": 120,
              "cache_read_input_tokens": 40_000, "cache_creation_input_tokens": 5},
})


# ── cancellable, timed calls ──────────────────────────────────────────────

def test_a_cancelled_call_dies_with_its_children_and_keeps_what_it_wrote(tmp_path):
    """Cancelling must kill the *group*: `claude` runs tools as children, and a
    SIGTERM to the leader alone leaves a `harness tool gpu-run` holding an
    H100 nobody is waiting for."""
    marker = tmp_path / "ran-to-the-end"
    binary = _fake_claude(tmp_path, f"echo half-a-thought\nsleep 1\ntouch {marker}\n")
    prop = ClaudeCodeProposer(binary=binary, timeout_s=30)

    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    t0 = time.time()
    text, _ = prop._run("go", cwd=tmp_path, cancel=cancel, phase="study")

    assert time.time() - t0 < 1.0, "the cancel waited for the process to finish"
    assert prop.last_call.cancelled and not prop.last_call.timed_out
    # Cancellation is not an error: the partial thought is the point.
    assert "half-a-thought" in text
    time.sleep(1.2)
    assert not marker.exists(), "the process group survived the cancel"


def test_a_call_past_its_limit_is_killed_and_says_so(tmp_path):
    binary = _fake_claude(tmp_path, "sleep 5\n")
    prop = ClaudeCodeProposer(binary=binary, timeout_s=0.4)
    with pytest.raises(TimeoutError):
        prop._run("go", cwd=tmp_path, phase="edit")
    assert prop.last_call.timed_out and prop.last_call.wall_s < 5
    assert prop.calls == [prop.last_call]


def test_call_stats_come_off_the_json_envelope(tmp_path):
    """Wall seconds next to the model's own duration is the whole diagnostic:
    a big gap between them is the host, not the model."""
    binary = _fake_claude(tmp_path, f"cat <<'JSON'\n{ENVELOPE}\nJSON\n")
    prop = ClaudeCodeProposer(binary=binary)
    text, use = prop._run("go", cwd=tmp_path, phase="edit")

    st = prop.last_call
    assert text == "I fused the two kernels."
    assert (st.duration_ms, st.duration_api_ms, st.num_turns) == (812345, 790000, 41)
    assert st.is_error is False and st.returncode == 0 and st.denials == 0
    assert st.phase == "edit" and st.model == "sonnet"
    assert st.wall_s >= 0 and st.started_at > 0
    assert not st.cancelled and not st.timed_out
    assert use.cache_read == 40_000 and use.output == 120
    assert prop.calls == [st], "every call is kept, not just the last"
    assert set(CallStats().as_dict()) >= {"wall_s", "duration_ms", "duration_api_ms",
                                          "num_turns", "is_error"}


def test_refused_tool_calls_are_counted(tmp_path):
    """"It decided not to run preflight" and "it was told it could not" read
    identically in a write-up. The envelope knows which it was."""
    envelope = ENVELOPE.replace(
        '"permission_denials": []',
        '"permission_denials": [{"tool_name": "Bash"}, {"tool_name": "Bash"}]')
    binary = _fake_claude(tmp_path, f"cat <<'JSON'\n{envelope}\nJSON\n")
    prop = ClaudeCodeProposer(binary=binary)
    prop._run("go", cwd=tmp_path)
    assert prop.last_call.denials == 2


def test_a_failing_call_still_raises(tmp_path):
    binary = _fake_claude(tmp_path, "echo boom >&2\nexit 3\n")
    prop = ClaudeCodeProposer(binary=binary)
    with pytest.raises(RuntimeError, match="exited 3"):
        prop._run("go", cwd=tmp_path)
    assert prop.last_call.returncode == 3


# ── the tools are actually allowed ────────────────────────────────────────

def test_the_harness_tools_are_allowed_on_the_command_line(tmp_path):
    """With only `--permission-mode acceptEdits` every shell command was
    refused, so no agent ran `preflight`, `recall` or `roofline` even once."""
    dump = tmp_path / "argv.txt"
    binary = _fake_claude(
        tmp_path, f"printf '%s\\n' \"$@\" > {dump}\ncat <<'JSON'\n{ENVELOPE}\nJSON\n")
    prop = ClaudeCodeProposer(binary=binary)
    prop._run("go", cwd=tmp_path)
    argv = dump.read_text().splitlines()

    assert "--allowedTools" in argv
    rules = argv[argv.index("--allowedTools") + 1:]
    for rule in ("Bash(harness tool:*)", "Bash(uv run harness tool:*)",
                 "Bash(harness tool gpu-run:*)", "Bash(harness tool equivalence:*)",
                 "Bash(ruff:*)", "Read", "Grep", "Glob"):
        assert rule in rules, rule
    # Rules contain spaces, so each must survive as ONE argv token -- joining
    # them into a list is how `Bash(harness tool *)` becomes three rules that
    # match nothing.
    assert "Bash(harness tool *)" in rules
    # Additive, not a replacement: edits still come from the permission mode.
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"


def test_trace_tools_are_allowed_when_a_profile_is_attached():
    """MCP tools need permission too; offered and then refused is the same bug."""
    cmd = ClaudeCodeProposer(mcp_config="/tmp/mcp.json")._cmd("hi", "sonnet")
    assert "mcp__tracedb" in cmd
    assert "mcp__tracedb" not in ClaudeCodeProposer()._cmd("hi", "sonnet")


# ── the edit prompt ───────────────────────────────────────────────────────

def test_the_edit_hands_over_the_design_and_the_workbench(tmp_path, stock_dir):
    dump = tmp_path / "prompt.txt"
    binary = _fake_claude(
        tmp_path, f"printf '%s' \"$2\" > {dump}\ncat <<'JSON'\n{ENVELOPE}\nJSON\n")
    ws = Workspace(tmp_path / "a01", agent_id="a01", source=FakeStock(stock_dir))
    prop = ClaudeCodeProposer(binary=binary)

    assert prop.targets == DEFAULT_TARGETS, "the menu starts at the kernels"
    assert prop._edit_timeout() == DEFAULT_EDIT_TIMEOUT_S
    assert ClaudeCodeProposer(edit_timeout_s=60)._edit_timeout() == 60

    idea = Idea(title="fused decode attention",
                hypothesis="fuse the KV gather into the decode kernel",
                design="expected gain: 12% decode time\n"
                       "risks: fp8 accumulation drift changes what the model says",
                targets=(P,))
    out = prop.edit(ws, idea, Brief(text="nothing on record"), 0, ())
    prompt = dump.read_text()

    assert "DESIGN.md" in prompt, "the design note is the first deliverable"
    assert "fuse the KV gather into the decode kernel" in prompt
    assert "expected gain: 12% decode time" in prompt
    assert "fp8 accumulation drift" in prompt
    assert "srt/managers/schedule_policy.py" in prompt      # the idea's targets win
    for tool in ("harness tool preflight", "harness tool gpu-run",
                 "harness tool equivalence"):
        assert tool in prompt, tool
    assert "pip dependency" in prompt
    assert "Multi-file diffs are expected" in prompt
    assert out == "I fused the two kernels."


def test_build_targets_exist_in_the_pinned_wheel():
    """A menu of paths that do not exist is worse than no menu: the agent
    spends its first ten minutes discovering that."""
    from harness.agent.stock import CACHE_ROOT, SGLANG_VERSION

    root = CACHE_ROOT / SGLANG_VERSION / "sglang"
    if not root.is_dir():
        pytest.skip("stock wheel is not extracted here and fetching needs the network")
    missing = [t for t in DEFAULT_TARGETS if not (root / t).is_file()]
    assert not missing, f"build targets missing from SGLang {SGLANG_VERSION}: {missing}"


# ── the loop: cancelled studies and timed phases ──────────────────────────

@dataclass
class StudyProposer:
    """A proposer whose study blocks until it is cancelled."""
    note: str = "I was halfway through reading the allocator"
    entered: threading.Event = field(default_factory=threading.Event)
    saw_cancel: bool = False
    last_call: CallStats = field(
        default_factory=lambda: CallStats(phase="edit", duration_ms=4200,
                                          duration_api_ms=4000, num_turns=9,
                                          wall_s=7.5))

    def seed(self, live_ideas, brief):
        return Idea(title="chunk", hypothesis="tune chunk", targets=(P,))

    def edit(self, ws, idea, brief, attempt, history):
        ws.edit(P, f"CHUNK = {8192 * (attempt + 2)}\n\n\nclass SchedulePolicy:\n"
                   "    pass\n")
        return "raised CHUNK"

    def study(self, ws, idea, brief, history, cancel=None):
        self.entered.set()
        if cancel is not None:
            cancel.wait(timeout=10)
            self.saw_cancel = cancel.is_set()
        return self.note


def _agent(tmp_path, stock_dir, memory, context, broker, proposer):
    ws = Workspace(pathlib.Path(tmp_path) / "a01", agent_id="a01",
                   source=FakeStock(stock_dir))
    return IterativeAgent(agent_id="a01", workspace=ws, memory=memory,
                          context=context, proposer=proposer, evals=broker,
                          baseline={"bill_per_1k": 12.23})


def _runner(delay=0.0):
    def run(req):
        if delay:
            time.sleep(delay)
        return True, {"bill_per_1k": 12.23, "n_star": 12, "cost_usd": 1.0}, ""
    return run


def _turns(context, ref):
    return list(context.read(ref))


def test_a_study_is_cut_short_when_the_result_arrives(
        tmp_path, stock_dir, memory, context):
    """Studying past the result is time spent answering a question that has
    been answered -- and it holds the attempt open while it does."""
    prop = StudyProposer()
    broker = EvalBroker(_runner(), capacity=1)
    agent = _agent(tmp_path, stock_dir, memory, context, broker, prop)
    try:
        out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                        AgentBudget(max_attempts=1, patience=1, screen_first=False,
                                    replicate_wins=False))
    finally:
        broker.shutdown()

    assert prop.entered.is_set(), "the study never ran"
    assert prop.saw_cancel, "the study was never told the result had landed"
    studies = [t for t in _turns(context, out.attempts[0].trace_ref)
               if t.name == "study"]
    assert len(studies) == 1
    assert prop.note in studies[0].content
    assert studies[0].content.endswith("(cut short: result arrived)")
    assert studies[0].data["cut_short"] is True
    # The agent was busy right up to the result: nothing to charge to idle.
    assert out.idle_s < 1.0


def test_a_study_that_outlives_its_budget_is_cancelled_anyway(
        tmp_path, stock_dir, memory, context):
    """The other direction: a study nobody stops would hold the attempt open
    for as long as it felt like thinking."""
    prop = StudyProposer()
    broker = EvalBroker(_runner(delay=0.6), capacity=1)
    agent = _agent(tmp_path, stock_dir, memory, context, broker, prop)
    agent.COLLECT_POLL_S = 0.02
    try:
        out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                        AgentBudget(max_attempts=1, patience=1, screen_first=False,
                                    replicate_wins=False, study_timeout_s=0.05))
    finally:
        broker.shutdown()

    studies = [t for t in _turns(context, out.attempts[0].trace_ref)
               if t.name == "study"]
    assert prop.saw_cancel
    assert studies[0].data["cut_short"] is False, "it finished before the result"
    # It stopped studying and then waited, which is what idle means.
    assert out.idle_s > 0.0


def test_every_turn_says_how_long_its_phase_took(
        tmp_path, stock_dir, memory, context):
    """A closed lid froze the fleet for five hours and the trace read exactly
    like a slow model. Wall seconds per phase is the difference."""
    prop = StudyProposer()
    broker = EvalBroker(_runner(), capacity=1)
    agent = _agent(tmp_path, stock_dir, memory, context, broker, prop)
    try:
        out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                        AgentBudget(max_attempts=1, patience=1, screen_first=False,
                                    replicate_wins=False))
    finally:
        broker.shutdown()

    turns = _turns(context, out.attempts[0].trace_ref)
    assert turns
    for t in turns:
        assert isinstance(t.data.get("elapsed_s"), (int, float)), t
        assert t.data.get("phase"), t
    phases = {t.data["phase"] for t in turns}
    assert {"start", "recall", "propose", "check", "submit", "study",
            "wait"} <= phases, phases

    # A model call carries its own accounting beside the loop's wall clock,
    # so "slow model" and "sleeping host" are distinguishable in one line.
    propose = next(t for t in turns if t.data["phase"] == "propose")
    assert propose.data["duration_ms"] == 4200
    assert propose.data["num_turns"] == 9
    assert propose.data["phase"] == "propose", "the call's own label must not win"


def test_agent_shell_can_find_harness():
    """The agent's cwd is its candidate directory, where `uv run` has no
    project; the console scripts must be on PATH."""
    import os
    import sys

    env = ClaudeCodeProposer()._env()
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(sys.executable)
    assert "ANTHROPIC_API_KEY" not in env


STREAM = "\n".join(json.dumps(e) for e in [
    {"type": "system", "subtype": "init"},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}],
                                      "usage": {"input_tokens": 2, "output_tokens": 20,
                                                "cache_read_input_tokens": 18000,
                                                "cache_creation_input_tokens": 100}}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "content": "hi"}]}},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}],
                                      "usage": {"input_tokens": 2, "output_tokens": 1,
                                                "cache_read_input_tokens": 28000,
                                                "cache_creation_input_tokens": 0}}},
    {"type": "result", "subtype": "success", "result": "done", "duration_ms": 3779,
     "num_turns": 2, "is_error": False, "permission_denials": [],
     "usage": {"input_tokens": 4, "output_tokens": 77, "cache_read_input_tokens": 46000,
               "cache_creation_input_tokens": 100}},
])


def test_stream_json_tokens_land_per_message_and_reconcile_to_the_envelope(tmp_path, stock_dir):
    """The fleet showed 0 tokens for an hour because usage arrived only in the
    envelope. Each assistant message is now reported as it lands, the
    envelope tops it up, and the total equals the envelope exactly."""
    binary = _fake_claude(tmp_path, f"cat <<'JSON'\n{STREAM}\nJSON\n")
    reports = []
    prop = ClaudeCodeProposer(binary=binary, calls_dir=str(tmp_path / "calls"))
    prop.on_tokens = reports.append
    text, use = prop._run("hi", cwd=str(tmp_path), phase="edit")
    assert text == "done"
    assert (use.input, use.output, use.cache_read, use.cache_write) == (4, 77, 46000, 100)
    assert len(reports) == 3                          # two messages, then the remainder
    total = reports[0] + reports[1] + reports[2]
    assert (total.input, total.output, total.cache_read) == (4, 77, 46000)
    assert reports[0].output == 20 and reports[2].output == 56
    st = prop.last_call
    assert st.n_messages == 2 and st.output_tokens == 77 and st.num_turns == 2
    log = pathlib.Path(st.log_path)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["type"] for r in rows] == ["assistant", "user", "assistant", "result"]
    assert rows[0]["tools"] == ["Bash"] and rows[1]["tool_result_chars"] == 2
    assert "stream-json" in prop._cmd("x", "sonnet") and "--verbose" in prop._cmd("x", "sonnet")


def test_a_cancelled_stream_still_keeps_text_and_tokens(tmp_path, stock_dir):
    """No envelope, but the messages that landed were counted and kept."""
    partial = "\n".join(STREAM.splitlines()[:2])
    binary = _fake_claude(tmp_path, f"cat <<'JSON'\n{partial}\nJSON\nsleep 30\n")
    import threading
    cancel = threading.Event()
    prop = ClaudeCodeProposer(binary=binary)
    threading.Timer(0.5, cancel.set).start()
    _text, use = prop._run("hi", cwd=str(tmp_path), phase="study", cancel=cancel)
    assert prop.last_call.cancelled and use.output == 20


@dataclass
class TimedOutProposer(StudyProposer):
    """An edit that is killed at its limit, with or without a diff in place."""
    leaves_diff: bool = True

    def edit(self, ws, idea, brief, attempt, history):
        if self.leaves_diff:
            ws.edit(P, "CHUNK = 16384\n\n\nclass SchedulePolicy:\n    pass\n")
        raise TimeoutError("claude (edit) killed after 7200s wall against a 7200s limit")

    def study(self, ws, idea, brief, history, cancel=None):
        return self.note


def test_an_edit_killed_at_its_limit_still_prices_the_diff_it_left(
        tmp_path, stock_dir, memory, context):
    """build-4's a00 wrote for two hours, was killed, and the idea closed as an
    error with the diff never seen by a GPU. The clock bounds the writing,
    not whether the writing counts."""
    broker = EvalBroker(_runner(), capacity=1)
    agent = _agent(tmp_path, stock_dir, memory, context, broker, TimedOutProposer())
    try:
        out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                        AgentBudget(max_attempts=1, patience=1, screen_first=False,
                                    replicate_wins=False))
    finally:
        broker.shutdown()
    assert out.stop != "error"
    assert out.attempts and out.attempts[0].metrics.get("bill_per_1k") == 12.23
    propose = [t for t in _turns(context, out.attempts[0].trace_ref) if t.name == "propose"]
    assert propose[0].kind == "thought" and "timed out" in propose[0].content


def test_an_edit_killed_with_nothing_written_is_still_an_error(
        tmp_path, stock_dir, memory, context):
    broker = EvalBroker(_runner(), capacity=1)
    agent = _agent(tmp_path, stock_dir, memory, context, broker,
                   TimedOutProposer(leaves_diff=False))
    try:
        out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                        AgentBudget(max_attempts=1, patience=1, screen_first=False,
                                    replicate_wins=False))
    finally:
        broker.shutdown()
    assert out.stop == "error"


def test_a_call_the_api_refused_is_retried_and_the_retry_counts(tmp_path):
    """529 Overloaded is the API's problem, not the idea's. build-4 closed an
    idea as an error every four minutes for an hour of it."""
    overloaded = json.dumps({
        "type": "result", "subtype": "success", "is_error": True, "num_turns": 1,
        "api_error_status": 529, "terminal_reason": "api_error",
        "result": "API Error: 529 Overloaded. This is a server-side issue",
        "duration_ms": 4000, "duration_api_ms": 1400, "permission_denials": [], "usage": {}})
    marker = tmp_path / "tried"
    binary = _fake_claude(tmp_path, (
        f"if [ ! -f {marker} ]; then touch {marker}; echo '{overloaded}'; exit 1; fi\n"
        f"echo '{ENVELOPE}'\n"))
    prop = ClaudeCodeProposer(binary=binary)
    prop.TRANSIENT_BACKOFF_S = (0.01, 0.01)
    text, use = prop._run("go", cwd=tmp_path)
    assert text == "I fused the two kernels." and use.output == 120
    assert [c.transient for c in prop.calls] == [True, False]
    assert prop.calls[0].returncode == 1 and prop.calls[-1].returncode == 0


def test_a_call_that_keeps_being_refused_is_the_error_it_was(tmp_path):
    overloaded = json.dumps({"type": "result", "is_error": True, "api_error_status": 529,
                             "result": "API Error: 529 Overloaded", "permission_denials": []})
    binary = _fake_claude(tmp_path, f"echo '{overloaded}'\nexit 1\n")
    prop = ClaudeCodeProposer(binary=binary)
    prop.TRANSIENT_BACKOFF_S = (0.01,)
    with pytest.raises(RuntimeError, match="exited 1"):
        prop._run("go", cwd=tmp_path)
    assert len(prop.calls) == 2 and all(c.transient for c in prop.calls)


def test_a_prompt_failure_is_not_retried(tmp_path):
    binary = _fake_claude(tmp_path, "echo boom >&2\nexit 3\n")
    prop = ClaudeCodeProposer(binary=binary)
    prop.TRANSIENT_BACKOFF_S = (0.01,)
    with pytest.raises(RuntimeError, match="exited 3"):
        prop._run("go", cwd=tmp_path)
    assert len(prop.calls) == 1 and not prop.calls[0].transient


def test_agent_shell_commands_get_a_gpu_run_sized_timeout(tmp_path):
    """build-4: 21 tool calls cut at exactly 600 s, Claude Code's default;
    each one a gpu-run that billed the GPU and lost the result."""
    env = ClaudeCodeProposer(binary=_fake_claude(tmp_path, "exit 0\n"))._env()
    assert int(env["BASH_DEFAULT_TIMEOUT_MS"]) >= 20 * 60 * 1000
    assert int(env["BASH_MAX_TIMEOUT_MS"]) >= int(env["BASH_DEFAULT_TIMEOUT_MS"])


@dataclass
class ToolSpendingProposer(StudyProposer):
    """An edit whose tools spent GPU money the evaluation queue never saw."""

    def edit(self, ws, idea, brief, attempt, history):
        from harness.agent import ledger
        ledger.append(ws.root, "gpu-run", 0.75, elapsed_s=420, gpu="H100")
        ws.edit(P, "CHUNK = 16384\n\n\nclass SchedulePolicy:\n    pass\n")
        return "measured then edited"

    def study(self, ws, idea, brief, history, cancel=None):
        return self.note


def test_what_an_agent_spends_in_its_own_tools_is_its_cost(
        tmp_path, stock_dir, memory, context):
    """build-4: 14 GPU-hours of gpu-run/ncu/equivalence against a fleet
    total that only counted evaluations."""
    reported = []

    class Control:
        def report(self, agent_id, **fields):
            if "cost_delta" in fields:
                reported.append(fields["cost_delta"])
        def wait_if_paused(self, agent_id, timeout_s=3600):
            return True
        def should_stop(self, agent_id):
            return False

    broker = EvalBroker(_runner(), capacity=1)
    agent = _agent(tmp_path, stock_dir, memory, context, broker, ToolSpendingProposer())
    agent.control = Control()
    try:
        out = agent.run(Idea(title="chunk", hypothesis="tune chunk", targets=(P,)),
                        AgentBudget(max_attempts=1, patience=1, screen_first=False,
                                    replicate_wins=False))
    finally:
        broker.shutdown()
    assert 0.75 in reported
    assert out.cost_usd == 1.0 + 0.75                # the sweep plus the tool
