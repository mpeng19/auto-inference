"""The manager stashes a tool only when it can name the hours it saves."""
import json

from harness.contracts import AgentOutcome, Attempt, Idea
from harness.manager import Manager, ToolStash


def _out(agent, title, stop="no_progress", bill=None, pct=None, failure=""):
    best = None
    if bill is not None:
        best = Attempt(idea_id="i", agent_id=agent, ok=True,
                       metrics={"bill_per_1k": bill}, delta={"bill_per_1k_pct": pct})
    att = (Attempt(idea_id="i", agent_id=agent, ok=False, failure=failure),) if failure else ()
    return AgentOutcome(agent_id=agent, idea=Idea(title=title, hypothesis=title),
                        stop=stop, best=best, attempts=att)


def test_stash_writes_script_index_and_readme(tmp_path):
    st = ToolStash(tmp_path)
    p = st.add("Bench Decode!", "micro-benchmark a decode kernel against stock",
               "python tools/bench_decode.py --batch 12", "print('hi')", 2.5)
    assert p.name == "bench_decode.py" and p.read_text() == "print('hi')\n"
    assert "bench_decode.py: micro-benchmark" in st.index()
    assert "## bench_decode" in (tmp_path / "tools" / "README.md").read_text()
    assert ToolStash(tmp_path).index() == st.index()          # persisted


def test_reviews_in_batches_and_respects_the_bar(tmp_path):
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        n = len(prompts)
        if n == 1:
            return json.dumps({"tool": None, "notes": "nothing reusable yet"})
        if n == 2:
            return json.dumps({"tool": {"name": "shape_dump", "purpose": "p", "usage": "u",
                                        "hours_saved": 0.2, "code": "print(1)"},
                               "notes": "marginal"})
        return "prose then " + json.dumps({"tool": {"name": "bench_attn", "purpose": "bench",
                                                     "usage": "python tools/bench_attn.py",
                                                     "hours_saved": 3, "code": "print(2)"},
                                            "notes": "three agents re-wrote this"})

    m = Manager(tmp_path, ask, every=2)
    m.on_outcome(_out("a00", "fused attention", bill=12.0, pct=-1.0))
    assert not prompts                                  # batch not full
    m.on_outcome(_out("a01", "kv int8", failure="quality"))
    assert len(prompts) == 1 and "fused attention" in prompts[0] and "quality" in prompts[0]
    assert m.tools_index() == ""
    for _ in range(2):
        m.on_outcome(_out("a02", "x"))
    assert len(prompts) == 2 and m.tools_index() == ""   # below the bar
    for _ in range(2):
        m.on_outcome(_out("a02", "y"))
    assert len(prompts) == 3
    assert "bench_attn.py" in m.tools_index()
    assert "(none yet)" in prompts[0] and "bench_attn" not in prompts[2]
    assert m.reviews == 3 and any("stashed" in line for line in m.log)


def test_a_broken_reply_never_raises(tmp_path):
    m = Manager(tmp_path, lambda p: "not json at all", every=1)
    m.on_outcome(_out("a00", "t"))
    assert m.tools_index() == "" and m.log[-1].startswith("review:")


def test_manager_writes_facts_and_uses_the_model_as_judge(tmp_path):
    from harness.skills import SqliteSkillBank

    bank = SqliteSkillBank(tmp_path / "skills.db")
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        if "CONTRADICT" in prompt:
            ids = [line.split(":")[0].strip() for line in prompt.split("Existing facts:")[1].splitlines()
                   if line.strip().startswith("fact_")]
            return json.dumps(ids[:1])
        return json.dumps({"tool": None, "notes": "n",
                           "facts": [{"claim": f"claim number {len(prompts)}",
                                      "topic": "kv", "evidence": "e", "confidence": 0.8}]})

    m = Manager(tmp_path, ask, every=1, skills=bank, session_id="s1")
    m.on_outcome(_out("a00", "x"))
    assert len(bank.list()) == 1                       # first review: one fact, no judge
    m.on_outcome(_out("a01", "y"))                     # second review, then the judge
    active = bank.list()
    assert len(active) == 1 and active[0].claim == "claim number 2"
    assert active[0].source == "s1"
    assert len(bank.list(status=None)) == 2
    assert any("supersedes" in line for line in m.log)
    assert "Facts already held" in prompts[-2] or "Facts already held" in prompts[-1]
