"""The skill bank: facts about serving this model, learned across runs.

`SqliteSkillBank` implements `harness.contracts.SkillBankService`. The manager
writes facts as it reviews outcomes (`add`, with a model as the contradiction
`Judge`; `lexical_judge` is the fallback); every agent reads the bank rendered
as a Claude Code skill (`render`) before it edits; `harness skills` lists,
adds and retracts by hand.

    bank = SqliteSkillBank(default_skills_path())
    bank.add(Fact(claim=..., topic=..., evidence=...), judge=...)   # (id, superseded ids)
    bank.list(topic=...) / bank.search(text) / bank.render()

One SQLite file shared by every run on this machine:
`~/.auto-inference/skills.db` (`HARNESS_HOME` moves it). Nothing is deleted;
a contradicted fact is marked superseded with a pointer to its successor.
`docs/*/SKILL.md` beside this package are the static skills installed with it.
"""
from .sqlite import SqliteSkillBank, default_skills_path, lexical_judge

__all__ = ["SqliteSkillBank", "default_skills_path", "lexical_judge"]
