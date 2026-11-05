"""Rules a session learns from what the agent says about its notes.

The agent answers every note with a verdict and its reason (``NoteFeedback``): used, not needed,
or wrong. Some of those reasons carry a rule the next session should start from — a user
preference ("you asked for one panel, not three"), a procedure ("tests run on port 5617"), or a
correction of what memory claimed ("that cap is gone since the gateway queue"). In beta those
reasons stayed in one session: the next session in the bank was told the same rejected rule again.

So, when the bank turns it on (``session_learn_rules``), each new verdict is weighed: the decision
model says whether the reason states a lasting rule, and only then the LLM writes it as a mental
model. Reflect searches a bank's mental models first, so every later check reads the rules. The
models are named ``rule-...`` so their effect can be measured.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

LEARN_TASK = "session_learn"
# The decision model's likelihood that the reason states a rule for later work, from which the LLM
# writes one. Below it nothing is written: most reasons are "I was already doing that".
LEARN_AT = 0.5
RULE_ID_PREFIX = "rule-"
# How many of the bank's learned rules the writer is shown to avoid writing one twice.
MAX_EXISTING_RULES = 30
MAX_RULE_CHARS = 1200

LEARN_QUESTION = (
    "Does the agent's reason below settle a rule for future work on this project, one a later "
    "session should follow: a preference the user stated, a procedure the project uses, or a "
    "correction of what memory claimed? A reason that only says the work already did it, or that "
    "the note was about something else, settles no rule."
)

WRITE_PROMPT = """You keep the rules of a software project that coding agents must follow.

A memory check told an agent this:
{note}

The agent's verdict on it: {verdict}
The agent's reason: {reason}
What the user had asked the agent: {goal}

Rules the project already has:
{existing}

Write the rule this exchange settles, for agents in later sessions:
- name: a short title naming the component and the rule (at most 8 words).
- query: the question an agent working on that component would ask, which this rule answers.
- content: the rule in at most 4 sentences: what to do or not do, where it applies, and why. When the
  agent's reason corrects the note (the note was outdated or wrong), state the correct rule and say it
  replaces the old one. Use the exact names, values and paths from the reason.

Set create to false when an existing rule above already says this, or when the reason is only about
this one task and settles nothing for later.
"""


class RuleDraft(BaseModel):
    """The LLM's answer: the rule to write, or that none is needed."""

    create: bool
    name: str = ""
    query: str = ""
    content: str = ""


class LearnLog(BaseModel):
    """A ``learn`` entry in the session log: what one note's verdict taught the bank."""

    note_seq: int
    outcome: str = Field(description="created, no_rule, covered, or skipped")
    score: float | None = None
    mental_model_id: str | None = None
    name: str | None = None
    reason: str | None = None


def learn_context(note: str, verdict: str, reason: str, goal: str) -> str:
    """What the decision model reads before the reason it judges."""
    return f"Request: {goal}\nThe memory check told the agent: {note}\nThe agent's verdict: {verdict}"


class RuleSummary(BaseModel):
    """A rule the bank already learned, as the writer is shown it."""

    name: str
    content: str


def existing_rules_text(rules: list[RuleSummary]) -> str:
    if not rules:
        return "(none yet)"
    return "\n".join(f"- {r.name}: {r.content[:300]}" for r in rules)
