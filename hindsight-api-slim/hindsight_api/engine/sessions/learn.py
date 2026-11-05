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
# is asked. Low on purpose: over 84 real verdicts its scores ran 0.04-0.89 with real rules on both
# sides of 0.5 ("tests run on port 5617" scored 0.49), so it only drops the reasons that are plainly
# no rule, and the writer, told what each verdict allows, makes the call (it wrote 20 of 84).
LEARN_AT = 0.15
RULE_ID_PREFIX = "rule-"
# A rule for a rejected note that shares this much of the note's words restates the note.
RESTATES_NOTE = 0.6
# How many of the bank's learned rules the writer is shown to avoid writing one twice.
MAX_EXISTING_RULES = 30
MAX_RULE_CHARS = 1200

LEARN_QUESTION = (
    "Does the agent's reason below settle a rule for future work on this project, one a later "
    "session should follow: a preference the user stated, a procedure the project uses, or a "
    "correction of what memory claimed? A reason that only says the work already did it, or that "
    "the note was about something else, settles no rule."
)

# What the verdict lets the rule say. A rule written from a rejected note restated the note: offline,
# over 84 real verdicts, the first prompt turned "you asked to remove per-pane Files" into "Do not drop
# per-pane files", and wrote 81 rules out of 84 verdicts.
VERDICT_GUIDE = {
    "used": (
        "The note was right and changed the work. The rule is what the note says, as the agent's reason confirms it."
    ),
    "wrong": (
        "The note was WRONG or outdated. Never write what the note claims. The rule is only the fact the "
        "agent's reason states is true instead. If the reason only says the note is wrong and states no "
        "fact of its own, set create to false."
    ),
    "not_needed": (
        "The note did not change the work. Write a rule only when the agent's reason states, in its own "
        "words, a request the user made or a fact about the project that replaces what the note said "
        '("you asked to ...", "X is now Y"). A reason that says the work already did it, or that the '
        "note was about something else, or that this task does not touch it, settles nothing: set create "
        "to false."
    ),
}

WRITE_PROMPT = """You keep the rules of a software project that coding agents must follow.

A memory check told an agent this:
{note}

The agent's verdict on it: {verdict}
The agent's reason: {reason}
What the user had asked the agent: {goal}

{guide}

Rules the project already has:
{existing}

Write the rule, for agents in later sessions:
- name: a short title naming the component and the rule (at most 8 words).
- query: the question an agent working on that component would ask, which this rule answers.
- content: the rule in at most 3 sentences: what to do or not do, where it applies, and why. Use the
  exact names, values and paths from the agent's reason. Never include a password, key or token.

Set create to false when an existing rule above already says this, or when the verdict above says to.
When this rule replaces existing rules above (they say an older value or the opposite), list their
names in replaces: they are removed.
"""

# For a rejected note the writer is not shown the note: told only "never write what the note claims",
# the writer still inverted the agent's reason into the note (offline: "you asked to merge them into
# one panel" became "Do not merge into a single container").
REJECTED_NOTE = "(not shown: the agent rejected it; work from the agent's reason)"


class RuleDraft(BaseModel):
    """The LLM's answer: the rule to write, or that none is needed."""

    create: bool
    name: str = ""
    query: str = ""
    content: str = ""
    replaces: list[str] = Field(default_factory=list)


class LearnLog(BaseModel):
    """A ``learn`` entry in the session log: what one note's verdict taught the bank."""

    note_seq: int
    outcome: str = Field(description="created, no_rule, covered, or skipped")
    score: float | None = None
    mental_model_id: str | None = None
    name: str | None = None
    reason: str | None = None
    replaced: list[str] | None = None


def learn_context(note: str, verdict: str, reason: str, goal: str) -> str:
    """What the decision model reads before the reason it judges."""
    return f"Request: {goal}\nThe memory check told the agent: {note}\nThe agent's verdict: {verdict}"


class RuleSummary(BaseModel):
    """A rule the bank already learned, as the writer is shown it."""

    id: str
    name: str
    content: str


def existing_rules_text(rules: list[RuleSummary]) -> str:
    if not rules:
        return "(none yet)"
    return "\n".join(f"- {r.name}: {r.content[:300]}" for r in rules)
