"""Rules a session learns from what the agent says about its notes, and from the user's corrections.

The agent answers every note with a verdict and its reason (``NoteFeedback``): used, not needed,
or wrong. Some of those reasons carry a rule the next session should start from — a user
preference ("you asked for one panel, not three"), a procedure ("tests run on port 5617"), or a
correction of what memory claimed ("that cap is gone since the gateway queue"). In beta those
reasons stayed in one session: the next session in the bank was told the same rejected rule again.

So, when the bank turns it on (``session_learn_rules``), each new verdict is weighed: the decision
model says whether the reason states a lasting rule, and only then the LLM proposes a knowledge page
for it: a title and the question the page answers. Reflect writes the page from the bank's memories
and refreshes it like any other page. Checks read these pages, and only these; they are named
``rule-...`` so their effect can be measured.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LEARN_TASK = "session_learn"
# The decision model's likelihood that the reason states a rule for later work, from which the LLM
# is asked. Low on purpose: over 84 real verdicts its scores ran 0.04-0.89 with real rules on both
# sides of 0.5 ("tests run on port 5617" scored 0.49), so it only drops the reasons that are plainly
# no rule, and the writer, told what each verdict allows, makes the call (it wrote 20 of 84).
LEARN_AT = 0.15
RULE_ID_PREFIX = "rule-"
# How many of the bank's learned rules the writer is shown to avoid writing one twice.
MAX_EXISTING_RULES = 30
# A learned page's size cap. At the default 4096 the broad pages grew to 10-16k characters, reached
# checks about unrelated work, and over 100 replayed checks did no better than no pages (21 better,
# 21 worse). Narrowed and capped at 1024 (~1.4k characters), checks that cited a page got the better
# note 13 times to 5 (2026-11-13).
PAGE_MAX_TOKENS = 1024
# A rule whose notes agents called wrong this many times, and never used, stops reaching checks: it is
# tagged, not deleted, so a person can see it and take the tag off. Live, the first rule learned from
# one run's free port ("tests must use 5617") was rejected on each of its first notes, and kept
# coming back to every test run of the bank (2026-11-06). Only "wrong" counts, not "not_needed"
# (see service._retire_rejected_rules).
RETIRE_AFTER = 2
# A verdict skipped for one of these is weighed again RETRY_AFTER_S later (see run_learn).
RETRY_REASONS = frozenset({"decision failed", "writer failed"})
RETRY_AFTER_S = 60
RETIRED_TAG = "rule:retired"
# A user reply is weighed as a correction only within these lengths: shorter is "go" or "yes", longer is
# a pasted log or a new spec. Over a week of live replies, the corrections ran 13 to ~1,500 characters.
REPLY_MIN_CHARS = 10
REPLY_MAX_CHARS = 2000

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
        "The note changed the work. The rule is what the agent's reason says the work now does and why. Where "
        "the reason differs from the note (the agent did something else, or the opposite, because of it), "
        "follow the reason and never the note."
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

# Shared by both writers; <SOURCE> and <SAYS> are each writer's own words. A learned rule is a
# knowledge page: the writer proposes its title and question, and reflect writes the page from the
# bank's memories and keeps it refreshed. The writer first wrote the rule's text itself, saved as is:
# it never changed when the project did, cited nothing, and was managed apart from every other page
# (2026-11-11). What it learned about wording rules carries over to the question: ask for the standing
# rule, not the task ("Merge X into Y" scored 0.15 for a later change to X, the same fact as an
# instruction for later changes 0.94, 2026-11-05), and never for a value one run chose (from "I ran
# with 5617; the default port was taken" it wrote "tests must use 5617", which nothing in the repo
# names). An open question, because reflect answers the question as asked: converted from the old rules,
# "Should the memlake migration include org migration or specific field updates?" opened its page with
# "specific field updates", though the observations it cited said documents only, and "Should I
# investigate failures in <test>?" opened with "No, you should not" (2026-11-11). The note writer's
# "from the agent's reason" is kept: said generically ("given above") it
# brought back the run's own port (test_a_value_the_agent_chose_for_its_run_is_not_the_rule).
PAGE_FORMAT = """Propose the page, for agents in later sessions. The page is written from the project's
memories by answering its question, and an agent about to change that component reads it:
- name: a short title naming the component and the topic (at most 8 words).
- query: the one question the page answers, about the standing rule <SOURCE> settles for any later
  change there. Name the exact file, component or setting from <SOURCE>. Ask for the rule, not about
  the task the agent did: "How must new panels be added to shell.tsx?", not "Were the panels merged
  into one?". A value the agent picked only for its own run (a free port, a temp folder, a branch
  name) is not the project's: ask about the condition ("Which port do tests use when the default 5556
  is taken?"), not that value. Ask an open question (what, how, which, why), never a yes/no or
  either/or one: the page answers it as asked. Never ask whether a check can be skipped or a failure
  ignored. Never include a password, key or token.

Set create to false when an existing page above already answers this, or when <SAYS>.
When this page replaces existing pages above (they hold an older value or the opposite), list their
names in replaces: they are removed.
"""

WRITE_PROMPT = """You keep the rules of a software project that coding agents must follow.

A memory check told an agent this:
{note}

The agent's verdict on it: {verdict}
The agent's reason: {reason}
What the user had asked the agent: {goal}

{guide}

Pages of rules the project already has (title: question):
{existing}

""" + PAGE_FORMAT.replace("<SOURCE>", "the agent's reason").replace("<SAYS>", "the verdict above says to")

# Rules from the user's own corrections. Measured over a week of live sessions (2026-11-10): of
# ~830 follow-up user messages, 164 corrected the agent, and for 28 of them memory already held the
# lesson the user had to repeat ("it's 4 cent, not 40": the price was in memory since Oct 22). Only
# note verdicts were learned from, so none of those corrections became a rule.
# "What the developer had asked" is the user's message before the reply, read from the log: each reply
# becomes the session's request, so the session's own request is that reply or a later one. Without the
# line, the writer wrote 54 rules from 829 real replies, many of them this prompt's own instructions.
CORRECTION_PROMPT = """You keep the rules of a software project that coding agents must follow.

The agent said this:
{agent}

The developer replied:
{user}

What the developer had asked the agent before: {goal}

First, corrects: true only when the developer says the agent got something WRONG: a wrong value,
fact, file, approach or assumption, or something it should have known (said before, a project
convention). A new request or a change of plan ("now do X", "rename Y to Z", "let's also add"), a
question, an approval, or "go on" is false: then create is false too.

Then create: true only when what the developer says will still hold for later, different work on this
project, not only for this task. The rule is what the developer says, never what the agent did that the
developer rejected.

Pages of rules the project already has (title: question):
{existing}

""" + PAGE_FORMAT.replace("<SOURCE>", "the developer's reply").replace("<SAYS>", "corrects is false")

# A used note is not always right as written: "Count transfers to the piva account as tax already set
# aside" was used to REMOVE that shortcut ("I removed the shortcut ...; the dashboard now uses the real
# balance"), and the writer, told "the rule is what the note says", wrote the removed shortcut back as
# a rule (live, fin, 2026-11-09).

# For a rejected note the writer is not shown the note: told only "never write what the note claims",
# the writer still inverted the agent's reason into the note (offline: "you asked to merge them into
# one panel" became "Do not merge into a single container").
REJECTED_NOTE = "(not shown: the agent rejected it; work from the agent's reason)"


class RuleDraft(BaseModel):
    """The LLM's answer: the page to propose (its title and question), or that none is needed."""

    create: bool
    name: str = ""
    query: str = ""
    replaces: list[str] = Field(default_factory=list)


class CorrectionDraft(BaseModel):
    """The writer's answer for a user's reply: whether it corrects the agent at all, then the rule.

    ``corrects`` comes first so the writer decides it before drafting: asked only "create?", it made
    239 rules from 298 real replies, most of them the user's next request reworded (2026-11-10).
    Its own model, not a RuleDraft subclass: a subclass puts the new field last in the schema.
    """

    corrects: bool
    create: bool
    name: str = ""
    query: str = ""
    replaces: list[str] = Field(default_factory=list)


class LearnLog(BaseModel):
    """A ``learn`` entry in the session log: what one note's verdict, or one user reply, taught the bank."""

    note_seq: int = Field(description="The note's seq, or the user message's for a correction")
    source: Literal["note", "correction"] = Field(
        default="note", description="note (an agent's verdict) or correction (a user reply)"
    )
    outcome: str = Field(description="created, no_rule, covered, skipped, or retired")
    score: float | None = None
    mental_model_id: str | None = None
    name: str | None = None
    reason: str | None = None
    replaced: list[str] | None = None


def learn_context(note: str, verdict: str, reason: str, goal: str) -> str:
    """What the decision model reads before the reason it judges."""
    return f"Request: {goal}\nThe memory check told the agent: {note}\nThe agent's verdict: {verdict}"


class RuleSummary(BaseModel):
    """A rule the bank already learned, as the writer is shown it: its page's title and question."""

    id: str
    name: str
    query: str


def existing_rules_text(rules: list[RuleSummary]) -> str:
    if not rules:
        return "(none yet)"
    return "\n".join(f"- {r.name}: {r.query[:300]}" for r in rules)


def is_reply_to_weigh(text: str) -> bool:
    """Whether a user message is weighed as a correction: plain words of the user, not a system
    reminder or tool notification riding in the user's turn ("<task-notification>", "[Request ...")."""
    text = text.strip()
    return REPLY_MIN_CHARS <= len(text) <= REPLY_MAX_CHARS and not text.startswith(("<", "["))
