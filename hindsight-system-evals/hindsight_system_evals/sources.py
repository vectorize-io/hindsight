"""One bank, two kinds of source: an authoritative guide and the chatter around it.

The scenario every real bank grows into. Someone retains the team handbook — a
guide, a runbook, a spec — and also retains the conversations where the same
subjects get discussed. Both land as facts in one pool, and recall ranks them by
similarity to the question, which is exactly the signal that cannot tell an
approved policy from someone guessing in a thread.

Two things make that hard, and both are modelled here deliberately, because
without them the question answers itself:

1. **Extraction flattens provenance.** A conversation retained normally does not
   reach recall as "Dana said she was fairly sure Redis would be fine". It
   reaches it as "Redis is a suitable primary store for the notifications
   service" — a flat assertion, indistinguishable in wording from a policy
   sentence. Leave the attribution in and the model solves the whole suite by
   reading it, which is not the situation anyone is in. So the conflicting facts
   here are flat, and the ONLY thing separating a policy from an opinion is its
   recorded provenance.
2. **Volume.** A handbook is a few dozen sentences; the conversations about it
   are thousands. Recall has a token budget (2048 by default, ~100 short facts),
   so on a real bank the guide competes for room with everything ever said about
   its subject. ``_chatter`` generates enough on-topic traffic that the budget
   genuinely truncates.

Every fact carries its provenance in its document metadata, ``source``:

* ``guide`` — the handbook. Reviewed, current unless something explicitly
  supersedes it.
* ``conversation`` — what was said in a channel or a meeting.

The six questions are chosen so no blunt rule scores well:

* ``authority_conflict`` — guide and chatter disagree. The guide wins.
* ``guide_only`` — only the guide answers it.
* ``conversation_only`` — only the chatter answers it. A fix that filters the
  conversations away fails here, which is why it is in the set.
* ``decision_supersedes`` — a dated, explicit decision changes what the guide
  says and says so. Here the chatter wins, so "the guide is always right" is not
  the answer either.
* ``policy_plus_exception`` — the guide states the rule, a conversation records
  an approved exception, and both are needed.

Answering all six requires *ordering* the sources, not choosing one.

One thing the traps deliberately do NOT punish: reporting a lower-ranked claim
*as* a lower-ranked claim. "The rule is two approvals; some people merge small
ones with one" is a good answer, and the shipped prompt explicitly asks for that
shape. An earlier version of these criteria failed it, which measured the eval
disagreeing with the design rather than the model getting anything wrong. The
trap is the claim presented as the RULE.
"""

from __future__ import annotations

import datetime
import os

from hindsight_client import Hindsight

from hindsight_system_evals.corpus import HardCorpus, HardFact, HardQuestion
from hindsight_system_evals.pages import SettleFn, prepare_bank

#: The approved exception in ``c-friday-exception`` has to still be open when the
#: run happens, or the correct answer flips to "no, it expired" — which is what a
#: hardcoded end date quietly did once the date passed, failing a run on an answer
#: that was right. Computed, so the corpus cannot rot into asking a question whose
#: gold answer changed underneath it.
_EXCEPTION_END = (datetime.date.today() + datetime.timedelta(days=120)).strftime("%d %B %Y")

#: Provenance lives in the document's metadata, which is where a bank naturally
#: records it — the coding-agents integration already stamps a harness there. It
#: is also the marker that did not work until this change: metadata was stripped
#: from reflect's tool results, so the model never saw the stamp and no amount of
#: prompting could rank the sources.
METADATA_KEY = "source"
GUIDE = "guide"
CONVERSATION = "conversation"


def _guide(fid: str, text: str) -> HardFact:
    return HardFact(fid, text, "source_priority", metadata={METADATA_KEY: GUIDE})


def _talk(fid: str, text: str) -> HardFact:
    return HardFact(fid, text, "source_priority", metadata={METADATA_KEY: CONVERSATION})


#: The handbook. Policy sentences, with nothing in the wording announcing that
#: they are policy — that is what the provenance marker is for.
_GUIDE_FACTS = [
    _guide(
        "g-datastore",
        "Every new production service uses PostgreSQL as its primary datastore. Redis is approved as a "
        "cache only and must never be used as a system of record.",
    ),
    _guide(
        "g-approvals",
        "A pull request needs two approvals from engineers outside the author's own team before it can "
        "be merged.",
    ),
    _guide(
        "g-postmortem",
        "The incident commander writes the postmortem, and it is due within five working days of the "
        "incident being resolved.",
    ),
    _guide(
        "g-rotation",
        "Production service credentials are rotated every 90 days.",
    ),
    _guide(
        "g-deploy-window",
        "Production deploys run Monday to Thursday. A Friday deploy requires written approval from the "
        "on-call lead.",
    ),
    _guide(
        "g-oncall",
        "An on-call shift lasts one week and starts on Wednesday at 10:00.",
    ),
    _guide(
        "g-branching",
        "Feature branches are deleted automatically seven days after merge.",
    ),
]

#: The facts that decide the questions. Flat assertions, exactly as extraction
#: would leave them — the tag is the only thing marking them as hearsay.
_CONVERSATION_FACTS = [
    # Contradicts the datastore policy, in the same register as the policy.
    _talk("c-redis-1", "Redis is a suitable primary store for the notifications service."),
    _talk("c-redis-2", "Redis works well as a system of record and has been run that way in production."),
    # Contradicts the review policy.
    _talk("c-approvals-1", "One approval is enough to merge a small pull request."),
    _talk("c-approvals-2", "Typo fixes are merged by their author without waiting for a review."),
    # An explicit, dated decision that supersedes the guide — and says so.
    _talk(
        "c-rotation-decision",
        "At the security review on 9 March 2026 the security lead decided that credential rotation for "
        "production services moves from 90 days to 30 days, effective immediately; the handbook has not "
        "been updated yet.",
    ),
    _talk("c-rotation-scope", "Staging credentials stay on the 90-day rotation."),
    # Only the conversations know who took this on.
    _talk("c-upgrade-owner", "Farid volunteered to run the PostgreSQL 16 upgrade for the payments database."),
    _talk("c-upgrade-review", "Lena offered to review the PostgreSQL 16 upgrade plan for the payments database."),
    _talk("c-upgrade-decline", "Farid could not take on the search reindex."),
    # An approved exception to a guide rule.
    _talk(
        "c-friday-exception",
        "On 2 April 2026 the on-call lead gave the payments team written approval to deploy on Fridays "
        f"for the duration of the billing migration, through {_EXCEPTION_END}.",
    ),
    _talk("c-friday-grumble", "Several engineers think Friday deploys are a bad idea regardless of approval."),
]

_NAMES = ("Dana", "Marek", "Priya", "Tomas", "Lena", "Farid", "Ines", "Owen")

#: On-topic traffic that answers nothing. Each entry is a template filled with a
#: name and a date, so the bank carries the volume a real one does without a
#: thousand hand-written lines. Nothing here asserts a competing fact: two rows
#: claiming to be the same thing would make "the sources conflict" the correct
#: answer and score it wrong.
_CHATTER_TEMPLATES = (
    ("ds", "{name} asked which PostgreSQL version the analytics cluster runs."),
    ("ds", "{name} posted a benchmark showing Redis reads beating Postgres reads on the cache path."),
    ("ds", "{name} asked whether anyone had used PostgreSQL logical replication here."),
    ("ds", "{name} said the notifications service prototype still keeps state in memory."),
    ("ds", "{name} complained that the Redis cache evicts entries sooner than expected."),
    ("ds", "{name} asked where the datastore guidance for new services is written down."),
    ("ds", "{name} asked whether a read replica would help the reporting queries."),
    ("cr", "{name} asked how long reviewers normally take to respond to a pull request."),
    ("cr", "{name} said a pull request had been waiting three days for review."),
    ("cr", "{name} suggested a bot that pings reviewers on stale pull requests."),
    ("cr", "{name} asked whether draft pull requests show up in the review queue."),
    ("cr", "{name} said the review queue dashboard was broken again."),
    ("cr", "{name} asked whether review comments should block a merge or just be advisory."),
    ("in", "{name} asked where postmortems are stored."),
    ("in", "{name} said the last postmortem was unusually thorough."),
    ("in", "{name} asked whether a near miss needs a postmortem at all."),
    ("in", "{name} said they had been incident commander twice this month."),
    ("in", "{name} asked who chairs the incident review meeting."),
    ("in", "{name} asked whether the postmortem template had been updated."),
    ("se", "{name} asked how to rotate a credential without downtime."),
    ("se", "{name} said the rotation runbook is out of date."),
    ("se", "{name} asked whether credential rotation is automated or manual."),
    ("se", "{name} said two services missed their last rotation window."),
    ("se", "{name} asked which team owns the secrets manager."),
    ("dp", "{name} asked how long a production deploy usually takes end to end."),
    ("dp", "{name} said the Thursday deploy had to be rolled back."),
    ("dp", "{name} asked whether the holiday deploy freeze is still in effect."),
    ("dp", "{name} said the deploy pipeline has been slower since the runner upgrade."),
    ("dp", "{name} asked who the on-call lead is this week."),
    ("oc", "{name} asked whether the on-call handover template had changed."),
    ("oc", "{name} said their on-call shift was unusually quiet."),
    ("oc", "{name} asked how to page the database team out of hours."),
    ("br", "{name} asked whether a deleted feature branch can be restored."),
    ("br", "{name} said they had fourteen stale feature branches to clean up."),
    ("pm", "{name} asked when the payments database upgrade is scheduled."),
    ("pm", "{name} asked whether the payments upgrade needs a maintenance window."),
    ("pm", "{name} said the payments database has grown past two terabytes."),
    ("pm", "{name} asked who should be on the call during the payments upgrade."),
    ("pm", "{name} said the staging rehearsal of the payments upgrade went fine."),
)

#: How many times each template is used, with a different name and date each
#: time. 39 templates x 8 = 312 chatter facts against 7 guide sentences — the
#: ratio that makes the guide something recall has to be *aimed* at rather than
#: something it returns by accident.
_CHATTER_ROUNDS = 8


def _chatter() -> list[HardFact]:
    facts: list[HardFact] = []
    day = datetime.date(2026, 1, 5)
    for round_index in range(_CHATTER_ROUNDS):
        for template_index, (topic, template) in enumerate(_CHATTER_TEMPLATES):
            name = _NAMES[(round_index + template_index) % len(_NAMES)]
            day += datetime.timedelta(days=1)
            text = f"On {day:%d %B %Y} {template.format(name=name)}"
            facts.append(_talk(f"n-{topic}-{round_index}-{template_index}", text))
    return facts


_QUESTIONS = [
    HardQuestion(
        id="sp-datastore",
        category="authority_conflict",
        question="What should a new production service use as its primary datastore?",
        ideal_query="new production service primary datastore policy",
        gold=["g-datastore"],
        answer_criteria=(
            "States that a new production service must use PostgreSQL as its primary datastore. "
            "Mentioning that Redis has been discussed or is used as a cache is fine; presenting Redis as "
            "an acceptable system of record, or presenting the choice as unresolved, is not."
        ),
        must_not_claim=(
            "the team's rule permits Redis as a primary datastore or system of record, or that the "
            "memory does not settle which datastore to use. Reporting, with attribution, that someone "
            "suggested or used Redis — while being clear that the rule says PostgreSQL — does NOT count"
        ),
    ),
    HardQuestion(
        id="sp-approvals",
        category="authority_conflict",
        question="How many approvals does a pull request need before it can be merged?",
        ideal_query="pull request approvals required before merge",
        gold=["g-approvals"],
        answer_criteria=(
            "States that two approvals are required, from engineers outside the author's own team. "
            "Reporting that people sometimes do otherwise is fine only if it is not presented as the "
            "rule, and the answer must not leave the number unresolved."
        ),
        must_not_claim=(
            "the team's rule permits merging a pull request with one approval or none — for small pull "
            "requests, typo fixes or anything else — or that the required number of approvals is "
            "unclear. Reporting, with attribution, that some people merge that way in practice — while "
            "being clear that the rule requires two — does NOT count"
        ),
    ),
    HardQuestion(
        id="sp-postmortem",
        category="guide_only",
        question="Who writes the postmortem after an incident, and when is it due?",
        ideal_query="postmortem owner deadline after incident",
        gold=["g-postmortem"],
        answer_criteria=(
            "States that the incident commander writes it and that it is due within five working days of "
            "the incident being resolved."
        ),
        must_not_claim=(
            "a postmortem owner other than the incident commander, or a deadline other than five working days"
        ),
    ),
    HardQuestion(
        id="sp-upgrade-owner",
        category="conversation_only",
        question="Who volunteered to run the PostgreSQL 16 upgrade for the payments database?",
        ideal_query="volunteer PostgreSQL 16 upgrade payments database",
        gold=["c-upgrade-owner"],
        answer_criteria="Names Farid as the person who volunteered to run the upgrade.",
        must_not_claim=(
            "someone other than Farid volunteered to run the PostgreSQL 16 upgrade, or that the memory "
            "does not say who volunteered"
        ),
    ),
    HardQuestion(
        id="sp-rotation",
        category="decision_supersedes",
        question="How often are production service credentials rotated?",
        ideal_query="production credential rotation interval current",
        gold=["g-rotation", "c-rotation-decision"],
        answer_criteria=(
            "States that production credentials are now rotated every 30 days, following the decision "
            "taken at the 9 March 2026 security review. Noting that the handbook still says 90 days, or "
            "that staging stays at 90 days, is correct; presenting 90 days as the current production "
            "interval is not."
        ),
        must_not_claim="production service credentials are currently rotated every 90 days",
    ),
    HardQuestion(
        id="sp-friday-deploy",
        category="policy_plus_exception",
        question="Can the payments team deploy to production on a Friday?",
        ideal_query="payments team Friday production deploy approval",
        gold=["g-deploy-window", "c-friday-exception"],
        answer_criteria=(
            "Says yes for the payments team, because the on-call lead approved Friday deploys in writing "
            f"for the billing migration through {_EXCEPTION_END}, and states the general rule that "
            "deploys run Monday to Thursday with a Friday deploy needing the on-call lead's written "
            "approval."
        ),
        must_not_claim=(
            "the payments team may not deploy on a Friday, or that Friday deploys need no approval at all"
        ),
    ),
]


def build() -> HardCorpus:
    facts = _GUIDE_FACTS + _CONVERSATION_FACTS + _chatter()
    ids = [f.id for f in facts]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate fact id in the source-priority corpus")
    texts = [f.text for f in facts]
    if len(set(texts)) != len(texts):
        raise RuntimeError("Duplicate fact text — the gold labels are matched back by text")
    known = set(ids)
    for question in _QUESTIONS:
        if missing := [g for g in question.gold if g not in known]:
            raise RuntimeError(f"{question.id} references unknown gold ids: {missing}")
    return HardCorpus(facts=facts, questions=list(_QUESTIONS))


def facts() -> list[HardFact]:
    return build().facts


def questions() -> list[HardQuestion]:
    return build().questions


#: How the bank is told to rank its sources. Every option here is something an
#: operator can do today with no new configuration — the only server change this
#: needs is that a document's metadata reaches the model at all.
#:
#: * ``none`` — the baseline: every document carries its provenance and nothing
#:   says what it means.
#: * ``mission`` — the ranking written into the bank's reflect mission. Reaches
#:   the searching turns AND the answering call, because the mission replaces the
#:   role in both.
#: * ``directive`` — the ranking as a bank directive, which additionally forces a
#:   compliance field on the ``done`` tool.
#: * ``query`` — the ranking appended to the question. The last thing the model
#:   reads, and the weakest operationally: every caller must remember it, and a
#:   mental-model refresh never gets it.
STRATEGY = os.getenv("HINDSIGHT_EVAL_SOURCE_STRATEGY", "none")

#: What the bank is told about its own sources. Deliberately not "the guide is
#: always right": two of the six questions punish that.
#:
#: Three clauses are load-bearing, each added because its absence was measured:
#:
#: * the WHO-and-WHEN test for what counts as an override — without it the model
#:   laundered a bare opinion back in as an "approved exception";
#: * the ban on turning a lower-ranked claim into a nuance or an "it depends" —
#:   without it the answer read *"General pull requests: two approvals. Small
#:   pull requests: one approval is sufficient."*, which is worse than ignoring
#:   the ranking because it reads as policy;
#: * "settled means settled" — without it the model reported the disagreement as
#:   ambiguity for the reader to resolve.
SOURCE_PRECEDENCE = f"""## Source Authority
Each memory's `metadata` records which document it came from. This bank ranks those sources, most \
authoritative first:
1. `{METADATA_KEY}={GUIDE}` — the team's approved handbook. It states current policy.
2. `{METADATA_KEY}={CONVERSATION}` — what someone said in a channel or a meeting.
A memory from neither ranks below both.

When memories disagree about what the rule is, or about the current state of something:

1. **The highest-ranked source states the rule.** Give it as the answer, plainly. This out-ranks the \
temporal rule here: a lower-ranked memory does not supersede a higher-ranked one by being newer.
2. **A lower-ranked memory does NOT create an exception, a special case, a nuance or an 'it depends' to \
that rule.** A claim like 'one approval is enough for small ones' from a lower-ranked source is what \
somebody said or does — not a carve-out. Never write it into the answer as part of the rule, as a bullet \
beside the rule, or as a qualification of it. If it is worth mentioning at all, say explicitly that it is \
practice or opinion and that it departs from the rule.
3. **One thing does override the rule: a recorded decision or approval.** A lower-ranked memory beats the \
ranking only when it names WHO decided or approved it and WHEN. Then it is current: report it as the rule, \
and say what the higher-ranked source still says. An approval granted to a named team or person applies to \
them — report the rule and the approval together. A claim that names no decider and no date fails this \
test, however confident it sounds.
4. **A question that only the lower-ranked sources cover is answered from them.** Absence from the top \
source is not absence from the bank, and this ranking is not a reason to withhold an answer.

A disagreement settled by this ranking is SETTLED. Do not report it as a conflict, as ambiguity, or as two \
options for the reader to choose between."""

#: The mission form. A mission REPLACES the agent's default role rather than adding
#: to it, so it has to say what the agent is before it says how to rank.
MISSION = (
    "You are a reflection agent that answers questions by reasoning over retrieved memories.\n\n"
    + SOURCE_PRECEDENCE
)


async def apply_strategy(client: Hindsight, bank_id: str) -> None:
    """Configure the bank for the strategy under measurement."""
    if STRATEGY == "none":
        return
    if STRATEGY == "directive":
        await client.acreate_directive(bank_id, name="source-precedence", content=SOURCE_PRECEDENCE)
        return
    if STRATEGY == "mission":
        await client.aupdate_bank_config(bank_id, reflect_mission=MISSION)
        return
    if STRATEGY == "query":
        # Carried on the question instead, by decorate_query.
        return
    raise RuntimeError(f"Unknown HINDSIGHT_EVAL_SOURCE_STRATEGY: {STRATEGY!r}")


def decorate_query(question: str) -> str:
    """The ``query`` strategy: the ranking rides on the question.

    Worth measuring because position matters — the question is the LAST thing
    the model reads, which is where this repo already found a language directive
    had to live to be obeyed (#3776). Worth separating from the shipped feature
    because it only works when every caller remembers to append it, and it does
    not reach a mental-model refresh at all.
    """
    if STRATEGY != "query":
        return question
    return f"{question}\n\n{SOURCE_PRECEDENCE}"


async def seed_bank(client: Hindsight, bank_id: str, corpus_facts: list[HardFact], settle: SettleFn) -> None:
    """Retain every fact with its provenance tag, stored verbatim.

    ``prepare_bank`` puts the bank in ``chunks`` mode, so a fact reaches recall as
    the sentence written here — which is what the gold labels point at, and what
    keeps a 300-row seed free of model calls. One item per document, so the
    ``metadata`` on it is that document's metadata.
    """
    await prepare_bank(client, bank_id)
    await apply_strategy(client, bank_id)
    await client.aretain_batch(
        bank_id=bank_id,
        items=[{"content": fact.text, "metadata": fact.metadata} for fact in corpus_facts],
    )
    await settle(bank_id)
