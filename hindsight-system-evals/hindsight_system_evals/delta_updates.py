"""Updates a delta refresh has to fold into a page it already wrote.

``test_01`` grades a page that is *built* across waves: most of what the second
wave carries is new to the page. This grades the other half of a delta's job —
an established page, then a wave that changes what it says. Each case is one
way an update reaches a page:

* ``linked_by_page`` — the new memory never names the topic; only the page
  connects them ("Dana left the company", on a page naming Dana as on-call);
* ``second_hop`` — the update names a service by its new name, and the rename
  arrives in the same wave;
* ``correction`` — a memory saying an earlier figure was wrong;
* ``membership`` — one item joins a list and another leaves it;
* ``new_item`` — something the topic covers that the page has never mentioned;
* ``quiet_wave`` — nothing in the wave bears on the page, which must survive it.

Behind it: the decision-model gate in front of a delta refresh asks, memory by
memory, "could this change the page?" and hands only the yeses to the LLM. A
memory it drops is never seen again — the next refresh starts after it. On one
cost run a decision model dropped 20 of 20 on-topic memories and the page kept
reading as it had. A wrong "no" leaves no trace but a page that stopped
updating, so this suite looks at the page. The first four cases are the ones a
yes/no on one memory at a time finds hardest, and every wave also carries the
hard corpus, so the gate has plenty to say no to.

Every case is graded twice, like ``test_01``: ``correct`` and the trap, which
here is the page still asserting the value the wave replaced. The wave-1 page
must state that old value first, or the update has nothing to replace.
"""

from __future__ import annotations

from dataclasses import dataclass

from hindsight_client import Hindsight

from hindsight_system_evals.corpus import HardQuestion
from hindsight_system_evals.pages import (
    PageOutcome,
    SettleFn,
    WaveSnapshot,
    create_page,
    facts,
    prepare_bank,
    read_page,
    split_into_waves,
)

#: The page is written from wave 1 and updated by wave 2; the dates make the
#: second wave unambiguously the later one.
_BASE_TIME = "2026-03-02T12:00:00Z"
_UPDATE_TIME = "2026-07-01T12:00:00Z"


@dataclass(frozen=True)
class DeltaUpdate:
    question: HardQuestion
    #: Wave 1: what the page is first written from.
    base: tuple[str, ...]
    #: Wave 2: what has to change it.
    update: tuple[str, ...]
    #: What the wave-1 page must already say, or the update replaces nothing.
    base_criteria: str


def _q(question_id: str, category: str, question: str, criteria: str, trap: str) -> HardQuestion:
    return HardQuestion(
        id=question_id,
        category=category,
        question=question,
        ideal_query=question,
        answer_criteria=criteria,
        must_not_claim=trap,
    )


CASES: tuple[DeltaUpdate, ...] = (
    DeltaUpdate(
        question=_q(
            "du-oncall-departed",
            "linked_by_page",
            "Who should be paged for a payments-api incident?",
            "Does not present Dana Reyes as someone to page now: it says she has left the company, or "
            "drops her, or names Marco Silva as the one to page.",
            "Dana Reyes is currently the person to page for payments-api incidents, with no mention that she has left",
        ),
        base=(
            "Dana Reyes is the primary on-call engineer for payments-api incidents.",
            "Unanswered payments-api pages escalate to Marco Silva after 15 minutes.",
        ),
        update=("Dana Reyes left the company on 30 June 2026 and her accounts were deactivated.",),
        base_criteria="Names Dana Reyes as the primary on-call engineer for payments-api.",
    ),
    DeltaUpdate(
        question=_q(
            "du-renamed-service",
            "second_hop",
            "What platform does the search-indexer service run on?",
            "States that the service (now called lexicon) runs on AWS Lambda since June 2026. Mentioning "
            "Kubernetes as where it ran before is fine.",
            "the search-indexer service currently runs on Kubernetes",
        ),
        base=("The search-indexer service runs on the shared Kubernetes cluster in us-east-1.",),
        update=(
            "The search-indexer service was renamed to lexicon on 2 May 2026.",
            "lexicon was moved off Kubernetes onto AWS Lambda on 10 June 2026.",
        ),
        base_criteria="States that the search-indexer service runs on Kubernetes.",
    ),
    DeltaUpdate(
        question=_q(
            "du-backup-correction",
            "correction",
            "How long are database backups retained?",
            "States that database backups are retained for 14 days. Mentioning that 30 days was an "
            "earlier, wrong figure is fine.",
            "database backups are retained for 30 days",
        ),
        base=("Database backups are retained for 30 days.",),
        update=(
            "The infra review found the 30-day backup figure was never true: backups have always been "
            "kept for 14 days.",
        ),
        base_criteria="States that database backups are retained for 30 days.",
    ),
    DeltaUpdate(
        question=_q(
            "du-enterprise-plan",
            "membership",
            "Which customers are on the Enterprise plan?",
            "Lists exactly Harbor Mills, Quanta Rail and Orbital Freight as the Enterprise customers. "
            "Kestrel Labs may appear only as a former Enterprise customer.",
            "Kestrel Labs is currently on the Enterprise plan",
        ),
        base=(
            "Harbor Mills is on the Enterprise plan.",
            "Quanta Rail is on the Enterprise plan.",
            "Kestrel Labs is on the Enterprise plan.",
            "Pinewood Dental is on the Team plan.",
        ),
        update=(
            "Orbital Freight signed an Enterprise plan contract on 4 June 2026.",
            "Kestrel Labs moved down to the Team plan on 12 June 2026.",
        ),
        base_criteria="Lists Harbor Mills, Quanta Rail and Kestrel Labs as Enterprise customers.",
    ),
    DeltaUpdate(
        question=_q(
            "du-launch-risk",
            "new_item",
            "What are the known risks to the September 2026 launch?",
            "Includes, among the risks, that the payment SDK vendor ends support for the SDK on 1 September "
            "2026, before the launch. Also keeps the load-test risk.",
            "",
        ),
        base=(
            "The launch is scheduled for 15 September 2026.",
            "A known risk for the September launch: the load tests have only reached 60% of the expected peak traffic.",
        ),
        update=(
            "The vendor of the payment SDK we ship with announced it ends support for that SDK on 1 September 2026.",
        ),
        base_criteria="Names the load tests reaching only 60% of peak traffic as a launch risk.",
    ),
    DeltaUpdate(
        question=_q(
            "du-quiet-wave",
            "quiet_wave",
            "Which region hosts the primary analytics warehouse?",
            "States that the primary analytics warehouse is in eu-west-1.",
            "the primary analytics warehouse is in a region other than eu-west-1",
        ),
        base=("The primary analytics warehouse is hosted in eu-west-1.",),
        update=(),
        base_criteria="States that the primary analytics warehouse is in eu-west-1.",
    ),
)

MINIMUM_ACCEPTANCE = ("du-oncall-departed",)


def _waves() -> list[list[dict[str, str]]]:
    """Every case's facts plus the hard corpus, so each page shares its bank with
    plenty that has nothing to do with it."""
    noise = split_into_waves(facts())
    waves: list[list[dict[str, str]]] = []
    for index, timestamp in enumerate((_BASE_TIME, _UPDATE_TIME)):
        texts = [t for case in CASES for t in (case.base if index == 0 else case.update)]
        texts += [f.text for f in noise[index]]
        waves.append([{"content": t, "timestamp": timestamp} for t in texts])
    return waves


async def build(client: Hindsight, bank_id: str, case: DeltaUpdate, settle: SettleFn) -> PageOutcome:
    """Write the page from wave 1, update it from wave 2, and snapshot each."""
    outcome = PageOutcome(question_id=case.question.id, category=case.question.category, bank_id=bank_id)
    await prepare_bank(client, bank_id)
    created = await create_page(client, bank_id, case.question)
    outcome.page_id, outcome.mental_model_id = created.page_id, created.mental_model_id
    await settle(bank_id)

    for index, wave in enumerate(_waves(), start=1):
        await client.aretain_batch(bank_id=bank_id, items=wave)
        await settle(bank_id)
        await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=outcome.mental_model_id)
        await settle(bank_id)
        content = await read_page(client, bank_id, outcome.mental_model_id)
        outcome.waves.append(WaveSnapshot(wave=index, facts_ingested=len(wave), content=content))

    outcome.final_content = outcome.waves[-1].content
    return outcome
