"""Extraction must not fill a fact's metadata with something the text never said.

The knowledge-page and reflect suites grade what comes OUT of the bank. This one
grades what goes IN, because a fabrication at retain time is the worst kind:
reflect can be asked again, but an invented date is written once and then read
back as fact by every later answer. Nothing downstream can tell it from a real
one. Its sibling next door (``test_03``) asks whether a fact is written in the
right language; this one asks whether it is true.

The incident (#4457): `when` / `where` / `who` / `why` were required non-null
strings, and under strict structured output every property is required — so a
model had no legal way to say "the text doesn't state this". Asked for a string,
a smaller model supplies the most plausible one in sight, which is whatever the
surrounding text mentions. A document whose first line carried a project start
date produced a *requirement* fact dated to it.
``HINDSIGHT_API_RETAIN_OPTIONAL_FACT_DIMENSIONS`` lets an operator make those
four values nullable, so a model has a legal way to say "not stated".

This suite sets nothing: it measures the server as configured, which on CI means
the default (required, "N/A") path — the one worth watching, since it is what
every deployment runs until someone opts in. A server started with the flag on
runs the same cases against its own setting.

It verifies behaviour; it is not a backwards-compatibility guard, and the
difference was measured rather than assumed. Qwen3.6-35B under strict schema
passes all three cases WITH the nullable fields and, re-run identically, with
the old required-non-null ones too: the pressure to invent a value only bites a
model weak enough to feel it, and the reported case was a 9B. So a green run
here means "extraction is sound on this model", not "the regression cannot come
back" — the test that fails on that is the schema test in
``hindsight-api-slim/tests/test_fact_extraction_nullable_dimensions.py``.

Graded trap-first, like every other suite here: the **trap** (a fact asserts the
metadata the text never gave it) before **correctness** (the fact was extracted
at all). The trap is the one that matters — a miss costs a memory, a fabrication
creates a false one.

The bank runs real extraction on purpose. The corpus suites set
``retain_extraction_mode="chunks"`` to skip the model and keep the corpus text
exactly as authored; here, as in the language suite, the extraction call IS the
thing under test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import evaluate
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FidelityCase:
    """A document that baits one dimension, and what must not come back.

    Every document states the baited value for a DIFFERENT subject, close by and
    in plain sight. That is the whole difficulty: the value is not absent from
    the document, it is absent from the fact — which is exactly the distinction a
    model under pressure to emit a non-null string cannot afford to make.
    """

    id: str
    category: str
    document: str
    #: The fact that must survive extraction. Correctness, graded second.
    must_extract: str
    #: Which stored fact the trap is about. The judge is pointed at this one and
    #: told to read it alone — see ``_run`` for why that scoping is load-bearing.
    subject: str
    #: What that fact must not have attached to it. The trap.
    must_not_carry: str


_CASES = {
    case.id: case
    for case in (
        FidelityCase(
            id="date-borrowed-from-nearby-subject",
            category="fabricated_when",
            document=(
                "Fieldwork planning note.\n\n"
                "The project has a target start of September 14.\n"
                "Customer notifications must be completed before fieldwork begins.\n"
                "The equipment audit finished last quarter."
            ),
            must_extract="customer notifications have to be completed before fieldwork begins",
            subject="the customer notifications",
            must_not_carry="carries a date or deadline of its own, such as September 14",
        ),
        FidelityCase(
            id="owner-borrowed-from-nearby-subject",
            category="fabricated_who",
            document=(
                "Migration status.\n\n"
                "Priya owns the schema migration and signed off on the rollback plan.\n"
                "The connection-pool limit still needs to be raised before launch.\n"
                "Both items are tracked in the launch checklist."
            ),
            must_extract="the connection-pool limit still needs to be raised before launch",
            subject="raising the connection-pool limit",
            must_not_carry="names a person as its owner, assignee, or the one responsible, such as Priya",
        ),
        FidelityCase(
            id="place-borrowed-from-nearby-subject",
            category="fabricated_where",
            document=(
                "Operations digest.\n\n"
                "The Rotterdam depot switched to the new intake process in June.\n"
                "Night-shift handovers are now recorded in writing.\n"
                "Fuel costs were flat across the network."
            ),
            must_extract="night-shift handovers are now recorded in writing",
            subject="the written night-shift handovers",
            must_not_carry="ties them to a named location, such as the Rotterdam depot",
        ),
    )
}

#: The reported regression. The rest run with ``--full``.
MINIMUM_ACCEPTANCE = ("date-borrowed-from-nearby-subject",)


async def _extracted_facts(client: Hindsight, bank_id: str) -> str:
    """Every stored fact with the metadata that travels with it.

    A borrowed date can surface in two places — inside the fact text the
    extractor composed (``... | When: ...``) or in the stored ``occurred_*``
    timestamps — and either one is a fabrication a later answer will repeat. So
    the judge is shown both rather than the text alone.
    """
    listed = await client.alist_memories(bank_id, limit=100)
    lines = []
    for item in listed.items:
        if item.fact_type not in ("world", "experience"):
            continue
        dates = " ".join(
            f"{label}={value}"
            for label, value in (("occurred_start", item.occurred_start), ("occurred_end", item.occurred_end))
            if value
        )
        lines.append(f"- {item.text}{f'  [{dates}]' if dates else ''}")
    return "\n".join(lines)


@dataclass
class FidelityOutcome:
    """The graded record plus the facts it was graded on, for the failure message."""

    record: EvalRecord
    facts: str


async def _run(client: Hindsight, bank_id: str, settled: SettleFn, case_id: str) -> FidelityOutcome:
    case = _CASES[case_id]
    # Observations and consolidation off for the same reason as every other suite
    # here — they cost model time and write rows this eval never reads. The
    # extraction mode is left at the default, because that call is the subject.
    #
    # Nothing is opted into: this measures the server as configured, which on CI
    # means the default (required, "N/A") path. That is the one worth watching —
    # a deployment that turns HINDSIGHT_API_RETAIN_OPTIONAL_FACT_DIMENSIONS on
    # runs the same suite against its own setting, because the flag is
    # server-level and the eval reads whatever the server was started with.
    await client.aupdate_bank_config(bank_id, enable_observations=False, enable_auto_consolidation=False)
    await client.aretain(bank_id=bank_id, content=case.document)
    await settled(bank_id)

    facts = await _extracted_facts(client, bank_id)
    assert facts, f"{case_id}: retain stored no facts at all — bank {bank_id}"

    # The trap is graded on ONE fact, read alone, and without showing the judge
    # the source document. Both halves of that are load-bearing, and the first
    # version of this eval got them wrong: asked whether any fact "states or
    # implies" the borrowed value, the judge combined two correctly-extracted
    # facts ("notifications precede fieldwork" + "the project starts September
    # 14") and reported a fabrication that no stored fact contained. Facts are
    # stored separately and retrieved separately, so what a reader could infer by
    # putting two of them side by side is not what this measures — the question
    # is only ever whether THIS fact carries metadata of its own that its source
    # never gave it. Showing the document invites the same inference, because the
    # borrowed value is in the document by construction.
    trap = await evaluate(
        facts,
        f"Consider ONLY the fact about {case.subject}, read entirely on its own. "
        f"That single fact {case.must_not_carry}. "
        "Judge that fact in isolation: do not combine it with any other fact in the list, and do not "
        "reason about what the facts imply together. A value appearing only in a different fact does "
        "not count.",
    )
    correct = await evaluate(
        facts,
        f"One of these facts records that {case.must_extract}.",
        context=f"These facts were extracted from this document:\n\n{case.document}",
    )

    record = EvalRecord(
        kind="retain",
        question_id=case_id,
        category=case.category,
        correct=correct.meets_criteria,
        hit_trap=trap.meets_criteria,
        bank_id=bank_id,
        reason=trap.reasoning if trap.meets_criteria else correct.reasoning,
    )
    RECORDED.append(record)
    log.info("%s: correct=%s trap=%s", case_id, record.correct, record.hit_trap)
    return FidelityOutcome(record=record, facts=facts)


def _assert_sound(outcome: FidelityOutcome) -> None:
    record = outcome.record
    evidence = f"bank {record.bank_id}\nfacts:\n{outcome.facts}"
    assert not record.hit_trap, (
        f"{record.question_id}: extraction invented metadata the document never gave this fact — "
        f"{record.reason}\n{evidence}"
    )
    assert record.correct, f"{record.question_id}: {record.reason}\n{evidence}"


@pytest.mark.parametrize("case_id", MINIMUM_ACCEPTANCE)
async def test_retain_fidelity_minimum_acceptance(
    client: Hindsight, bank_id: str, settled: SettleFn, case_id: str
) -> None:
    _assert_sound(await _run(client, bank_id, settled, case_id))


@pytest.mark.full
@pytest.mark.parametrize("case_id", [c for c in _CASES if c not in MINIMUM_ACCEPTANCE])
async def test_retain_fidelity_full(client: Hindsight, bank_id: str, settled: SettleFn, case_id: str) -> None:
    _assert_sound(await _run(client, bank_id, settled, case_id))
