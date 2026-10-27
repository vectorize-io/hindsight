"""Retain: what the bank stores from what it was given.

Every case runs real extraction against the server's model, and a judge independent
of it grades the stored facts. Each case has a ``category`` and its test id is
``<category>/<case id>``, so ``-k "temporal/"`` runs one group and ``eval-table``
compares models per group:

* selectivity — what is worth storing (chatter stores nothing);
* completeness — exact figures, sensory and emotional detail;
* attribution — who is meant ("my roommate" -> Tamsin) and who did it (world vs
  experience, a speaker named in context or metadata);
* temporal — relative, coarse and open-ended dates; no invented ones;
* language — facts stay in the language and script of their input;
* fidelity — hedges, negation, corrections, the prompt's own examples, instructions
  inside content;
* no_invention — no date, owner or place borrowed from a neighbouring subject;
* mission, mode, entities, entity_labels, inputs — the bank's retain mission, the
  extraction modes, entities and labels, structured and long inputs.

Three runners, because three things are measured differently. The capability cases
(``hindsight_system_evals.retain_capabilities``, where each names its history) grade
one retain. The language cases retain the same input ``RETAIN_REPEATS`` times, since
the drift they guard against is a rate. The no-invention cases point the judge at ONE
fact read alone.

A failure is a signal to re-run, as everywhere in this package — but a trap is a
stored falsehood, so it is what to look at first.

Language — Retain must write each fact in the language of its input — never another one.

The incident (discussion #4283): coding-agent sessions held entirely in English came
back as Spanish, French or Russian facts and observations. Measured on gpt-5.6-luna
with the old "detect the language, then STRICTLY never switch" rule, an English
coding-session transcript produced wrong-language facts in ~18% of runs. Nothing in
the prompt was Spanish; the model got the detection step wrong.

Because the failure is a rate, one retain proves little: each case retains the same
input ``RETAIN_REPEATS`` times, each as its own document, and grades every document.
A wrong-language fact is the trap — it is a stored memory nobody can read — and any
trap fails the case. The non-English cases guard the other direction: a fix that
simply forces English would pass the English cases and silently translate these.

Only extraction runs: observations and consolidation are off, so every graded fact is
exactly what retain produced from its document.

No invention — Extraction must not fill a fact's metadata with something the text never said.

A fabrication at retain time is the worst kind: reflect can be asked again, but an
invented date is written once and then read back as fact by every later answer.
Nothing downstream can tell it from a real one.

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
import os
from collections import defaultdict
from dataclasses import dataclass, field

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import evaluate
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord, case_params
from hindsight_system_evals.retain_capabilities import CASES, Case, run_case

log = logging.getLogger(__name__)

# ── Capabilities ─────────────────────────────────────────────────────────────


async def _check(client: Hindsight, bank_id: str, settled: SettleFn, case: Case) -> None:
    outcome = await run_case(client, bank_id, settled, case)
    failed = outcome.missing + outcome.broken
    RECORDED.append(
        EvalRecord(
            kind="retain",
            question_id=case.id,
            category=case.category,
            correct=not failed,
            hit_trap=bool(outcome.trapped),
            bank_id=bank_id,
            reason="; ".join(outcome.trapped or failed) or "ok",
        )
    )
    log.info("%s [%s]:\n%s", case.id, case.category, outcome.rendered)
    evidence = f"bank {bank_id} ({case.origin})\nfacts:\n{outcome.rendered}"
    assert not outcome.trapped, f"{case.id}: stored a falsehood —\n" + "\n".join(outcome.trapped) + f"\n{evidence}"
    assert not failed, f"{case.id}:\n" + "\n".join(failed) + f"\n{evidence}"


@pytest.mark.parametrize("case", case_params(CASES, {c.id for c in CASES if c.minimum}))
async def test_retain(client: Hindsight, bank_id: str, settled: SettleFn, case: Case) -> None:
    await _check(client, bank_id, settled, case)


# ── Language, as a rate ──────────────────────────────────────────────────────

#: Retains per case. The drift this guards against showed in roughly one run in six,
#: so a single retain would pass a regressed prompt most of the time.
RETAIN_REPEATS = int(os.getenv("HINDSIGHT_EVAL_RETAIN_REPEATS", "6"))


@dataclass(frozen=True)
class LanguageCase:
    id: str
    language: str
    context: str
    content: str
    category: str = "language"


LANGUAGE_CASES = (
        # The shape that regressed: an English agent transcript dense with numbers,
        # tool names and paths, which leaves the language less obvious than prose.
        LanguageCase(
            id="en-coding-session",
            language="English",
            context="coding agent session",
            content=(
                "user: run the full validation on the selected work and tell me what passed\n"
                "assistant: Fresh validation complete. Server: 230 passed, 52 skipped. Portal production build "
                "succeeded and 55 portal tests passed (I reran with the newer installed Node runtime because the "
                "initial build picked up Node 17 from the shell). 17 shared native security checks passed, 9 "
                "customer security checks passed, Admin iOS simulator build succeeded (needed an approved "
                "temporary directory because the sandbox restricts Xcode's output dir). Fixed five "
                "whitespace-only issues and updated the deployment record to remove a stale status. The "
                "prior-agent recovery report still needs independent verification.\n"
            ),
        ),
        LanguageCase(
            id="en-git-commit",
            language="English",
            context="git commit-message history (last 1)",
            content=(
                "commit 196b69f\nDate: Wed Sep 9 2026\n\n    Initialize Codex git baseline\n\n"
                "    Co-authored-by: Codex <noreply@openai.com>\n"
            ),
        ),
        LanguageCase(
            id="it-coding-session",
            language="Italian",
            context="sessione di un coding agent",
            content=(
                "user: esegui la validazione completa sul lavoro selezionato e dimmi cosa è passato\n"
                "assistant: Validazione completata. Server: 230 test passati, 52 saltati. La build di produzione "
                "del portale è riuscita e sono passati 55 test del portale (ho rilanciato con il runtime Node più "
                "recente perché la prima build aveva preso Node 17 dalla shell). Sono passati i 17 controlli di "
                "sicurezza nativi condivisi e i 9 controlli di sicurezza del cliente. Ho corretto cinque problemi "
                "di soli spazi e aggiornato il registro di deploy.\n"
            ),
        ),
        LanguageCase(
            id="ja-personal-note",
            language="Japanese",
            context="",
            content=(
                "田中さんは先週東京から大阪に引っ越しました。新しい職場はパナソニックで、毎朝電車で通勤しています。"
                "週末は奈良で書道を教えています。"
            ),
        ),
)



@dataclass
class LanguageOutcome:
    case_id: str
    bank_id: str
    documents: int = 0
    empty: int = 0
    wrong_language: list[str] = field(default_factory=list)


async def _run_language(client: Hindsight, bank_id: str, settled: SettleFn, case: LanguageCase) -> LanguageOutcome:
    case_id = case.id
    await client.aupdate_bank_config(bank_id, enable_observations=False, enable_auto_consolidation=False)
    await client.aretain_batch(
        bank_id=bank_id,
        items=[
            {"content": case.content, "context": case.context, "document_id": f"{case_id}-{i}"}
            for i in range(RETAIN_REPEATS)
        ],
    )
    await settled(bank_id)

    by_document: dict[str, list[str]] = defaultdict(list)
    listed = await client.alist_memories(bank_id, limit=1000)
    for item in listed.items:
        if item.fact_type in ("world", "experience") and item.document_id:
            by_document[item.document_id].append(item.text or "")

    outcome = LanguageOutcome(case_id=case_id, bank_id=bank_id)
    for i in range(RETAIN_REPEATS):
        document_id = f"{case_id}-{i}"
        facts = by_document.get(document_id, [])
        outcome.documents += 1
        if not facts:
            outcome.empty += 1
            reason, trap = "retain extracted no facts", False
        else:
            # The judge, not a word list: the failure is "a different language", and the
            # facts legitimately carry English tool names and paths inside Italian prose.
            # Each fact renders as "<sentence> | When: … | Involving: …"; those trailing
            # fields are the server's own labels, so only the sentence is graded — the
            # first run flagged an Italian fact over its English "When:" field alone.
            verdict = await evaluate(
                "\n".join(f"- {fact}" for fact in facts),
                f"At least one fact's main sentence (the text before the first ' | ') is written in a "
                f"language other than {case.language}. Ignore the ' | When: …' and ' | Involving: …' "
                "fields, proper nouns, identifiers, code, file paths, numbers and quoted text.",
                context=f"The retained input was written in {case.language}.",
            )
            trap, reason = verdict.meets_criteria, verdict.reasoning
            if trap:
                outcome.wrong_language.append(f"{document_id}: {reason} :: {facts[0][:160]}")
    # One record per case, like every other case, so a category's count is its cases: the
    # repeats are how this case measures a rate, not cases of their own.
    RECORDED.append(
        EvalRecord(
            kind="retain",
            question_id=case_id,
            category=case.category,
            correct=not outcome.wrong_language and outcome.empty < outcome.documents,
            hit_trap=bool(outcome.wrong_language),
            bank_id=bank_id,
            reason=f"{len(outcome.wrong_language)}/{outcome.documents} wrong-language, {outcome.empty} empty",
        )
    )
    log.info(
        "%s: %d documents, %d wrong-language, %d empty",
        case_id,
        outcome.documents,
        len(outcome.wrong_language),
        outcome.empty,
    )
    return outcome


def _assert_language_sound(outcome: LanguageOutcome) -> None:
    assert not outcome.wrong_language, (
        f"{outcome.case_id}: {len(outcome.wrong_language)}/{outcome.documents} retains wrote facts in another "
        f"language — bank {outcome.bank_id}\n" + "\n".join(outcome.wrong_language)
    )
    assert outcome.empty < outcome.documents, (
        f"{outcome.case_id}: no retain extracted any fact — bank {outcome.bank_id}"
    )


@pytest.mark.parametrize("case", case_params(LANGUAGE_CASES, {"en-coding-session"}))
async def test_retain_language(client: Hindsight, bank_id: str, settled: SettleFn, case: LanguageCase) -> None:
    _assert_language_sound(await _run_language(client, bank_id, settled, case))


# ── No invention: one fact, read alone ───────────────────────────────────────


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


FIDELITY_CASES = (
        FidelityCase(
            id="date-borrowed-from-nearby-subject",
            category="no_invention",
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
            category="no_invention",
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
            category="no_invention",
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


async def _run_fidelity(client: Hindsight, bank_id: str, settled: SettleFn, case: FidelityCase) -> FidelityOutcome:
    case_id = case.id
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


def _assert_fidelity_sound(outcome: FidelityOutcome) -> None:
    record = outcome.record
    evidence = f"bank {record.bank_id}\nfacts:\n{outcome.facts}"
    assert not record.hit_trap, (
        f"{record.question_id}: extraction invented metadata the document never gave this fact — "
        f"{record.reason}\n{evidence}"
    )
    assert record.correct, f"{record.question_id}: {record.reason}\n{evidence}"


@pytest.mark.parametrize("case", case_params(FIDELITY_CASES, {"date-borrowed-from-nearby-subject"}))
async def test_retain_no_invention(client: Hindsight, bank_id: str, settled: SettleFn, case: FidelityCase) -> None:
    _assert_fidelity_sound(await _run_fidelity(client, bank_id, settled, case))
