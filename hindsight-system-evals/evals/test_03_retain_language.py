"""Retain must write each fact in the language of its input — never another one.

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
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)

#: Retains per case. The drift this guards against showed in roughly one run in six,
#: so a single retain would pass a regressed prompt most of the time.
RETAIN_REPEATS = int(os.getenv("HINDSIGHT_EVAL_RETAIN_REPEATS", "6"))


@dataclass(frozen=True)
class LanguageCase:
    id: str
    language: str
    context: str
    content: str


_CASES = {
    case.id: case
    for case in (
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
}

#: The case that regressed. Everything else runs with ``--full``.
MINIMUM_ACCEPTANCE = ("en-coding-session",)


@dataclass
class LanguageOutcome:
    case_id: str
    bank_id: str
    documents: int = 0
    empty: int = 0
    wrong_language: list[str] = field(default_factory=list)


async def _run(client: Hindsight, bank_id: str, settled: SettleFn, case_id: str) -> LanguageOutcome:
    case = _CASES[case_id]
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
        RECORDED.append(
            EvalRecord(
                kind="retain",
                question_id=document_id,
                category=f"language:{case.language}",
                correct=bool(facts) and not trap,
                hit_trap=trap,
                bank_id=bank_id,
                reason=reason,
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


def _assert_sound(outcome: LanguageOutcome) -> None:
    assert not outcome.wrong_language, (
        f"{outcome.case_id}: {len(outcome.wrong_language)}/{outcome.documents} retains wrote facts in another "
        f"language — bank {outcome.bank_id}\n" + "\n".join(outcome.wrong_language)
    )
    assert outcome.empty < outcome.documents, (
        f"{outcome.case_id}: no retain extracted any fact — bank {outcome.bank_id}"
    )


@pytest.mark.parametrize("case_id", MINIMUM_ACCEPTANCE)
async def test_retain_language_minimum_acceptance(
    client: Hindsight, bank_id: str, settled: SettleFn, case_id: str
) -> None:
    _assert_sound(await _run(client, bank_id, settled, case_id))


@pytest.mark.full
@pytest.mark.parametrize("case_id", [c for c in _CASES if c not in MINIMUM_ACCEPTANCE])
async def test_retain_language_full(client: Hindsight, bank_id: str, settled: SettleFn, case_id: str) -> None:
    _assert_sound(await _run(client, bank_id, settled, case_id))
