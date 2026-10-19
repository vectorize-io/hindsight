"""A delta page refresh that asks a decision model which new memories to read.

With a decision model set up, a delta refresh skips the reflect loop. It reads the
memories new since its last run, asks the decision model of each whether it could change
the page, and hands only those to the edit. None kept means no LLM call at all. A bank
turns it off with ``mental_model_delta_use_decision_model``.

Each step is unit-tested on its own. What only a story shows is that they meet: the
page the gate is shown is the page the bank stored, the memories it is asked about are
the ones retain wrote after the last refresh, and what it keeps is what reaches the
edit and the refresh's report.

The decision model is server-level, so the server under test is ``decision_model_server``.
Its worker is off, so the main server builds the page and the story refreshes by dry run,
which runs in the gate's process and changes nothing between the stories.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from hindsight_system_tests import reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import add_section, consolidation, extracted, fact
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
ANSWER = "Alice lives in Berlin."
MOVED = "Alice moved to Munich"
UNRELATED = "The office printer is out of toner"


@dataclass(frozen=True)
class _Page:
    bank_id: str
    mental_model_id: str


@pytest.fixture
async def page(decision_model_client, client, llm) -> AsyncIterator[_Page]:
    """A page built from one fact, then two new facts it has not read yet.

    Asks for ``client`` only to have the main server running: the decision-model server's
    worker is off, so the main one extracts the facts and builds the page.
    """
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER, query=QUERY)
    await decision_model_client.aretain(bank_id=bank, content="Alice moved to Berlin.")
    await wait_until_settled(decision_model_client, bank)
    created = await decision_model_client.mental_models.create_mental_model(
        bank,
        {
            "name": "Home",
            "source_query": QUERY,
            "trigger": {"mode": "delta", "fact_types": ["world"], "exclude_mental_models": True},
        },
    )
    await wait_until_settled(decision_model_client, bank)

    llm.reset()
    llm.on_step("extract_facts").returns(
        extracted(fact(MOVED, who="Alice", entities=["Alice"]), fact(UNRELATED, entities=["printer"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await decision_model_client.aretain(bank_id=bank, content=f"{MOVED}. {UNRELATED}.")
    await wait_until_settled(decision_model_client, bank)
    llm.reset()
    yield _Page(bank_id=bank, mental_model_id=created.mental_model_id)
    with contextlib.suppress(Exception):
        await decision_model_client.banks.delete_bank(bank)


def _calls(llm, step: str) -> list[str]:
    return [call.all_text for call in llm.calls if anchor_for(step) in call.all_text]


async def test_only_the_memory_that_changes_the_page_reaches_the_edit(decision_model_client, llm, stubs, page):
    stubs.rerank.verdicts = {"Munich": 1.0, "printer": 0.0}
    llm.on_step("mental_model_delta").returns(add_section("Moves", "Alice moved to Munich."))

    result = await decision_model_client.mental_models.dry_run_refresh_mental_model(page.bank_id, page.mental_model_id)

    assert len(stubs.rerank.states) == 2, "one question per new memory, and the old one is not asked about again"
    assert all(ANSWER in state and QUERY in state for state in stubs.rerank.states), "the gate saw the stored page"
    assert not _calls(llm, "reflect"), "the gate replaces the reflect loop"
    edits = _calls(llm, "mental_model_delta")
    assert len(edits) == 1 and MOVED in edits[0] and UNRELATED not in edits[0]
    assert result.facts.retrieved == {"world": 2}
    assert result.facts.used == {"world": 1}
    # ``based_on`` also carries what earlier refreshes kept, so Berlin stays; the printer never arrives.
    grounding = " ".join(f["text"] for f in result.based_on["world"])
    assert MOVED in grounding and UNRELATED not in grounding
    assert result.would_persist is True
    assert result.effective_mode == "delta"
    assert result.outcome == "content_written"
    assert result.warnings == []
    assert ANSWER in result.preview_content and "Moves" in result.preview_content
    assert "Alice moved to Munich." in result.preview_content
    assert [op["op"] for op in result.delta_operations.applied] == ["add_section"]


async def test_nothing_that_changes_the_page_means_no_llm_call(decision_model_client, llm, stubs, page):
    stubs.rerank.verdicts = {"Munich": 0.0, "printer": 0.0}

    result = await decision_model_client.mental_models.dry_run_refresh_mental_model(page.bank_id, page.mental_model_id)

    assert len(stubs.rerank.states) == 2
    assert not llm.calls, "nothing kept, so nothing to pay an LLM for"
    assert result.outcome == "content_preserved_no_new_facts"
    assert result.facts.retrieved == {"world": 2}
    assert result.facts.used == {}
    assert result.would_persist is False
    assert result.preview_content == result.current_content
    assert result.warnings == []


async def test_a_bank_that_opts_out_refreshes_with_reflect(decision_model_client, llm, stubs, page):
    await decision_model_client.banks.update_bank_config(
        page.bank_id, {"updates": {"mental_model_delta_use_decision_model": False}}
    )
    llm.on_step("reflect_round_query").returns_text(QUERY)
    reflect_loop(llm, answer=ANSWER, query=QUERY)
    llm.on_step("mental_model_delta").returns(add_section("Moves", "Alice moved to Munich."))

    result = await decision_model_client.mental_models.dry_run_refresh_mental_model(page.bank_id, page.mental_model_id)

    assert not any("New memory: " in state for state in stubs.rerank.states), "the gate was not asked"
    assert _calls(llm, "reflect_round_query"), "the refresh ran reflect instead"
    assert result.effective_mode == "delta"
