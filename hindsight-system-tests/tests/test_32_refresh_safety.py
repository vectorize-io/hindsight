"""A refresh that finds nothing leaves the answer alone, and a dry run changes
nothing at all.

Both are guards against the same class of accident: a mental model losing a good
answer to a refresh that had nothing to say.

The empty-scope case is the sharp one. A model scoped to tags that currently
match no memories — a project not started yet, a tag renamed, a document deleted
— will run its refresh and retrieve nothing. If "no evidence" is written through
as "no content", a working answer is replaced by an empty one, and the only way
back is to notice and re-derive it. Silence has to mean "keep what you have".

The dry run is the other half: a way to see what a refresh *would* do before
letting it. That is only useful if it is genuinely read-only.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

ANSWER = "Alice lives in Berlin."


@pytest.fixture
async def model(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", tags=["housing"])
    await settled(bank_id)

    created = await client.mental_models.create_mental_model(
        bank_id, {"name": "Housing", "source_query": "Where does Alice live?"}
    )
    await settled(bank_id)
    return created.mental_model_id


async def _read(client, bank: str, model_id: str):
    return await client.mental_models.get_mental_model(bank, model_id, detail="full")


async def test_a_refresh_over_a_scope_that_matches_nothing_keeps_the_answer(client, llm, bank_id, settled):
    """The wipe guard.

    This model is scoped to a tag no memory carries, so its refresh retrieves
    nothing — the state of any model whose subject has not happened yet. The
    answer it already has must survive: writing "nothing retrieved" through as
    "no content" destroys work that cannot be recovered from the bank.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", tags=["housing"])
    await settled(bank_id)

    created = await client.mental_models.create_mental_model(
        bank_id, {"name": "Scoped", "source_query": "Where does Alice live?", "tags": ["housing"]}
    )
    await settled(bank_id)
    assert (await _read(client, bank_id, created.mental_model_id)).content.strip() == ANSWER

    # Retag the scope onto something the bank has never seen, then refresh.
    await client.mental_models.update_mental_model(
        bank_id, created.mental_model_id, {"tags": ["a-tag-nothing-carries"]}
    )
    llm.reset()
    reflect_loop(llm, answer="")
    await client.mental_models.refresh_mental_model(bank_id, created.mental_model_id)
    await settled(bank_id)

    current = await _read(client, bank_id, created.mental_model_id)
    assert current.content.strip() == ANSWER, "an empty refresh wiped an answer it could not replace"


async def test_a_sibling_page_does_not_stand_in_for_an_empty_memory_scope(client, llm, bank_id, settled):
    """The wipe guard, with another page in reach (#5728).

    The same empty scope, but the bank now holds a second model the refresh can
    read. Pages alone cannot tell this model anything about its topic: running the
    loop over them is how a self-hosted bank got "there are no records" written over
    a good answer. Built from memories, the answer stays until memories come back.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", tags=["housing"])
    await settled(bank_id)

    scoped = await client.mental_models.create_mental_model(
        bank_id, {"name": "Scoped", "source_query": "Where does Alice live?", "tags": ["housing"]}
    )
    sibling = await client.mental_models.create_mental_model(
        bank_id, {"name": "Sibling", "source_query": "Where is Alice?", "tags": ["housing"]}
    )
    await settled(bank_id)
    assert (await _read(client, bank_id, sibling.mental_model_id)).content.strip() == ANSWER

    # Both move onto a tag no memory carries: the sibling stays readable to the
    # scoped model, while the scoped model's memories are all out of reach.
    for model_id in (scoped.mental_model_id, sibling.mental_model_id):
        await client.mental_models.update_mental_model(bank_id, model_id, {"tags": ["a-tag-nothing-carries"]})
    await settled(bank_id)
    llm.reset()
    reflect_loop(llm, answer="There are no records about where Alice lives.")
    await client.mental_models.refresh_mental_model(bank_id, scoped.mental_model_id)
    await settled(bank_id)

    current = await _read(client, bank_id, scoped.mental_model_id)
    assert current.content.strip() == ANSWER, "a sibling page let an empty refresh overwrite the answer"


async def test_a_dry_run_reports_what_would_happen(client, llm, bank_id, model, settled):
    """The point of a dry run is the report, so it has to say something useful:
    which mode it would run in, what scope it would read, and whether it would
    write anything."""
    llm.reset()
    reflect_loop(llm, answer="Alice lives in Berlin and is settled there.")

    result = await client.mental_models.dry_run_refresh_mental_model(bank_id, model)

    assert result.mental_model_id == model
    assert result.effective_mode
    assert result.outcome
    assert result.scope is not None


async def test_a_dry_run_does_not_touch_the_stored_answer(client, llm, bank_id, model, settled):
    """Read-only, or it is not a dry run. Someone checking whether a refresh is
    safe must not perform the thing they were checking."""
    before = await _read(client, bank_id, model)

    llm.reset()
    reflect_loop(llm, answer="A completely different answer.")
    await client.mental_models.dry_run_refresh_mental_model(bank_id, model)

    after = await _read(client, bank_id, model)
    assert after.content == before.content
    assert after.last_refreshed_at == before.last_refreshed_at
    assert after.last_memory_seen_at == before.last_memory_seen_at


async def test_a_dry_run_leaves_no_history_entry(client, llm, bank_id, model, settled):
    """History records what actually changed. A dry run that logged itself would
    make the audit trail lie about what the model has said."""
    before = await client.mental_models.get_mental_model_history(bank_id, model)

    llm.reset()
    reflect_loop(llm, answer="A completely different answer.")
    await client.mental_models.dry_run_refresh_mental_model(bank_id, model)

    after = await client.mental_models.get_mental_model_history(bank_id, model)
    assert len(after) == len(before)
