"""A refresh whose document comes back in the wrong shape is re-asked, not stored.

The refresh is the one path that stores a *document*: the agent states its
structure and the markdown is rendered from it, so the model never writes the
text that gets persisted (#3361). That only holds while the structure is
actually a structure. When a model emitted a block as `{"text": "..."}` where
the schema asks for a plain string, the tolerant coercion ran it through
`str()` — the Python repr — and the literal `{'text': '...'}` became the
block's markdown. It reached `mental_models.content`, and from there every
prompt built from that content, until the next full refresh happened to comply
(#4910).

This is the seam no single-mechanism test crosses: the shape arrives in reflect
and the damage shows up in what a later read returns. So the story is told from
the outside — script the refresh's `done` call with the wrong shape, then read
the model back and look at what a consumer would get.
"""

from __future__ import annotations

from typing import Any

import pytest

from hindsight_system_tests import reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

SOURCE_QUERY = "Where does Alice live?"
RETRIEVING_QUERY = "Alice Berlin"
HEADING = "Housing"
BODY = "Alice lives in Berlin."
RENDERED = f"## {HEADING}\n\n{BODY}"

# The shape the schema asks for, and the shape the model sent instead.
GOOD_BLOCKS: list[Any] = [BODY]
OBJECT_BLOCKS: list[Any] = [{"text": BODY}]


def _done_with_blocks(llm, *blocks_per_call: list[Any]) -> None:
    """Drive the refresh's ``done`` call, with a different document each time.

    Registered before ``reflect_loop`` so it claims the ``done`` turn (rules match
    in registration order) while the search ladder is still driven by the shared
    helper. One entry per expected call; the last entry answers any call beyond
    them, so an unexpected extra re-ask shows up as a wrong *answer*, not as a
    missing rule the stub would fail on for the wrong reason.
    """
    calls = iter(range(len(blocks_per_call)))

    def build(_request):
        from hindsight_system_tests.rulebook import ToolCall

        index = next(calls, len(blocks_per_call) - 1)
        return ToolCall(
            name="done",
            arguments={"document": {"sections": [{"heading": HEADING, "level": 2, "blocks": blocks_per_call[index]}]}},
        )

    llm.on_step("reflect", tool="done").answers_with_tool_call(build)
    # RETRIEVING_QUERY, not the helper's default: the loop refuses `done` until a
    # search has actually returned something, and a done call refused for *that*
    # reason never reaches the document -- it would spend a scripted reply on a
    # turn this story is not about.
    reflect_loop(llm, answer=BODY, query=RETRIEVING_QUERY)


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=BODY, query=RETRIEVING_QUERY)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


async def _read(client, bank: str, model_id: str):
    return await client.mental_models.get_mental_model(bank, model_id, detail="full")


async def test_a_document_with_object_blocks_is_re_asked_and_the_retry_is_stored(
    client, llm, bank_with_facts, settled
):
    """The whole point: no repr reaches the stored content, and the answer survives.

    The first ``done`` carries the wrong shape and the second carries the right
    one, so the refresh completes — the model is told what was wrong and fixes
    it, rather than the refresh being thrown away or the repr being stored.
    """
    llm.reset()
    _done_with_blocks(llm, OBJECT_BLOCKS, GOOD_BLOCKS)

    created = await client.mental_models.create_mental_model(
        bank_with_facts, {"name": "Housing", "source_query": SOURCE_QUERY}
    )
    await settled(bank_with_facts)

    model = await _read(client, bank_with_facts, created.mental_model_id)
    assert model.content.strip() == RENDERED
    assert "{'text'" not in model.content, "the Python repr of a block object reached the stored content"

    # The rejection was handed back to the model, which is why the retry complied.
    retry_prompts = [p for p in llm.prompts_for("reflect") if "sections[0].blocks[0]" in p]
    assert retry_prompts, "the model was re-asked without being told what was wrong"


async def test_a_model_that_never_sends_a_document_fails_instead_of_storing_the_repr(
    client, llm, bank_with_facts
):
    """Out of re-asks, the refresh fails and the model keeps the content it had.

    Failing is the right outcome: a mental model with no content is visibly
    unfinished, whereas one holding `{'text': '...'}` reads as a real answer to
    every consumer that injects it into a prompt.
    """
    llm.reset()
    _done_with_blocks(llm, OBJECT_BLOCKS, OBJECT_BLOCKS, OBJECT_BLOCKS)

    created = await client.mental_models.create_mental_model(
        bank_with_facts, {"name": "Housing", "source_query": SOURCE_QUERY}
    )
    await wait_until_settled(client, bank_with_facts, allow_failed=True)

    model = await _read(client, bank_with_facts, created.mental_model_id)
    assert "{'text'" not in (model.content or ""), "a refused document was stored anyway"
    assert not (model.content or "").strip(), "the refresh should have failed, leaving the model without content"
