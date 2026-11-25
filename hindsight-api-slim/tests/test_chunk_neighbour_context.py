"""A chunk is extracted with the end of the chunk before it and the start of the one after (#6645).

Deterministic half: what reaches the prompt. Whether the model then reads a reply against its
question is test_chunk_neighbour_context_llm.py and the chunk_boundary cases in hindsight-system-evals.
"""

import dataclasses
import json
from datetime import datetime, timezone

import pytest

from hindsight_api.engine.retain.fact_extraction import _build_user_message, _neighbour_context


def test_neighbours_are_trimmed_to_the_side_next_to_the_boundary():
    n = _neighbour_context(2, before="one two three four", after="five six seven eight")
    assert (n.before, n.after) == (" three four", "five six")


def test_zero_turns_the_context_off():
    n = _neighbour_context(0, before="abcdefgh", after="12345678")
    assert (n.before, n.after) == ("", "")


def test_the_prompt_shows_both_neighbours_around_the_content():
    message = _build_user_message(
        "option 1", 1, 3, None, "", context_before="Which one? 1) ship 2) wait", context_after="Shipping."
    )
    before, content, after = (
        message.index("Which one? 1) ship 2) wait"),
        message.index("option 1"),
        message.index("Shipping."),
    )
    assert before < content < after
    assert message.count("CONTEXT ONLY —") == 2


def test_no_neighbours_leaves_the_prompt_unchanged():
    assert "CONTEXT ONLY" not in _build_user_message("option 1", 0, 1, None, "")


# ---------------------------------------------------------------------------
# Through the engine: the streaming path and the append (delta) path pass the neighbours
# ---------------------------------------------------------------------------

_QUESTION = "Which flat? A) Via Po for 1450 B) Via Nizza for 1250."


@pytest.fixture
def _context_on(memory, monkeypatch):
    # The engine copies the server config once at start, so an env var set here would not reach it.
    config = memory._config_resolver._global_config
    small = dataclasses.replace(
        config,
        retain_chunk_size=1000,
        retain_chunk_context_tokens=50,
        enable_auto_consolidation=False,
        enable_observations=False,
    )
    monkeypatch.setattr(memory._config_resolver, "_global_config", small)


def _conversation(*messages: str) -> str:
    return json.dumps([{"role": "user", "content": m} for m in messages])


def _extraction_prompts(memory) -> list[str]:
    calls = memory._retain_llm_config._provider_impl.get_mock_calls()
    return [c["messages"][-1]["content"] for c in calls if c.get("scope") == "retain_extract_facts"]


def _prompt_for(prompts: list[str], text: str) -> str:
    """The prompt whose Content (not its context) holds ``text``."""
    matching = [p for p in prompts if text in p.split("\nContent:\n", 1)[1].split("AFTER THIS CHUNK", 1)[0]]
    assert len(matching) == 1, prompts
    return matching[0]


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_a_retained_chunk_is_extracted_with_the_end_of_the_chunk_before(memory, request_context, _context_on):
    bank_id = f"test_neighbours_one_{datetime.now(timezone.utc).timestamp()}"
    seen = len(_extraction_prompts(memory))
    try:
        body = _conversation("filler " * 250, _QUESTION, "the cheaper one")
        item = {"content": body, "context": "conversation", "document_id": "conv"}
        await memory.retain_batch_async(bank_id=bank_id, contents=[item], request_context=request_context)
        prompt = _prompt_for(_extraction_prompts(memory)[seen:], "the cheaper one")
        assert "BEFORE THIS CHUNK" in prompt and _QUESTION in prompt
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_an_appended_reply_is_extracted_with_the_turn_it_answers(memory, request_context, _context_on):
    bank_id = f"test_neighbours_append_{datetime.now(timezone.utc).timestamp()}"
    try:
        first = {"content": _conversation("filler " * 250, _QUESTION), "context": "conversation", "document_id": "c"}
        await memory.retain_batch_async(bank_id=bank_id, contents=[first], request_context=request_context)
        seen = len(_extraction_prompts(memory))
        reply = {
            "content": _conversation("the cheaper one " + "ok " * 300),
            "context": "conversation",
            "document_id": "c",
            "update_mode": "append",
        }
        await memory.retain_batch_async(bank_id=bank_id, contents=[reply], request_context=request_context)
        prompt = _prompt_for(_extraction_prompts(memory)[seen:], "the cheaper one")
        assert "BEFORE THIS CHUNK" in prompt and _QUESTION in prompt
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
