"""The wire formats a conversation is retained in — the same checks run against each one.

- ``jsonl``: one message object per line. The format integrations send (coding-agents, Hermes).
- ``flat``: one JSON array of message objects, also accepted and merged on append.

Not supported: a JSON array of ``[user, assistant]`` arrays. It is chunked as plain text, so a
message is cut mid-JSON, and an append never merges into the stored body.

A format is only correct if the chunks extraction sees are the same whether the conversation
arrived in one retain or was built up by ``update_mode="append"`` batches, and if no message is
ever cut in half. The case that made this matter (#6645): a terse reply ("option 1") that opens
an append batch was extracted in a chunk that did not contain the assistant question it answers.
"""

import dataclasses
import json
from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from hindsight_api.engine.retain.fact_extraction import chunk_text

CHUNK = 1000

Message = dict[str, str]
Turn = tuple[Message, Message]


def _flat(turns: list[Turn]) -> str:
    return json.dumps([m for t in turns for m in t], ensure_ascii=False)


def _jsonl(turns: list[Turn]) -> str:
    return "\n".join(json.dumps(m, ensure_ascii=False) for t in turns for m in t)


FORMATS: dict[str, Callable[[list[Turn]], str]] = {"flat": _flat, "jsonl": _jsonl}


def _turn(i: int, user_len: int = 150, assistant_len: int = 450) -> Turn:
    # Distinct words per turn so every message is identifiable inside a chunk.
    return (
        {"role": "user", "content": f"U{i}: " + f"question{i} " * (user_len // 11)},
        {"role": "assistant", "content": f"A{i}: " + f"answer{i} " * (assistant_len // 9)},
    )


def _messages_in(chunk: str) -> list[Message]:
    """Every whole message object in a chunk; raises if the chunk is not whole JSON messages."""
    chunk = chunk.strip()
    if chunk.startswith("["):
        return json.loads(chunk)
    return [json.loads(line) for line in chunk.splitlines() if line.strip()]


@pytest.fixture(params=list(FORMATS), ids=list(FORMATS))
def fmt(request) -> str:
    return request.param


# ---------------------------------------------------------------------------
# Pure chunking: what chunk_text does to one body
# ---------------------------------------------------------------------------


def test_no_message_is_cut_across_chunks(fmt):
    turns = [_turn(i) for i in range(8)]
    chunks = chunk_text(FORMATS[fmt](turns), CHUNK)
    assert len(chunks) > 1
    seen = [m for c in chunks for m in _messages_in(c)]
    assert seen == [m for t in turns for m in t]


def test_every_chunk_fits_the_budget(fmt):
    turns = [_turn(i) for i in range(8)]
    for c in chunk_text(FORMATS[fmt](turns), CHUNK):
        assert len(c) <= CHUNK


def test_a_message_longer_than_a_chunk_is_split_but_nothing_is_lost(fmt):
    turns = [_turn(0), _turn(1, assistant_len=3 * CHUNK), _turn(2)]
    body = FORMATS[fmt](turns)
    chunks = chunk_text(body, CHUNK)
    for c in chunks:
        assert len(c) <= CHUNK
    joined = "".join(chunks)
    for t in turns:
        for m in t:
            for word in set(m["content"].split()):
                assert word in joined


def test_non_ascii_survives_chunking(fmt):
    turns = [
        ({"role": "user", "content": "Ciao, perché? 🙂 " * 30}, {"role": "assistant", "content": "日本語 " * 120})
        for _ in range(4)
    ]
    seen = [m for c in chunk_text(FORMATS[fmt](turns), CHUNK) for m in _messages_in(c)]
    assert seen == [m for t in turns for m in t]


# ---------------------------------------------------------------------------
# Through the engine: one retain vs the same conversation built by appends
# ---------------------------------------------------------------------------


@pytest.fixture
def _small_chunks(memory, monkeypatch):
    # The engine copies the server config once at start, so an env var set here would not reach it.
    config = memory._config_resolver._global_config
    small = dataclasses.replace(
        config, retain_chunk_size=CHUNK, enable_auto_consolidation=False, enable_observations=False
    )
    monkeypatch.setattr(memory._config_resolver, "_global_config", small)


def _bank(name: str) -> str:
    return f"test_convfmt_{name}_{datetime.now(timezone.utc).timestamp()}"


async def _retain(memory, request_context, bank_id: str, doc: str, content: str, append: bool) -> None:
    item = {"content": content, "context": "conversation", "document_id": doc}
    if append:
        item["update_mode"] = "append"
    await memory.retain_batch_async(bank_id=bank_id, contents=[item], request_context=request_context)


async def _chunks(memory, request_context, bank_id: str, doc: str) -> list[str]:
    listing = await memory.list_document_chunks(bank_id, doc, limit=1000, request_context=request_context)
    return [row["chunk_text"] for row in listing["items"]]


async def _build(memory, request_context, fmt: str, batches: list[list[Turn]]) -> tuple[str, str]:
    bank_id, doc = _bank(fmt), "conv"
    for i, batch in enumerate(batches):
        await _retain(memory, request_context, bank_id, doc, FORMATS[fmt](batch), append=i > 0)
    return bank_id, doc


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_a_reply_that_opens_a_batch_is_chunked_with_its_question(memory, request_context, fmt, _small_chunks):
    """#6645: the short reply and the question it answers fit one chunk, so they must share one."""
    question = (
        {"role": "user", "content": "Plan the release."},
        {"role": "assistant", "content": "Two options: 1) ship Friday 2) wait for the fix. Which one?"},
    )
    reply = ({"role": "user", "content": "option 1"}, {"role": "assistant", "content": "Shipping Friday."})
    bank_id, doc = await _build(memory, request_context, fmt, [[question], [reply]])
    try:
        chunks = await _chunks(memory, request_context, bank_id, doc)
        holding_reply = [c for c in chunks if "option 1" in c]
        assert len(holding_reply) == 1
        assert "Which one?" in holding_reply[0], chunks
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_appended_conversation_chunks_like_one_retain(memory, request_context, fmt, _small_chunks):
    """Batching is a transport detail: N appends must leave the chunks one retain would."""
    turns = [_turn(i) for i in range(9)]
    one_bank, doc = await _build(memory, request_context, fmt, [turns])
    app_bank, _ = await _build(memory, request_context, fmt, [turns[0:3], turns[3:6], turns[6:9]])
    try:
        assert await _chunks(memory, request_context, app_bank, doc) == await _chunks(
            memory, request_context, one_bank, doc
        )
    finally:
        await memory.delete_bank(one_bank, request_context=request_context)
        await memory.delete_bank(app_bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_appended_document_text_equals_one_retain(memory, request_context, fmt, _small_chunks):
    """The stored body after appends must be the same text, in the same format, as one retain."""
    turns = [_turn(i) for i in range(6)]
    bank_id, doc = await _build(memory, request_context, fmt, [turns[0:2], turns[2:4], turns[4:6]])
    try:
        stored = await memory.get_document(doc, bank_id, request_context=request_context)
        assert stored["original_text"] == FORMATS[fmt](turns)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
