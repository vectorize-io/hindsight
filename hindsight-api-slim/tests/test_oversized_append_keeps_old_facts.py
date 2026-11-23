"""An append whose TAIL alone exceeds retain_batch_tokens must not re-extract the stored body (#6766).

The tail is cut into slices. When the stored document is a JSON conversation and the tail is one
huge message, the first slice is a piece of that message, not a JSON array. Gluing it onto the
stored body with a newline made the whole body invalid JSON, so it re-chunked as prose, no stored
chunk matched, and delta deleted and re-extracted every fact in the document.
"""

import json
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache

_DOC = "openclaw:agent:main:main"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "2000")
    clear_config_cache()
    yield
    clear_config_cache()


def _turn(i: int, fmt: str, repeats: int = 60) -> str:
    text = f"Turn {i:03d} MARK{i:03d}. " + f"MARK{i:03d} talks about topic {i}. " * repeats
    if fmt == "jsonl":
        return json.dumps({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    if fmt == "openclaw":
        # What the OpenClaw plugin sends: one JSON array of the turn's messages.
        return json.dumps([{"role": "user", "content": text}, {"role": "assistant", "content": f"ok {i}"}])
    return f"[user]: {text}"


async def _append(memory, request_context, bank_id: str, content: str, turn: int) -> None:
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {"content": content, "document_id": _DOC, "update_mode": "append", "metadata": {"turn_index": str(turn)}}
        ],
        request_context=request_context,
    )


async def _ids(memory, request_context, bank_id: str) -> set[str]:
    units = await memory.list_memory_units(bank_id, document_id=_DOC, limit=10000, request_context=request_context)
    return {u["id"] for u in units["items"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["text", "jsonl", "openclaw"])
async def test_oversized_append_keeps_old_facts(memory, request_context, fmt: str):
    bank_id = f"test_6766_{fmt}_{datetime.now(timezone.utc).timestamp()}"
    try:
        # Grow the document turn by turn, the way OpenClaw does: each append is small.
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": _turn(0, fmt), "document_id": _DOC}],
            request_context=request_context,
        )
        for i in range(1, 12):
            await _append(memory, request_context, bank_id, _turn(i, fmt), i)
        before = await _ids(memory, request_context, bank_id)
        assert before

        # One huge turn: its tail alone is several times retain_batch_tokens.
        await _append(memory, request_context, bank_id, _turn(99, fmt, repeats=900), 99)
        lost = before - await _ids(memory, request_context, bank_id)
        assert not lost, f"{len(lost)}/{len(before)} old facts were deleted and re-extracted"

        # The next ordinary turn must not redo the document either.
        await _append(memory, request_context, bank_id, _turn(100, fmt), 100)
        lost = before - await _ids(memory, request_context, bank_id)
        assert not lost, f"{len(lost)}/{len(before)} old facts were re-extracted by the turn after"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
