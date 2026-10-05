"""Reflect output cites real ids, never the per-run aliases the model reads (#4876).

The prompt shows the model short aliases (``f1``, ``o2``) instead of UUIDs. They
mean nothing once the reflect ends, so a caller asking for ids — a mental model
used as a report that drives ``update_memory`` / ``invalidate_memory`` — must get
ids it can act on. Which ids the model chooses to cite is its call, so the
assertions are structural: no alias survives, and every UUID cited is real.
"""

import re
import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine

pytestmark = pytest.mark.hs_llm_core

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_ALIAS_RE = re.compile(r"\b[fopc]\d+\b")
_QUERY = (
    "Which memories about Alice's employer contradict or supersede each other? Cite the id of every memory you mention."
)


def _assert_cites_real_ids(text: str, real_ids: set[str]) -> None:
    leftover = _ALIAS_RE.findall(_UUID_RE.sub("", text))
    assert not leftover, f"aliases leaked into the output: {leftover}\n{text}"
    cited = set(_UUID_RE.findall(text))
    assert cited, f"the output cites no ids at all:\n{text}"
    assert cited <= real_ids, f"ids that are not in the bank: {cited - real_ids}\n{text}"


async def test_answer_and_mental_model_cite_real_ids(memory_real_llm: MemoryEngine, request_context):
    memory = memory_real_llm
    bank_id = f"test-real-ids-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice works at Google as a software engineer.", "event_date": "2025-01-10"},
                {"content": "Alice left Google and joined Stripe in March 2025.", "event_date": "2025-03-15"},
                {"content": "Alice works at Stripe on the payments team.", "event_date": "2025-06-01"},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()
        units = await memory.list_memory_units(bank_id, limit=500, request_context=request_context)
        real_ids = {str(u["id"]) for u in units["items"]}

        result = await memory.reflect_async(bank_id=bank_id, query=_QUERY, request_context=request_context)
        _assert_cites_real_ids(result.text, real_ids)

        # A mental model is the same reflect, stored: its content outlives the run.
        model = await memory.create_mental_model(
            bank_id=bank_id,
            name="Employer contradictions",
            source_query=_QUERY,
            content="pending",
            request_context=request_context,
        )
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=model["id"], request_context=request_context
        )
        assert refreshed is not None
        _assert_cites_real_ids(refreshed["content"], real_ids)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
