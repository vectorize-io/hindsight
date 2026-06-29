"""LLM-behaviour test for the folder curator's decisions.

The curator's *mechanics* (applying ops) are covered deterministically in
test_knowledge_curator.py. This test exercises how the model interprets a folder
mission + memories and decides which pages to create — non-deterministic, so it
runs the real pipeline and judges the outcome rather than string-matching.
"""

import uuid

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


async def test_curator_creates_page_for_mission_topic(memory_real_llm: MemoryEngine, request_context):
    memory = memory_real_llm
    bank_id = f"test-kc-llm-{uuid.uuid4().hex[:8]}"

    # Seed memories the curator should turn into a page.
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {"content": "I adopted a golden retriever named Rex in March."},
            {"content": "Rex loves swimming and is great with kids."},
            {"content": "My favorite programming language is Rust."},
        ],
        request_context=request_context,
    )
    await memory.wait_for_background_tasks()

    folder = await memory.create_knowledge_folder(
        bank_id, "Pets", mission="Everything about the user's pets", request_context=request_context
    )

    result = await memory.curate_folder(bank_id, folder["id"], request_context=request_context)

    assert result.applied, "curator should have created at least one page for a clear mission topic"

    summary = "\n".join(f"- {a.action}: {a.target} ({a.detail})" for a in result.applied)
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "At least one created page is about the user's pet/dog Rex (the folder's mission). "
            "A page about the Rust programming language would be off-mission and should NOT appear."
        ),
        context="Folder mission: 'Everything about the user's pets'. Memories included a dog named Rex and an unrelated note about Rust.",
    )

    await memory.delete_bank(bank_id, request_context=request_context)
