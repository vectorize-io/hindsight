"""End-to-end: drive the provider against a live Hindsight server.

Exercises the real Agent Framework hooks + the real Hindsight client (no LLM
agent needed). Gated behind HINDSIGHT_API_URL and marked requires_real_llm, so
it is excluded from the deterministic PR-CI bucket.
"""

import asyncio
import os
import uuid

import pytest
from conftest import session_context

from hindsight_agent_framework import HindsightProvider

pytestmark = pytest.mark.requires_real_llm

API_URL = os.environ.get("HINDSIGHT_API_URL")


@pytest.mark.asyncio
async def test_retain_then_recall_roundtrip():
    if not API_URL:
        pytest.skip("HINDSIGHT_API_URL not set")

    bank_id = f"af-e2e-{uuid.uuid4().hex[:8]}"
    provider = HindsightProvider(
        bank_id=bank_id,
        hindsight_api_url=API_URL,
        api_key=os.environ.get("HINDSIGHT_API_KEY"),
    )

    # Turn 1 — retain a clear, recallable fact.
    ctx1 = session_context(
        input_texts=("Remember that my favorite programming language is Haskell.",),
        response_text="Got it — I'll remember you prefer Haskell.",
    )
    await provider.after_run(agent=None, session=None, context=ctx1, state={})

    # Turn 2 — recall should surface it (allow time for server-side extraction).
    found = False
    for _ in range(20):
        ctx2 = session_context(input_texts=("What programming language do I like?",))
        await provider.before_run(agent=None, session=None, context=ctx2, state={})
        if any("haskell" in instr.lower() for instr in ctx2.instructions):
            found = True
            break
        await asyncio.sleep(3)

    assert found, "retained memory was not recalled within the timeout"
