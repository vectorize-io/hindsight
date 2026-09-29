"""
Live integration test for hindsight-agentcore.

Skipped unless ``HINDSIGHT_API_KEY`` is set. Targets the URL in
``HINDSIGHT_API_URL`` (default: http://localhost:8888).

Verifies the full retain → recall cycle through the adapter:
  1. Turn 1: agent receives a fact-bearing prompt; adapter retains the turn
  2. Wait for async fact extraction
  3. Turn 2 (new Runtime session, same user): adapter recalls and the
     memory_context surfaces the fact
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import pytest

from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure, reset_config

_api_key = os.environ.get("HINDSIGHT_API_KEY")
_api_url = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")

pytestmark = pytest.mark.skipif(
    not _api_key,
    reason="HINDSIGHT_API_KEY not set — skipping live integration test",
)


async def _echo_agent(payload: dict[str, Any], memory_context: str) -> dict[str, Any]:
    """Stub agent: echoes the prompt and surfaces any recalled memory."""
    return {
        "output": f"prompt={payload['prompt']!r} memory_present={bool(memory_context)}",
        "memory_context": memory_context,
    }


@pytest.mark.asyncio
async def test_run_turn_retain_then_recall() -> None:
    reset_config()
    configure(hindsight_api_url=_api_url, api_key=_api_key)

    user_id = f"agentcore-live-{uuid.uuid4().hex[:8]}"
    adapter = HindsightRuntimeAdapter(agent_name="agentcore-live-test")

    # Turn 1: plant a memorable fact
    fact = "My favorite programming language is Rust."
    turn_1 = TurnContext(
        runtime_session_id="session-1",
        user_id=user_id,
        agent_name="agentcore-live-test",
    )
    result_1 = await adapter.run_turn(
        context=turn_1,
        payload={"prompt": fact},
        agent_callable=_echo_agent,
    )
    assert "output" in result_1

    # Wait for async fact extraction to land
    await asyncio.sleep(8)

    # Turn 2: new Runtime session, same user — recall should surface the fact
    turn_2 = TurnContext(
        runtime_session_id="session-2",
        user_id=user_id,
        agent_name="agentcore-live-test",
    )
    result_2 = await adapter.run_turn(
        context=turn_2,
        payload={"prompt": "What is my favorite programming language?"},
        agent_callable=_echo_agent,
    )
    memory_context = result_2.get("memory_context", "")
    assert memory_context, "Expected recalled memory_context to be non-empty on turn 2"
    assert "Rust" in memory_context or "rust" in memory_context.lower(), (
        f"Expected recalled memory to mention Rust. Got: {memory_context!r}"
    )
