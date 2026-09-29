"""Basic AgentCore Runtime handler with Hindsight persistent memory.

This example shows how to wrap an AgentCore Runtime invocation with the
``HindsightRuntimeAdapter`` so the agent recalls relevant memories before
each turn and retains its output afterwards. Memory persists across
Runtime session reprovisioning — your agent stops forgetting users.

Requirements:
    pip install hindsight-agentcore

Environment variables:
    HINDSIGHT_API_URL    - Hindsight API URL (default: https://api.hindsight.vectorize.io)
    HINDSIGHT_API_KEY    - Hindsight API key (required for Hindsight Cloud)

Run:
    python examples/basic_runtime_handler.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure

# 1. Configure Hindsight once at module import (or in your Runtime cold start).
configure(
    hindsight_api_url=os.environ.get("HINDSIGHT_API_URL", "https://api.hindsight.vectorize.io"),
    api_key=os.environ.get("HINDSIGHT_API_KEY"),
)

# 2. Create the adapter once and reuse it across invocations.
adapter = HindsightRuntimeAdapter(agent_name="support-agent")


async def my_agent(payload: dict[str, Any], memory_context: str) -> dict[str, Any]:
    """Stand-in for your real agent. Receives recalled memories as a string."""
    prompt = payload["prompt"]
    # Real code would inject memory_context into the system prompt for an LLM call.
    if memory_context:
        response = f"[recalled context applied]\n{prompt} → handled with prior context"
    else:
        response = f"{prompt} → handled (no prior context)"
    return {"output": response}


async def handler(event: dict[str, Any]) -> dict[str, Any]:
    """AgentCore Runtime entry point.

    Build a TurnContext from the AgentCore event (validated auth + session
    metadata) then delegate to adapter.run_turn(). The adapter:
      1. Recalls relevant memories for the user's prompt
      2. Calls your agent_callable with the recalled context
      3. Retains the user/assistant turn for future sessions
    """
    context = TurnContext(
        runtime_session_id=event["sessionId"],
        user_id=event["userId"],
        agent_name="support-agent",
        tenant_id=event.get("tenantId"),
    )
    return await adapter.run_turn(
        context=context,
        payload={"prompt": event["prompt"]},
        agent_callable=my_agent,
    )


async def main() -> None:
    """Local smoke test. Two simulated turns under the same user."""
    fake_event_1 = {
        "sessionId": "session-001",
        "userId": "user-alex",
        "prompt": "Hi, my name is Alex and I prefer email over phone.",
    }
    fake_event_2 = {
        "sessionId": "session-002",  # New Runtime session — bank persists
        "userId": "user-alex",
        "prompt": "What do you know about me?",
    }

    print("Turn 1:", await handler(fake_event_1))
    print("Turn 2 (new session):", await handler(fake_event_2))


if __name__ == "__main__":
    asyncio.run(main())
