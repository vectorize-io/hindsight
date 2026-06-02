"""End-to-end smoke test: ADK Runner + Hindsight against dev cloud.

Two phases, both using a real Gemini-backed Runner:

PHASE 1 — HindsightMemoryService (automatic memory):
1. Session A: tell the agent some facts. Runner saves them to Hindsight on
   session end via `add_session_to_memory`.
2. Session B: ask the agent to recall a fact from session A. The Runner
   calls `search_memory`, which the integration serves from Hindsight.

PHASE 2 — create_hindsight_tools (explicit tools):
1. Session C: agent calls `hindsight_retain` directly to store facts.
2. Session D: agent calls `hindsight_recall` directly to retrieve them.

If both phases find their facts after the session boundary, the integration
is fully verified end-to-end.

Required env:
    GOOGLE_API_KEY        — Gemini API key (aistudio.google.com/apikey)
    HINDSIGHT_API_URL     — defaults to https://api.dev.hindsight.vectorize.io
    HINDSIGHT_API_KEY     — Hindsight Cloud bearer token
"""

from __future__ import annotations

import asyncio
import os
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.load_memory_tool import load_memory_tool
from google.genai import types

from hindsight_google_adk import HindsightMemoryService, create_hindsight_tools

HINDSIGHT_API_URL = os.environ.get("HINDSIGHT_API_URL", "https://api.dev.hindsight.vectorize.io")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY")

APP_NAME = "hindsight-adk-smoke"
USER_ID = f"smoke-{uuid.uuid4().hex[:8]}"  # fresh user every run, isolates banks
TOOLS_BANK_ID = f"smoke-tools-{uuid.uuid4().hex[:8]}"


async def run_turn(runner: Runner, session_id: str, prompt: str) -> str:
    """Run one user turn and return the final assistant text."""
    user_msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=user_msg,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text
    return final_text


async def phase_memory_service() -> bool:
    """Phase 1: HindsightMemoryService end-to-end."""
    memory = HindsightMemoryService.from_url(
        hindsight_api_url=HINDSIGHT_API_URL,
        api_key=HINDSIGHT_API_KEY,
    )
    session_service = InMemorySessionService()
    agent = LlmAgent(
        name="hindsight_smoke_mem",
        model="gemini-2.0-flash",
        instruction=(
            "You are a helpful assistant with persistent memory. "
            "When the user asks about themselves or refers to past conversations, "
            "ALWAYS call the load_memory tool first to retrieve relevant context, "
            "then answer using what you find."
        ),
        tools=[load_memory_tool],
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
        memory_service=memory,
    )

    print("\n=== PHASE 1: HindsightMemoryService (automatic memory) ===")
    print(f"User:          {USER_ID}")
    print(f"Expected bank: {APP_NAME}::{USER_ID}\n")

    session_a_id = f"sess-a-{uuid.uuid4().hex[:6]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_a_id)
    print(f"--- Session A ({session_a_id}) ---")
    reply = await run_turn(
        runner,
        session_a_id,
        "Hi! My name is Ben and my favorite programming language is Rust. Also, I have a dog named Pixel.",
    )
    print(f"Agent: {reply[:200]}\n")

    # Manually flush session to memory (Runner calls this on session end in
    # real lifecycle; InMemorySessionService skips it).
    session_a = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_a_id)
    await memory.add_session_to_memory(session_a)
    print(f"  -> session A retained ({len(session_a.events)} events)\n")

    session_b_id = f"sess-b-{uuid.uuid4().hex[:6]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_b_id)
    print(f"--- Session B ({session_b_id}) ---")
    reply = await run_turn(
        runner,
        session_b_id,
        "What's my favorite programming language and my dog's name?",
    )
    print(f"Agent: {reply[:400]}\n")

    rust = "rust" in reply.lower()
    pixel = "pixel" in reply.lower()
    print(f"  Mentions Rust:  {'YES' if rust else 'NO'}")
    print(f"  Mentions Pixel: {'YES' if pixel else 'NO'}")
    passed = rust and pixel
    print(f"  Phase 1: {'PASS' if passed else 'FAIL'}")
    return passed


async def phase_tools() -> bool:
    """Phase 2: create_hindsight_tools end-to-end."""
    tools = create_hindsight_tools(
        bank_id=TOOLS_BANK_ID,
        hindsight_api_url=HINDSIGHT_API_URL,
        api_key=HINDSIGHT_API_KEY,
    )
    session_service = InMemorySessionService()
    agent = LlmAgent(
        name="hindsight_smoke_tools",
        model="gemini-2.0-flash",
        instruction=(
            "You are an assistant with three memory tools: "
            "hindsight_retain(content) stores facts; "
            "hindsight_recall(query) looks them up; "
            "hindsight_reflect(query) synthesizes answers. "
            "When the user shares a fact about themselves, ALWAYS call "
            "hindsight_retain to store it. When the user asks about "
            "themselves, ALWAYS call hindsight_recall first."
        ),
        tools=tools,
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    print("\n=== PHASE 2: create_hindsight_tools (explicit) ===")
    print(f"Bank: {TOOLS_BANK_ID}\n")

    session_c_id = f"sess-c-{uuid.uuid4().hex[:6]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_c_id)
    print(f"--- Session C ({session_c_id}) ---")
    reply = await run_turn(
        runner,
        session_c_id,
        "Please remember the following facts about me: I drive a Tesla Model 3 "
        "and my coffee order is an oat milk latte. Use your tools to save them.",
    )
    print(f"Agent: {reply[:300]}\n")

    # Give Hindsight a beat to commit before we recall.
    await asyncio.sleep(2)

    session_d_id = f"sess-d-{uuid.uuid4().hex[:6]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_d_id)
    print(f"--- Session D ({session_d_id}) ---")
    reply = await run_turn(
        runner,
        session_d_id,
        "What car do I drive and what's my coffee order? Use your recall tool.",
    )
    print(f"Agent: {reply[:400]}\n")

    tesla = "tesla" in reply.lower() or "model 3" in reply.lower()
    latte = "latte" in reply.lower() or "oat milk" in reply.lower()
    print(f"  Mentions Tesla/Model 3: {'YES' if tesla else 'NO'}")
    print(f"  Mentions latte/oat milk: {'YES' if latte else 'NO'}")
    passed = tesla and latte
    print(f"  Phase 2: {'PASS' if passed else 'FAIL'}")
    return passed


async def main() -> None:
    if not HINDSIGHT_API_KEY:
        raise SystemExit("HINDSIGHT_API_KEY env var is required.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY env var is required.")

    print(f"\nHindsight URL: {HINDSIGHT_API_URL}")

    p1 = await phase_memory_service()
    p2 = await phase_tools()

    print("\n=== Summary ===")
    print(f"  Phase 1 (HindsightMemoryService): {'PASS' if p1 else 'FAIL'}")
    print(f"  Phase 2 (create_hindsight_tools): {'PASS' if p2 else 'FAIL'}")
    print("\nBanks to inspect:")
    print(f"  Phase 1: {APP_NAME}::{USER_ID}")
    print(f"  Phase 2: {TOOLS_BANK_ID}")
    if not (p1 and p2):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
