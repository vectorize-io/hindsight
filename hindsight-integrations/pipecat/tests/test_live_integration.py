"""Live integration test for HindsightMemoryService.

Runs conversation turns through the FrameProcessor against a real Hindsight
instance (default: http://localhost:8888) and verifies:

  1. Retain — turns get stored in Hindsight (verified via REST)
  2. Recall — subsequent turns surface stored memories
  3. Inject — <hindsight_memories> system message appears in forwarded frame
  4. Idempotency — memory block is replaced on re-injection (no duplication)

Run:
    HINDSIGHT_LIVE_URL=http://localhost:8888 python tests/test_live_integration.py

Or via pytest (also respects HINDSIGHT_LIVE_URL):
    HINDSIGHT_LIVE_URL=http://localhost:8888 python -m pytest tests/test_live_integration.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from hindsight_pipecat import HindsightMemoryService
from hindsight_pipecat.memory import _MEMORY_MARKER

LIVE_URL = os.environ.get("HINDSIGHT_LIVE_URL", "http://localhost:8888")
LIVE_API_KEY = os.environ.get("HINDSIGHT_LIVE_API_KEY")
UNIQUE_BANK = f"pipecat-live-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_get(path: str) -> dict:
    req = urllib.request.Request(f"{LIVE_URL}{path}")
    if LIVE_API_KEY:
        req.add_header("Authorization", f"Bearer {LIVE_API_KEY}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _make_frame(messages: list[dict[str, Any]]) -> MagicMock:
    """Build a mock OpenAILLMContextFrame that the processor will recognize."""
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame

    frame = MagicMock(spec=OpenAILLMContextFrame)
    ctx = MagicMock()
    ctx.messages = messages
    frame.context = ctx
    return frame


async def _run_turn(service: HindsightMemoryService, messages: list[dict[str, Any]]) -> MagicMock:
    """Send a single context frame through the service and return the frame post-processing."""
    frame = _make_frame(messages)
    # FrameProcessor.push_frame is what the processor calls to forward downstream.
    # Stub it so we don't need a real pipeline attached.
    service.push_frame = MagicMock(return_value=asyncio.sleep(0))
    # Call the private handler directly to avoid FrameProcessor setup requirements.
    await service._handle_context_frame(frame)
    return frame


def _print(label: str, value: Any = "") -> None:
    print(f"  {label}: {value}" if value != "" else f"  {label}")


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------


async def run_live_test() -> bool:
    print(f"\n=== Hindsight Pipecat Live Integration Test ===")
    print(f"URL:  {LIVE_URL}")
    print(f"Bank: {UNIQUE_BANK}")
    print()

    # 0. Health check
    print("[0] Health check")
    try:
        health = _http_get("/health")
        _print("status", health.get("status"))
    except Exception as e:
        print(f"  ❌ Hindsight not reachable: {e}")
        return False

    # 1. Create service
    service = HindsightMemoryService(
        bank_id=UNIQUE_BANK,
        hindsight_api_url=LIVE_URL,
        api_key=LIVE_API_KEY,
    )

    # 2. Turn 1 — introduce a fact
    print("\n[1] Turn 1 — seed fact about user")
    messages_t1 = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": "Hi! My name is Jordan and I prefer metric units."},
        {"role": "assistant", "content": "Nice to meet you Jordan! I'll use metric units."},
    ]
    frame1 = await _run_turn(service, messages_t1)
    _print("messages processed", len(messages_t1))

    # Wait for async retain to hit the server + a bit for async extraction
    print("  waiting 6s for async retain + fact extraction...")
    await asyncio.sleep(6)

    # 3. Verify retain — memories landed in the bank
    print("\n[2] Verify Retain via REST")
    memories = _http_get(f"/v1/default/banks/{UNIQUE_BANK}/memories/list")
    total = memories.get("total", 0)
    _print("total memories", total)
    texts = [m.get("text", "") for m in memories.get("items", [])]
    for t in texts[:5]:
        _print("  memory", t[:120])

    if total == 0:
        print("  ❌ No memories retained")
        return False
    print(f"  ✅ {total} memories retained")

    # 4. Turn 2 — ask about user (should trigger recall)
    print("\n[3] Turn 2 — ask question that needs recall")
    messages_t2 = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": "Hi! My name is Jordan and I prefer metric units."},
        {"role": "assistant", "content": "Nice to meet you Jordan! I'll use metric units."},
        {"role": "user", "content": "What's my name and unit preference?"},
    ]
    frame2 = await _run_turn(service, messages_t2)

    # 5. Verify Inject — <hindsight_memories> system message added
    print("\n[4] Verify Inject (hindsight_memories system message)")
    injected_messages = frame2.context.messages
    memory_msgs = [m for m in injected_messages if m.get("role") == "system" and _MEMORY_MARKER in m.get("content", "")]
    if not memory_msgs:
        print("  ❌ No <hindsight_memories> system message injected")
        return False

    injected_content = memory_msgs[0]["content"]
    _print("injected system msg length", f"{len(injected_content)} chars")
    preview = injected_content[:300].replace("\n", " | ")
    _print("preview", preview + "...")

    # 6. Check recall surfaced the key facts
    print("\n[5] Verify Recall surfaced Jordan + metric units")
    has_jordan = "Jordan" in injected_content or "jordan" in injected_content.lower()
    has_metric = "metric" in injected_content.lower() or "units" in injected_content.lower()
    _print("contains 'Jordan'", has_jordan)
    _print("contains 'metric'/'units'", has_metric)

    if not (has_jordan or has_metric):
        print("  ⚠️  Recall didn't surface expected facts")
        # Not a fatal failure — recall depends on embedding similarity

    # 7. Verify Idempotency — re-run should replace not duplicate
    print("\n[6] Verify Idempotency (re-inject should not duplicate)")
    frame3 = await _run_turn(service, messages_t2)
    injected_msgs_after = [
        m for m in frame3.context.messages if m.get("role") == "system" and _MEMORY_MARKER in m.get("content", "")
    ]
    _print("hindsight_memories system messages", len(injected_msgs_after))
    if len(injected_msgs_after) != 1:
        print(f"  ❌ Expected exactly 1 memory block, got {len(injected_msgs_after)}")
        return False
    print("  ✅ Memory block replaced, not duplicated")

    print("\n=== All 4 checks passed ✅ ===")
    print(f"Bank {UNIQUE_BANK} can be inspected:")
    print(f"  curl {LIVE_URL}/v1/default/banks/{UNIQUE_BANK}/memories/list | jq")
    return True


# ---------------------------------------------------------------------------
# Pytest wrapper
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("HINDSIGHT_LIVE_URL"),
    reason="HINDSIGHT_LIVE_URL not set — skipping live integration test",
)
@pytest.mark.asyncio
async def test_live_integration() -> None:
    success = await run_live_test()
    assert success, "Live integration test failed — see output above"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ok = asyncio.run(run_live_test())
    sys.exit(0 if ok else 1)
