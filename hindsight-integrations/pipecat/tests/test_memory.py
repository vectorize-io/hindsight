"""Unit tests for HindsightMemoryService.

All Pipecat and Hindsight API calls are mocked — no live servers required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_pipecat import (
    HindsightMemoryService,
    HindsightPipecatError,
    configure,
    reset_config,
)
from hindsight_pipecat.memory import _MEMORY_MARKER, _resolve_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(recall_texts: list[str] | None = None) -> MagicMock:
    """Return a mock Hindsight client with async recall/retain methods."""
    client = MagicMock()
    response = MagicMock()
    if recall_texts:
        results = []
        for t in recall_texts:
            r = MagicMock()
            r.text = t
            results.append(r)
        response.results = results
    else:
        response.results = []
    client.arecall = AsyncMock(return_value=response)
    client.aretain = AsyncMock(return_value=None)
    return client


def _make_messages(*pairs: tuple[str, str]) -> list[dict]:
    """Build a flat messages list from (user, assistant) string pairs."""
    msgs: list[dict] = []
    for user_text, assistant_text in pairs:
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": assistant_text})
    return msgs


def _make_context(messages: list[dict]) -> MagicMock:
    """Return a mock LLMContext carrying the given messages list."""
    ctx = MagicMock()
    ctx.messages = messages
    return ctx


def _make_frame(messages: list[dict]) -> MagicMock:
    """Return a mock OpenAILLMContextFrame."""
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame  # noqa: F401

    frame = MagicMock(spec=OpenAILLMContextFrame)
    frame.context = _make_context(messages)
    return frame


# ---------------------------------------------------------------------------
# _resolve_client
# ---------------------------------------------------------------------------


class TestResolveClient:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_returns_explicit_client(self) -> None:
        client = _mock_client()
        assert _resolve_client(client, None, None) is client

    def test_creates_client_from_url(self) -> None:
        with patch("hindsight_pipecat.memory.Hindsight") as MockHindsight:
            _resolve_client(None, "http://localhost:8888", None)
            MockHindsight.assert_called_once()

    def test_raises_when_no_url(self) -> None:
        with pytest.raises(HindsightPipecatError, match="No Hindsight API URL"):
            _resolve_client(None, None, None)

    def test_uses_global_config_url(self) -> None:
        configure(hindsight_api_url="http://config-host:8888")
        with patch("hindsight_pipecat.memory.Hindsight") as MockHindsight:
            _resolve_client(None, None, None)
            MockHindsight.assert_called_once()


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    async def test_memories_injected_as_system_message(self) -> None:
        client = _mock_client(recall_texts=["User prefers dark mode"])
        messages = [{"role": "user", "content": "Set up my editor"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)

        sys_msgs = [m for m in frame.context.messages if m.get("role") == "system"]
        assert any(_MEMORY_MARKER in m["content"] for m in sys_msgs)
        assert any("dark mode" in m["content"] for m in sys_msgs)

    async def test_empty_recall_skips_injection(self) -> None:
        client = _mock_client(recall_texts=[])
        messages = [{"role": "user", "content": "Hello"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)

        sys_msgs = [m for m in frame.context.messages if m.get("role") == "system"]
        assert not any(_MEMORY_MARKER in m.get("content", "") for m in sys_msgs)

    async def test_recall_error_swallowed_frame_forwarded(self) -> None:
        client = _mock_client()
        client.arecall = AsyncMock(side_effect=RuntimeError("network error"))
        messages = [{"role": "user", "content": "Hello"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client)
        pushed: list = []
        with patch.object(
            svc,
            "push_frame",
            new_callable=AsyncMock,
            side_effect=lambda f, d: pushed.append(f),
        ):
            await svc._handle_context_frame(frame)

        assert len(pushed) == 1  # Frame was forwarded despite recall error

    async def test_no_recall_when_disabled(self) -> None:
        client = _mock_client(recall_texts=["some memory"])
        messages = [{"role": "user", "content": "Hello"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client, enable_recall=False)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)

        client.arecall.assert_not_called()

    async def test_recall_uses_last_user_message_as_query(self) -> None:
        client = _mock_client(recall_texts=["fact"])
        messages = _make_messages(("first turn", "ok")) + [{"role": "user", "content": "current query"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)

        call_args = client.arecall.call_args
        assert call_args.kwargs["query"] == "current query"


# ---------------------------------------------------------------------------
# Retain
# ---------------------------------------------------------------------------


class TestRetain:
    async def test_complete_pair_retained(self) -> None:
        client = _mock_client()
        messages = _make_messages(("Hello", "Hi there")) + [{"role": "user", "content": "Next question"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client, enable_recall=False)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)
            await asyncio.sleep(0)  # Let fire-and-forget tasks run

        client.aretain.assert_called_once()
        retained_content = client.aretain.call_args.kwargs["content"]
        assert "Hello" in retained_content
        assert "Hi there" in retained_content

    async def test_already_retained_pairs_not_re_retained(self) -> None:
        client = _mock_client()
        messages = _make_messages(("Turn 1 user", "Turn 1 assistant")) + [{"role": "user", "content": "Turn 2"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client, enable_recall=False)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)
            await asyncio.sleep(0)

        first_call_count = client.aretain.call_count

        # Second frame — same pairs already retained
        frame2 = _make_frame(messages)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame2)
            await asyncio.sleep(0)

        assert client.aretain.call_count == first_call_count  # No new calls

    async def test_retain_error_swallowed(self) -> None:
        client = _mock_client()
        client.aretain = AsyncMock(side_effect=RuntimeError("write error"))
        messages = _make_messages(("Hello", "Hi")) + [{"role": "user", "content": "Next"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client, enable_recall=False)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            # Should not raise
            await svc._handle_context_frame(frame)
            await asyncio.sleep(0)

    async def test_no_retain_when_disabled(self) -> None:
        client = _mock_client()
        messages = _make_messages(("Hello", "Hi")) + [{"role": "user", "content": "Next"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client, enable_retain=False, enable_recall=False)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)
            await asyncio.sleep(0)

        client.aretain.assert_not_called()

    async def test_incomplete_turn_not_retained(self) -> None:
        """A lone user message (no following assistant message) is not retained."""
        client = _mock_client()
        messages = [{"role": "user", "content": "Hello"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client, enable_recall=False)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)
            await asyncio.sleep(0)

        client.aretain.assert_not_called()


# ---------------------------------------------------------------------------
# Memory injection
# ---------------------------------------------------------------------------


class TestMemoryInjection:
    async def test_existing_memory_message_replaced(self) -> None:
        client = _mock_client(recall_texts=["new memory"])
        existing_memory_msg = {
            "role": "system",
            "content": f"{_MEMORY_MARKER}\nold memory\n</hindsight_memories>",
        }
        messages = [existing_memory_msg, {"role": "user", "content": "Hello"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)

        memory_msgs = [m for m in frame.context.messages if _MEMORY_MARKER in m.get("content", "")]
        assert len(memory_msgs) == 1, "Memory message should not be duplicated"
        assert "new memory" in memory_msgs[0]["content"]
        assert "old memory" not in memory_msgs[0]["content"]

    async def test_memory_message_prepended(self) -> None:
        client = _mock_client(recall_texts=["some fact"])
        messages = [{"role": "user", "content": "Hello"}]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)

        assert frame.context.messages[0]["role"] == "system"
        assert _MEMORY_MARKER in frame.context.messages[0]["content"]

    async def test_multimodal_user_message_text_extracted(self) -> None:
        client = _mock_client(recall_texts=["fact"])
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "voice transcription here"}],
            }
        ]
        frame = _make_frame(messages)

        svc = HindsightMemoryService(bank_id="test", client=client)
        with patch.object(svc, "push_frame", new_callable=AsyncMock):
            await svc._handle_context_frame(frame)

        call_args = client.arecall.call_args
        assert call_args.kwargs["query"] == "voice transcription here"
