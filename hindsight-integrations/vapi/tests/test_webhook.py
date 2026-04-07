"""Unit tests for HindsightVapiWebhook."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_vapi import HindsightVapiWebhook, configure, reset_config
from hindsight_vapi.errors import HindsightVapiError
from hindsight_vapi.webhook import _resolve_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(results: list[str] | None = None) -> MagicMock:
    """Return a mock Hindsight client with preset recall results."""
    client = MagicMock()
    if results:
        mock_results = [MagicMock(text=r) for r in results]
        client.arecall = AsyncMock(return_value=MagicMock(results=mock_results))
    else:
        client.arecall = AsyncMock(return_value=MagicMock(results=[]))
    client.aretain = AsyncMock(return_value=None)
    return client


def _make_assistant_request(caller_number: str | None = "+15555550100") -> dict[str, Any]:
    call: dict[str, Any] = {}
    if caller_number is not None:
        call["customer"] = {"number": caller_number}
    return {"message": {"type": "assistant-request", "call": call}}


def _make_end_of_call(transcript: str = "User: hi\nAssistant: hello") -> dict[str, Any]:
    return {
        "message": {
            "type": "end-of-call-report",
            "artifact": {"transcript": transcript},
        }
    }


# ---------------------------------------------------------------------------
# TestResolveClient
# ---------------------------------------------------------------------------


class TestResolveClient:
    def setup_method(self) -> None:
        reset_config()

    def test_returns_explicit_client(self) -> None:
        client = MagicMock()
        result = _resolve_client(client, None, None)
        assert result is client

    def test_creates_client_from_url(self) -> None:
        with patch("hindsight_vapi.webhook.Hindsight") as mock_cls:
            _resolve_client(None, "http://localhost:8888", None)
            mock_cls.assert_called_once()

    def test_raises_when_no_url(self) -> None:
        with pytest.raises(HindsightVapiError, match="No Hindsight API URL"):
            _resolve_client(None, None, None)

    def test_uses_global_config_url(self) -> None:
        configure(hindsight_api_url="http://configured:8888")
        with patch("hindsight_vapi.webhook.Hindsight") as mock_cls:
            _resolve_client(None, None, None)
            mock_cls.assert_called_once()
        reset_config()


# ---------------------------------------------------------------------------
# TestAssistantRequest
# ---------------------------------------------------------------------------


class TestAssistantRequest:
    def _svc(self, results: list[str] | None = None, **kwargs: Any) -> HindsightVapiWebhook:
        client = _make_client(results)
        return HindsightVapiWebhook(bank_id="test-bank", client=client, **kwargs)

    async def test_memories_injected_into_assistant_overrides(self) -> None:
        svc = self._svc(["Caller is Jordan", "Prefers metric units"])
        event = _make_assistant_request()
        result = await svc.handle(event)
        assert result is not None
        assert "assistantOverrides" in result
        messages = result["assistantOverrides"]["model"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "<hindsight_memories>" in messages[0]["content"]
        assert "Jordan" in messages[0]["content"]

    async def test_empty_recall_returns_empty_dict(self) -> None:
        svc = self._svc(results=None)
        event = _make_assistant_request()
        result = await svc.handle(event)
        assert result == {}

    async def test_recall_error_swallowed_empty_dict_returned(self) -> None:
        client = _make_client()
        client.arecall = AsyncMock(side_effect=RuntimeError("network error"))
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        event = _make_assistant_request()
        result = await svc.handle(event)
        assert result == {}

    async def test_caller_number_used_as_recall_query(self) -> None:
        client = _make_client(["fact"])
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        event = _make_assistant_request(caller_number="+15555550199")
        await svc.handle(event)
        client.arecall.assert_called_once()
        call_kwargs = client.arecall.call_args.kwargs
        assert call_kwargs["query"] == "+15555550199"

    async def test_no_caller_number_uses_fallback_query(self) -> None:
        client = _make_client()
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        event = _make_assistant_request(caller_number=None)
        await svc.handle(event)
        call_kwargs = client.arecall.call_args.kwargs
        assert call_kwargs["query"] == "returning caller"

    async def test_enable_recall_false_skips_recall(self) -> None:
        client = _make_client(["fact"])
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client, enable_recall=False)
        event = _make_assistant_request()
        result = await svc.handle(event)
        client.arecall.assert_not_called()
        assert result == {}


# ---------------------------------------------------------------------------
# TestEndOfCall
# ---------------------------------------------------------------------------


class TestEndOfCall:
    def _svc(self, **kwargs: Any) -> HindsightVapiWebhook:
        client = _make_client()
        return HindsightVapiWebhook(bank_id="test-bank", client=client, **kwargs)

    async def test_transcript_retained_fire_and_forget(self) -> None:
        svc = self._svc()
        event = _make_end_of_call("User: hello\nAssistant: hi there")
        with patch("hindsight_vapi.webhook.asyncio.create_task") as mock_task:
            result = await svc.handle(event)
        assert result is None
        mock_task.assert_called_once()

    async def test_retain_called_with_transcript(self) -> None:
        client = _make_client()
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        transcript = "User: hello\nAssistant: hi there"

        retained: list[str] = []

        async def _fake_retain(content: str) -> None:
            retained.append(content)

        ensure = asyncio.ensure_future
        with patch.object(svc, "_retain", side_effect=_fake_retain):
            with patch("hindsight_vapi.webhook.asyncio.create_task", side_effect=ensure):
                await svc.handle(_make_end_of_call(transcript))
                await asyncio.sleep(0)  # allow the task to run

        assert retained == [transcript]

    async def test_retain_error_swallowed(self) -> None:
        client = _make_client()
        client.aretain = AsyncMock(side_effect=RuntimeError("retain failed"))
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        # Should not raise
        await svc._retain("some transcript")

    async def test_empty_transcript_is_noop(self) -> None:
        svc = self._svc()
        event = _make_end_of_call(transcript="")
        with patch("hindsight_vapi.webhook.asyncio.create_task") as mock_task:
            await svc.handle(event)
        mock_task.assert_not_called()

    async def test_enable_retain_false_skips_retain(self) -> None:
        svc = self._svc(enable_retain=False)
        with patch("hindsight_vapi.webhook.asyncio.create_task") as mock_task:
            await svc.handle(_make_end_of_call())
        mock_task.assert_not_called()


# ---------------------------------------------------------------------------
# TestBuildOverrides
# ---------------------------------------------------------------------------


class TestBuildOverrides:
    async def test_returns_overrides_with_memories(self) -> None:
        client = _make_client(["User is Jordan", "Prefers metric"])
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        result = await svc.build_assistant_overrides("what do I know about this caller?")
        assert "assistantOverrides" in result
        messages = result["assistantOverrides"]["model"]["messages"]
        assert messages[0]["role"] == "system"
        assert "Jordan" in messages[0]["content"]

    async def test_empty_recall_returns_empty_dict(self) -> None:
        client = _make_client(results=None)
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        result = await svc.build_assistant_overrides("query")
        assert result == {}

    async def test_enable_recall_false_returns_empty_dict(self) -> None:
        client = _make_client(["fact"])
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client, enable_recall=False)
        result = await svc.build_assistant_overrides("query")
        client.arecall.assert_not_called()
        assert result == {}


# ---------------------------------------------------------------------------
# TestUnknownEvent
# ---------------------------------------------------------------------------


class TestUnknownEvent:
    async def test_unknown_event_type_returns_none(self) -> None:
        client = _make_client()
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        for event_type in ["call-started", "call-ended", "speech-update", "transcript"]:
            result = await svc.handle({"message": {"type": event_type}})
            assert result is None, f"Expected None for event type {event_type!r}"
        client.arecall.assert_not_called()
        client.aretain.assert_not_called()

    async def test_empty_event_returns_none(self) -> None:
        client = _make_client()
        svc = HindsightVapiWebhook(bank_id="test-bank", client=client)
        result = await svc.handle({})
        assert result is None
