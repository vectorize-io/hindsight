"""Tests for HindsightMemoryService."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

from hindsight_google_adk import (
    HindsightError,
    HindsightMemoryService,
    reset_config,
)


@pytest.fixture(autouse=True)
def _reset_global_config():
    reset_config()
    yield
    reset_config()


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock(return_value=SimpleNamespace(results=[]))
    client.acreate_bank = AsyncMock()
    return client


def _event(author: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        author=author,
        content=types.Content(parts=[types.Part(text=text)]),
        id=f"evt-{author}-{abs(hash(text)) % 10000}",
        timestamp=0.0,
    )


def _session(app_name: str, user_id: str, events: list, session_id: str = "sess-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        events=events,
    )


class TestCreation:
    def test_from_client(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        assert svc._client is client
        assert svc._bank_id_template == "{app_name}::{user_id}"

    def test_init_requires_client(self):
        with pytest.raises(HindsightError):
            HindsightMemoryService()

    def test_bank_id_default_template(self):
        svc = HindsightMemoryService.from_client(_mock_client())
        assert svc._bank_id("apple", "alice") == "apple::alice"

    def test_bank_id_custom_template(self):
        svc = HindsightMemoryService.from_client(_mock_client(), bank_id_template="ns::{user_id}")
        assert svc._bank_id("anything", "bob") == "ns::bob"


class TestAddSessionToMemory:
    async def test_empty_session_does_not_retain(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_session_to_memory(_session("apple", "alice", []))
        client.aretain.assert_not_awaited()

    async def test_session_with_only_empty_events_does_not_retain(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        empty_event = SimpleNamespace(author="user", content=None, id="e", timestamp=0)
        await svc.add_session_to_memory(_session("apple", "alice", [empty_event]))
        client.aretain.assert_not_awaited()

    async def test_concatenates_events_in_order(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        events = [
            _event("user", "Hello"),
            _event("assistant", "Hi there"),
            _event("user", "What's up?"),
        ]
        await svc.add_session_to_memory(_session("apple", "alice", events))
        client.aretain.assert_awaited_once()
        kwargs = client.aretain.await_args.kwargs
        assert "user: Hello" in kwargs["content"]
        assert "assistant: Hi there" in kwargs["content"]
        assert kwargs["content"].index("Hello") < kwargs["content"].index("Hi there")

    async def test_tags_include_app_and_user(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client, tags=["env:prod"])
        await svc.add_session_to_memory(_session("apple", "alice", [_event("user", "hi")]))
        kwargs = client.aretain.await_args.kwargs
        assert "app:apple" in kwargs["tags"]
        assert "user:alice" in kwargs["tags"]
        assert "env:prod" in kwargs["tags"]

    async def test_document_id_is_session_id(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_session_to_memory(_session("apple", "alice", [_event("user", "hi")], session_id="sess-xyz"))
        assert client.aretain.await_args.kwargs["document_id"] == "sess-xyz"

    async def test_bank_id_derived_from_app_and_user(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_session_to_memory(_session("orange", "bob", [_event("user", "hi")]))
        assert client.aretain.await_args.kwargs["bank_id"] == "orange::bob"

    async def test_retain_failure_is_logged_not_raised(self, caplog):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("boom")
        svc = HindsightMemoryService.from_client(client)
        with caplog.at_level("ERROR"):
            await svc.add_session_to_memory(_session("apple", "alice", [_event("user", "hi")]))
        assert "retain failed" in caplog.text.lower()


class TestAddEventsToMemory:
    async def test_empty_events_no_op(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_events_to_memory(app_name="apple", user_id="alice", events=[])
        client.aretain.assert_not_awaited()

    async def test_single_retain_call(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_events_to_memory(
            app_name="apple",
            user_id="alice",
            events=[_event("user", "delta")],
            session_id="sess-1",
        )
        client.aretain.assert_awaited_once()

    async def test_session_id_in_tags_and_document_id(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_events_to_memory(
            app_name="apple",
            user_id="alice",
            events=[_event("user", "delta")],
            session_id="sess-z",
        )
        kwargs = client.aretain.await_args.kwargs
        assert "session:sess-z" in kwargs["tags"]
        assert kwargs["document_id"].startswith("sess-z-")

    async def test_no_session_id_yields_events_document_prefix(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_events_to_memory(app_name="apple", user_id="alice", events=[_event("user", "delta")])
        assert client.aretain.await_args.kwargs["document_id"].startswith("events-")

    async def test_custom_metadata_merged(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_events_to_memory(
            app_name="apple",
            user_id="alice",
            events=[_event("user", "delta")],
            custom_metadata={"trace": "abc"},
        )
        assert client.aretain.await_args.kwargs["metadata"]["trace"] == "abc"


class TestAddMemory:
    async def test_empty_memories_no_op(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_memory(app_name="apple", user_id="alice", memories=[])
        client.aretain.assert_not_awaited()

    async def test_each_memory_retained(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        m1 = MemoryEntry(content=types.Content(parts=[types.Part(text="one")]))
        m2 = MemoryEntry(content=types.Content(parts=[types.Part(text="two")]))
        await svc.add_memory(app_name="apple", user_id="alice", memories=[m1, m2])
        assert client.aretain.await_count == 2

    async def test_memory_author_in_metadata(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        m = MemoryEntry(
            content=types.Content(parts=[types.Part(text="hi")]),
            author="alice",
        )
        await svc.add_memory(app_name="apple", user_id="alice", memories=[m])
        assert client.aretain.await_args.kwargs["metadata"]["author"] == "alice"

    async def test_empty_content_skipped(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        m = MemoryEntry(content=types.Content(parts=[]))
        await svc.add_memory(app_name="apple", user_id="alice", memories=[m])
        client.aretain.assert_not_awaited()


class TestSearchMemory:
    async def test_bank_id_derived_from_app_and_user(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.search_memory(app_name="apple", user_id="alice", query="q")
        assert client.arecall.await_args.kwargs["bank_id"] == "apple::alice"

    async def test_user_tag_added_to_recall(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client, recall_tags=["env:prod"])
        await svc.search_memory(app_name="apple", user_id="alice", query="q")
        tags = client.arecall.await_args.kwargs["tags"]
        assert "user:alice" in tags
        assert "env:prod" in tags

    async def test_empty_results_returns_empty_response(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        resp = await svc.search_memory(app_name="apple", user_id="alice", query="q")
        assert resp.memories == []

    async def test_results_mapped_to_memory_entries(self):
        client = _mock_client()
        client.arecall.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(id="m1", text="first fact", occurred_start=None),
                SimpleNamespace(id="m2", text="second fact", occurred_start="2026-05-01T00:00:00Z"),
            ]
        )
        svc = HindsightMemoryService.from_client(client)
        resp = await svc.search_memory(app_name="apple", user_id="alice", query="q")
        assert len(resp.memories) == 2
        assert resp.memories[0].content.parts[0].text == "first fact"
        assert resp.memories[0].author == "hindsight"
        assert resp.memories[1].timestamp == "2026-05-01T00:00:00Z"

    async def test_recall_failure_returns_empty(self, caplog):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("boom")
        svc = HindsightMemoryService.from_client(client)
        with caplog.at_level("ERROR"):
            resp = await svc.search_memory(app_name="apple", user_id="alice", query="q")
        assert resp.memories == []
        assert "recall failed" in caplog.text.lower()

    async def test_budget_and_max_tokens_forwarded(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client, budget="high", max_tokens=1024)
        await svc.search_memory(app_name="apple", user_id="alice", query="q")
        kwargs = client.arecall.await_args.kwargs
        assert kwargs["budget"] == "high"
        assert kwargs["max_tokens"] == 1024


class TestBankMission:
    async def test_create_bank_called_once_when_mission_set(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client, mission="Track user preferences")
        await svc.add_session_to_memory(_session("apple", "alice", [_event("user", "hi")]))
        await svc.add_session_to_memory(_session("apple", "alice", [_event("user", "hi again")], session_id="sess-2"))
        assert client.acreate_bank.await_count == 1

    async def test_create_bank_not_called_without_mission(self):
        client = _mock_client()
        svc = HindsightMemoryService.from_client(client)
        await svc.add_session_to_memory(_session("apple", "alice", [_event("user", "hi")]))
        client.acreate_bank.assert_not_awaited()
