"""HindsightCallback's async hooks against a real in-process Hindsight stub.

The async hooks talk to Hindsight with aiohttp on the running loop, so the
stub is a real ``aiohttp.web`` server: requests, status handling and JSON
parsing all go through the actual transport.
"""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_litellm import cleanup, configure, reset_config, set_defaults
from hindsight_litellm.callbacks import HindsightCallback, HindsightError

BANK = "bank-a"
RECALL = ("POST", f"/v1/default/banks/{BANK}/memories/recall")
REFLECT = ("POST", f"/v1/default/banks/{BANK}/reflect")
RETAIN = ("POST", f"/v1/default/banks/{BANK}/memories")
DOCUMENT = ("GET", f"/v1/default/banks/{BANK}/documents/sess-1")


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: Any


@dataclass
class HindsightStub:
    base_url: str = ""
    requests: list[Recorded] = field(default_factory=list)
    routes: dict[tuple[str, str], tuple[int, Any]] = field(default_factory=dict)

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.path) for r in self.requests]


@pytest.fixture
async def hindsight():
    stub = HindsightStub()

    async def dispatch(request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        stub.requests.append(
            Recorded(request.method, request.path, dict(request.headers), json.loads(raw) if raw else None)
        )
        status, body = stub.routes.get((request.method, request.path), (404, {"detail": "not stubbed"}))
        return web.json_response(body, status=status)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", dispatch)
    server = TestServer(app)
    await server.start_server()
    stub.base_url = str(server.make_url("")).rstrip("/")
    reset_config()
    configure(
        hindsight_api_url=stub.base_url,
        api_key="tok",
        inject_memories=True,
        store_conversations=True,
    )
    set_defaults(bank_id=BANK)
    try:
        yield stub
    finally:
        cleanup()
        await server.close()


@pytest.fixture
async def callback():
    cb = HindsightCallback()
    yield cb
    await cb.aclose()


def _llm_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class TestAsyncPreCall:
    async def test_recall_injects_memories(self, hindsight, callback):
        hindsight.routes[RECALL] = (200, {"results": [{"text": "Alice likes cats", "type": "world"}]})
        messages = [{"role": "user", "content": "What does Alice like?"}]

        await callback.async_log_pre_api_call("gpt-4o-mini", messages, {})

        assert hindsight.calls() == [RECALL]
        request = hindsight.requests[0]
        assert request.body["query"] == "What does Alice like?"
        assert request.body["budget"] == "mid"
        assert request.headers["Authorization"] == "Bearer tok"
        assert "Alice likes cats" in json.dumps(messages)

    async def test_reflect_injects_answer(self, hindsight, callback):
        hindsight.routes[REFLECT] = (200, {"text": "Alice is a cat person"})
        messages = [{"role": "user", "content": "What does Alice like?"}]

        await callback.async_log_pre_api_call("gpt-4o-mini", messages, {"hindsight_use_reflect": True})

        assert hindsight.calls() == [REFLECT]
        assert "Alice is a cat person" in json.dumps(messages)

    async def test_recall_failure_raises(self, hindsight, callback):
        hindsight.routes[RECALL] = (500, {"detail": "boom"})
        messages = [{"role": "user", "content": "What does Alice like?"}]

        with pytest.raises(HindsightError, match="Memory recall failed"):
            await callback.async_log_pre_api_call("gpt-4o-mini", messages, {})
        assert messages == [{"role": "user", "content": "What does Alice like?"}]


class TestAsyncSuccessEvent:
    def _kwargs(self) -> dict:
        return {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "hindsight_session_id": "sess-1",
        }

    async def test_appends_to_existing_document(self, hindsight, callback):
        hindsight.routes[DOCUMENT] = (200, {"original_text": "USER: earlier"})
        hindsight.routes[RETAIN] = (200, {"success": True})

        await callback.async_log_success_event(self._kwargs(), _llm_response("Hi there"), 0.0, 1.0)

        assert hindsight.calls() == [DOCUMENT, RETAIN]
        item = hindsight.requests[1].body["items"][0]
        assert item["content"] == "USER: earlier\n\nUSER: Hello\n\nASSISTANT: Hi there"
        assert item["document_id"] == "sess-1"
        assert item["metadata"]["source"] == "litellm"
        assert hindsight.requests[1].headers["Authorization"] == "Bearer tok"

    async def test_missing_document_starts_a_new_one(self, hindsight, callback):
        # DOCUMENT is not stubbed: the stub answers 404.
        hindsight.routes[RETAIN] = (200, {"success": True})

        await callback.async_log_success_event(self._kwargs(), _llm_response("Hi there"), 0.0, 1.0)

        assert hindsight.calls() == [DOCUMENT, RETAIN]
        assert hindsight.requests[1].body["items"][0]["content"] == "USER: Hello\n\nASSISTANT: Hi there"

    async def test_store_failure_raises(self, hindsight, callback):
        hindsight.routes[RETAIN] = (503, {"detail": "unavailable"})

        with pytest.raises(HindsightError, match="Memory storage failed"):
            await callback.async_log_success_event(self._kwargs(), _llm_response("Hi there"), 0.0, 1.0)

    async def test_duplicate_is_stored_once(self, hindsight, callback):
        hindsight.routes[RETAIN] = (200, {"success": True})

        await callback.async_log_success_event(self._kwargs(), _llm_response("Hi there"), 0.0, 1.0)
        await callback.async_log_success_event(self._kwargs(), _llm_response("Hi there"), 0.0, 1.0)

        assert hindsight.calls().count(RETAIN) == 1


async def test_aclose_closes_the_loop_session(hindsight):
    cb = HindsightCallback()
    hindsight.routes[RECALL] = (200, {"results": []})
    await cb.async_log_pre_api_call("gpt-4o-mini", [{"role": "user", "content": "hi"}], {})
    session = cb._get_aiohttp_session()
    assert not session.closed

    await cb.aclose()

    assert session.closed
