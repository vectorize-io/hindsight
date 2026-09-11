"""A stub Ollama upstream for tests of the native ``/api/chat`` path.

Serves the given chat bodies in order (the last one repeats) from a real local
server via :func:`tests.aiohttp_stub.stub_server`, and records every request.

    async with ollama_stub(chat_body('{"ok": true}')) as stub:
        llm = OpenAICompatibleLLM(provider="ollama", base_url=stub.openai_base_url, ...)
        ...
    assert stub.requests[0].json["options"]["num_batch"] == 512
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from tests.aiohttp_stub import stub_server


@dataclass
class CapturedRequest:
    path: str
    json: dict[str, Any]
    headers: dict[str, str]


@dataclass
class OllamaStub:
    base_url: str
    requests: list[CapturedRequest] = field(default_factory=list)
    # Providers to clean up (closing their HTTP session) when the stub shuts down.
    clients: list[OpenAICompatibleLLM] = field(default_factory=list)

    @property
    def openai_base_url(self) -> str:
        """The ``/v1`` URL an Ollama provider is configured with."""
        return f"{self.base_url}/v1"


def chat_body(content: str, *, done_reason: str | None = None, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "test-model",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    if done_reason is not None:
        body["done_reason"] = done_reason
    body.update(extra)
    return body


@asynccontextmanager
async def ollama_stub(*bodies: dict[str, Any]) -> AsyncIterator[OllamaStub]:
    remaining = list(bodies)
    stub: OllamaStub | None = None

    async def handler(request: web.Request) -> web.StreamResponse:
        assert stub is not None
        stub.requests.append(CapturedRequest(request.path, await request.json(), dict(request.headers)))
        body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return web.json_response(body)

    async with stub_server(handler) as base_url:
        stub = OllamaStub(base_url=base_url)
        try:
            yield stub
        finally:
            for client in stub.clients:
                await client.cleanup()
