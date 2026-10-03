"""Reflection mode must consume the SDK response field, including in HTTP calls."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import AsyncMock, MagicMock

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.models.reflect_request import ReflectRequest
from hindsight_client_api.models.reflect_response import ReflectResponse

from hindsight_agentcore import HindsightRuntimeAdapter, RecallPolicy, TurnContext, reset_config


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Synthesized context", "Prefers café ☕", ""])
async def test_before_turn_accepts_sdk_reflect_response(text: str) -> None:
    reset_config()
    try:
        client = MagicMock()
        client.areflect = AsyncMock(return_value=ReflectResponse(text=text))
        client.arecall = AsyncMock()
        adapter = HindsightRuntimeAdapter(recall_policy=RecallPolicy(mode="reflect", budget="high", max_tokens=1234))
        adapter._client = client
        context = TurnContext(runtime_session_id="session", user_id="user", agent_name="agent")
        assert await adapter.before_turn(context, query="preferences") == text
        client.areflect.assert_awaited_once_with(
            bank_id="user:user:agent:agent", query="preferences", budget="high", max_tokens=1234
        )
        client.arecall.assert_not_called()
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_before_turn_reflects_through_actual_sdk() -> None:
    requests: list[ReflectRequest] = []
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(ReflectRequest.model_validate_json(self.rfile.read(int(self.headers["Content-Length"]))))
            paths.append(self.path)
            body = b'{"text": "Synthesized project context"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reset_config()
    client = Hindsight(base_url=f"http://127.0.0.1:{server.server_port}", timeout=2.0)
    try:
        adapter = HindsightRuntimeAdapter(
            hindsight_api_url=f"http://127.0.0.1:{server.server_port}",
            recall_policy=RecallPolicy(mode="reflect", budget="high", max_tokens=1234),
        )
        adapter._client = client
        context = TurnContext(runtime_session_id="session", user_id="user", agent_name="agent")
        assert await adapter.before_turn(context, query="preferences") == "Synthesized project context"
        assert paths == ["/v1/default/banks/user:user:agent:agent/reflect"]
        assert len(requests) == 1
        assert requests[0].query == "preferences"
        assert requests[0].budget == "high"
        assert requests[0].max_tokens == 1234
    finally:
        await client.aclose()
        reset_config()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
