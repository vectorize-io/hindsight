"""Verify configured defaults reach the API through the real tools and SDK."""

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.models.recall_request import RecallRequest
from hindsight_client_api.models.reflect_request import ReflectRequest
from hindsight_smolagents import configure, create_hindsight_tools, reset_config


@dataclass(frozen=True)
class CapturedRequest:
    path: str
    payload_json: str


@pytest.mark.parametrize("explicit", [False, True])
def test_factory_configuration_reaches_api(explicit: bool) -> None:
    requests: list[CapturedRequest] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload_json = self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8")
            requests.append(CapturedRequest(path=self.path, payload_json=payload_json))
            body = b'{"results": []}' if self.path.endswith("/recall") else b'{"text": "reflection"}'
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
    try:
        configure(budget="high", max_tokens=1234, recall_tags=["team"], recall_tags_match="all")
        client = Hindsight(base_url=f"http://127.0.0.1:{server.server_port}", timeout=2.0)
        if explicit:
            created_tools = create_hindsight_tools(
                bank_id="test", client=client, budget="mid", max_tokens=4096, recall_tags_match="any"
            )
        else:
            created_tools = create_hindsight_tools(bank_id="test", client=client)
        tools = {tool.name: tool for tool in created_tools}
        assert tools["hindsight_recall"](query="preferences") == "No relevant memories found."
        assert tools["hindsight_reflect"](query="preferences") == "reflection"
        recall = RecallRequest.model_validate_json(requests[0].payload_json)
        reflect = ReflectRequest.model_validate_json(requests[1].payload_json)
        assert requests[0].path == "/v1/default/banks/test/memories/recall"
        assert requests[1].path == "/v1/default/banks/test/reflect"
        assert recall.query == reflect.query == "preferences"
        assert recall.budget == reflect.budget == ("mid" if explicit else "high")
        assert recall.max_tokens == (4096 if explicit else 1234)
        assert recall.tags == ["team"]
        assert recall.tags_match == ("any" if explicit else "all")
    finally:
        reset_config()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
