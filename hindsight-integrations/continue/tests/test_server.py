"""HTTP roundtrip tests: drive the adapter exactly as Continue's http provider does."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.client import HTTPConnection

from hindsight_continue import build_server, configure

from .conftest import continue_request, make_client


@contextmanager
def running_server(client):
    """Start the adapter on an ephemeral port; yield its base address."""
    config = configure(bank_id="proj-1", host="127.0.0.1", port=0)
    server = build_server(config=config, client=client)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(host, port, body: dict):
    conn = HTTPConnection(host, port, timeout=5)
    try:
        conn.request("POST", "/", body=json.dumps(body), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _get(host, port, path: str):
    conn = HTTPConnection(host, port, timeout=5)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


class TestServerRoundtrip:
    def test_post_returns_context_items(self):
        client = make_client(["User prefers tabs over spaces"])
        with running_server(client) as (host, port):
            status, raw = _post(host, port, continue_request(query="formatting prefs"))

        assert status == 200
        items = json.loads(raw)
        assert isinstance(items, list) and len(items) == 1
        assert set(items[0].keys()) == {"name", "description", "content"}
        assert "User prefers tabs over spaces" in items[0]["content"]
        # The integration actually called Hindsight with the typed query.
        assert client.recall.call_args.kwargs["query"] == "formatting prefs"

    def test_no_results_returns_empty_array(self):
        client = make_client([])
        with running_server(client) as (host, port):
            status, raw = _post(host, port, continue_request(query="nothing here"))

        assert status == 200
        assert json.loads(raw) == []

    def test_health_endpoint(self):
        client = make_client()
        with running_server(client) as (host, port):
            status, raw = _get(host, port, "/health")

        assert status == 200
        assert json.loads(raw)["status"] == "ok"

    def test_invalid_json_returns_400(self):
        client = make_client()
        with running_server(client) as (host, port):
            conn = HTTPConnection(host, port, timeout=5)
            conn.request("POST", "/", body="not json{", headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            status = resp.status
            conn.close()

        assert status == 400

    def test_recall_error_surfaces_as_502(self):
        client = make_client()
        client.recall.side_effect = RuntimeError("upstream down")
        with running_server(client) as (host, port):
            status, raw = _post(host, port, continue_request(query="hi"))

        assert status == 502
        assert "error" in json.loads(raw)
