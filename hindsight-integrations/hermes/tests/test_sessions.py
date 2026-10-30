"""Session mode: the conversation streams into a Hindsight session, notes come back before a turn.

A real HTTP server on a thread stands in for Hindsight's sessions endpoint, so the plugin's own
requests (body, ids, acknowledgement) are what is asserted.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _Sessions:
    def __init__(self, status: int = 200) -> None:
        self.bodies: list[dict] = []
        self.notes: list[dict] = []
        self.status = status


@pytest.fixture
def server():
    state = _Sessions()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.bodies.append({"path": self.path, **body})
            if state.status != 200:
                self.send_response(state.status)
                self.end_headers()
                return
            out = {"notes": [n for n in state.notes if n["seq"] > body.get("ack", 0)], "checking": False}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_port}"
    yield state
    httpd.shutdown()


def _session_provider(provider, server):
    return provider({"live_session": True, "api_url": server.url, "bank_id": "team"})


def test_a_turn_streams_as_openai_messages_and_is_not_retained_twice(provider, server):
    instance, fake = _session_provider(provider, server)
    instance.on_turn_start(1, "Fix the rounding")
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Fix the rounding (with recalled context)"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "pay.py"},
        {"role": "assistant", "content": "Done."},
    ]
    instance.sync_turn("Fix the rounding", "Done.", messages=messages)
    instance.shutdown()

    start, turn = server.bodies
    assert start["path"] == "/v1/default/banks/team/sessions/session-1/events"
    assert start["session"]["metadata"]["harness"] == "hermes" and start["session"]["live"] is True
    assert start["messages"] == [{"id": "t1:0", "role": "user", "content": "Fix the rounding"}]
    # The turn from its user message on; the user message keeps the id the turn start sent.
    assert [m["id"] for m in turn["messages"]] == ["t1:0", "t1:1", "t1:2", "t1:3"]
    assert turn["messages"][1]["tool_calls"][0]["function"]["name"] == "terminal"
    assert turn["turn_end"] is True and "session" not in turn
    assert fake.retains == []  # the session saves the turn


def test_notes_are_injected_before_the_next_turn_and_acknowledged_once(provider, server):
    instance, _ = _session_provider(provider, server)
    instance.on_turn_start(1, "Fix the rounding")
    server.notes = [
        {"seq": 7, "text": "Half cents round down.", "trigger": "task", "source": {"document_id": "d", "quote": "q"}}
    ]
    block = instance.prefetch("Fix the rounding")
    assert "Half cents round down." in block and 'Source (d): "q"' in block
    assert "From Hindsight memory (live session #<note number>)" in block  # the agent is asked to credit it
    assert "Hindsight note (live session #<note number>)" in block  # and to mention it even when not needed
    assert "Note #" in block  # each note carries the number to name it by
    # Delivered: the next request acknowledges it, and it is not injected again.
    assert instance.prefetch("next") == ""
    assert server.bodies[-1]["ack"] == 7


def test_a_server_without_sessions_falls_back_to_the_per_turn_retain(provider, server):
    server.status = 404
    instance, fake = _session_provider(provider, server)
    instance.on_turn_start(1, "hello")
    instance.sync_turn("hello", "hi")
    instance.shutdown()
    assert len(fake.retains) == 1


def test_tool_calls_stream_mid_turn_and_notes_ride_on_the_result_without_waiting(provider, server):
    instance, _ = _session_provider(provider, server)
    instance.on_turn_start(1, "Fix the rounding")
    assert instance.on_tool_result("terminal", {"command": "ls"}, "pay.py", tool_call_id="c1") is None
    live = server.bodies[-1]["messages"]
    assert [m["id"] for m in live] == ["t1:call:c1", "t1:result:c1"]
    assert live[0]["tool_calls"][0]["function"]["name"] == "terminal"

    server.notes = [{"seq": 3, "text": "Half cents round down.", "source": {}}]
    out = instance.on_tool_result("patch", {"path": "pay.py"}, "ok", tool_call_id="c2")
    assert out.startswith("ok") and "Half cents round down." in out

    messages = [
        {"role": "user", "content": "Fix the rounding"},
        {
            "role": "assistant",
            "content": "Looking.",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "pay.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "patch", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "assistant", "content": "Done."},
    ]
    instance.sync_turn("Fix the rounding", "Done.", messages=messages)
    instance.shutdown()
    turn = server.bodies[-1]
    # The calls sent live are not sent again; the text around them is.
    assert [m["id"] for m in turn["messages"]] == ["t1:0", "t1:1", "t1:5"]
    assert "tool_calls" not in turn["messages"][1] and turn["ack"] == 3
