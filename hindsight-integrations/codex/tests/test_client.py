"""Tests for lib/client.py — Hindsight REST API client."""

import json
from unittest.mock import patch

from conftest import FakeHTTPResponse
from lib.client import USER_AGENT, HindsightClient


class TestUserAgentHeader:
    """Regression tests for #1041.

    The stdlib default ``Python-urllib/X.Y`` UA is blocked by Cloudflare with
    error 1010, so every request must carry our identifying UA.
    """

    def test_recall_sends_user_agent(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.recall("bank", "query")

        assert captured["ua"] == USER_AGENT
        assert captured["ua"].startswith("hindsight-codex/")

    def test_health_check_sends_user_agent(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({}, status=200)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.health_check(timeout=1)

        assert captured["ua"] == USER_AGENT


def test_retain_sends_idempotency_key():
    client = HindsightClient("http://localhost:9077")
    captured = {}

    def fake_open(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return FakeHTTPResponse({"operation_id": "op-1"})

    with patch("urllib.request.urlopen", side_effect=fake_open):
        client.retain("bank", "content", idempotency_key="stable-key")

    assert captured["body"]["idempotency_key"] == "stable-key"


def test_retain_preserves_positional_timeout():
    client = HindsightClient("http://localhost:9077")
    captured = {}

    def fake_open(req, timeout=None):
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode())
        return FakeHTTPResponse({"operation_id": "op-1"})

    with patch("urllib.request.urlopen", side_effect=fake_open):
        client.retain("bank", "content", "doc", "codex", {}, [], 37)

    assert captured["timeout"] == 37
    assert "idempotency_key" not in captured["body"]


def test_auth_context_identity_changes_without_exposing_token():
    first = HindsightClient("http://localhost:9077", "secret-one")
    same = HindsightClient("http://localhost:9077", "secret-one")
    second = HindsightClient("http://localhost:9077", "secret-two")
    anonymous = HindsightClient("http://localhost:9077")

    assert first.auth_context_id == same.auth_context_id
    assert first.auth_context_id != second.auth_context_id
    assert first.auth_context_id != anonymous.auth_context_id
    assert "secret-one" not in first.auth_context_id


def test_durable_retain_delivery_requires_both_capabilities():
    client = HindsightClient("http://localhost:9077")

    with patch(
        "urllib.request.urlopen",
        return_value=FakeHTTPResponse(
            {
                "api_version": "0.8.5",
                "features": {
                    "retain_idempotency": True,
                    "retain_serialized_upsert": True,
                },
            }
        ),
    ):
        assert client.supports_durable_retain_delivery() is True

    incomplete_feature_sets = (
        {},
        {"retain_idempotency": True},
        {"retain_serialized_upsert": True},
    )
    for features in incomplete_feature_sets:
        with patch(
            "urllib.request.urlopen",
            return_value=FakeHTTPResponse({"api_version": "0.8.5", "features": features}),
        ):
            assert client.supports_durable_retain_delivery() is False
