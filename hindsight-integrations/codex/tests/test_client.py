"""Tests for lib/client.py — Hindsight REST API client."""

import json
from unittest.mock import patch

import pytest
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


class TestRetainStrategy:
    @pytest.mark.parametrize("strategy", [None, "agent-session", "custom-session"])
    def test_strategy_is_item_scoped_without_changing_legacy_request(self, strategy):
        captured = []

        def fake_open(req, timeout=None):
            captured.append((req, timeout))
            return FakeHTTPResponse({"operation_id": "op-1"})

        client = HindsightClient("http://fake:9077")
        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = client.retain(
                "shared/bank",
                "source",
                "session-1",
                "codex",
                {"origin": "test"},
                ["session-1"],
                7,
                strategy=strategy,
            )

        assert result == {"operation_id": "op-1"}
        assert len(captured) == 1
        req, timeout = captured[0]
        assert req.full_url == "http://fake:9077/v1/default/banks/shared%2Fbank/memories"
        assert req.method == "POST"
        assert timeout == 7  # Existing positional timeout callers remain compatible.
        expected_item = {
            "content": "source",
            "document_id": "session-1",
            "context": "codex",
            "metadata": {"origin": "test"},
            "tags": ["session-1"],
        }
        if strategy is not None:
            expected_item["strategy"] = strategy
        assert json.loads(req.data) == {"items": [expected_item], "async": True}
