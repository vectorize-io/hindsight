"""Tests for lib/client.py — Hindsight REST API client."""

from unittest.mock import patch

from conftest import FakeHTTPResponse
from lib.client import USER_AGENT, HindsightClient


class TestUserAgentHeader:
    """Regression tests for #1041.

    The stdlib default ``Python-urllib/X.Y`` UA is blocked by Cloudflare
    with error 1010, so every request must carry our identifying UA.
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
        assert captured["ua"].startswith("hindsight-cursor-cli/")

    def test_health_check_sends_user_agent(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeHTTPResponse({}, status=200)

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.health_check(timeout=1)

        assert captured["ua"] == USER_AGENT


class TestURLValidation:
    def test_rejects_non_http_scheme(self):
        import pytest

        with pytest.raises(ValueError):
            HindsightClient("ftp://example.com")

    def test_rejects_missing_hostname(self):
        import pytest

        with pytest.raises(ValueError):
            HindsightClient("http://")


class TestRetainEncoding:
    def test_retain_posts_async_true(self):
        import json

        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse({})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.retain("bank", "content", document_id="d1", context="cursor-cli", tags=["x"])

        assert captured["body"]["async"] is True
        assert captured["body"]["items"][0]["context"] == "cursor-cli"
        assert captured["body"]["items"][0]["tags"] == ["x"]
        assert captured["body"]["items"][0]["document_id"] == "d1"
