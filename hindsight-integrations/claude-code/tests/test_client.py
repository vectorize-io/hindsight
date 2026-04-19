"""Tests for lib/client.py — Hindsight REST API client."""

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from lib.client import USER_AGENT, HindsightClient, _validate_api_url


class TestValidateApiUrl:
    def test_valid_http(self):
        assert _validate_api_url("http://localhost:9077") == "http://localhost:9077"

    def test_valid_https(self):
        assert _validate_api_url("https://api.example.com/") == "https://api.example.com"

    def test_trailing_slash_stripped(self):
        assert _validate_api_url("http://host:8080/") == "http://host:8080"

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="http or https"):
            _validate_api_url("ftp://host")

    def test_no_hostname_raises(self):
        with pytest.raises(ValueError):
            _validate_api_url("http://")


class FakeResp:
    def __init__(self, data, status=200):
        self.status = status
        self._body = json.dumps(data).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class TestHindsightClientInit:
    def test_rejects_non_http_url(self):
        with pytest.raises(ValueError):
            HindsightClient("ftp://bad")

    def test_stores_token(self):
        c = HindsightClient("http://localhost:9077", api_token="tok123")
        assert c.api_token == "tok123"

    def test_no_token(self):
        c = HindsightClient("http://localhost:9077")
        assert c.api_token is None


class TestHindsightClientRecall:
    def test_posts_to_correct_path(self):
        c = HindsightClient("http://localhost:9077")
        response_data = {"results": [{"text": "Paris", "type": "world"}]}
        with patch("urllib.request.urlopen", return_value=FakeResp(response_data)):
            resp = c.recall("my-bank", "capital of France")
        assert resp["results"][0]["text"] == "Paris"

    def test_bank_id_url_encoded(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["url"] = req.full_url
            return FakeResp({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.recall("bank with spaces", "query")

        assert "bank%20with%20spaces" in captured["url"]

    def test_includes_auth_header_when_token_set(self):
        c = HindsightClient("http://localhost:9077", api_token="mytoken")
        captured = {}

        def fake_open(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return FakeResp({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.recall("bank", "query")

        assert "Authorization" in captured["headers"]
        assert "mytoken" in captured["headers"]["Authorization"]

    def test_no_auth_header_without_token(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return FakeResp({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.recall("bank", "query")

        assert "Authorization" not in captured["headers"]

    def test_sends_user_agent_header(self):
        # Regression test for #1041: the stdlib default "Python-urllib/X.Y" UA
        # is blocked by Cloudflare with error 1010, so we must always send our own.
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["ua"] = req.get_header("User-agent")
            return FakeResp({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.recall("bank", "query")

        assert captured["ua"] == USER_AGENT
        assert captured["ua"].startswith("hindsight-claude-code/")

    def test_http_error_raises_runtime_error(self):
        c = HindsightClient("http://localhost:9077")
        err = urllib.error.HTTPError(
            url="http://localhost:9077/v1/default/banks/b/memories/recall",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO(b"server exploded"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                c.recall("b", "query")

    def test_sends_budget_and_types(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return FakeResp({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.recall("bank", "query", budget="high", types=["world", "experience"])

        assert captured["body"]["budget"] == "high"
        assert captured["body"]["types"] == ["world", "experience"]


class TestHindsightClientRetain:
    def test_posts_with_async_true(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return FakeResp({"status": "accepted"})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.retain("bank", "transcript content", document_id="doc-1", context="claude-code")

        assert captured["body"]["async"] is True
        assert captured["body"]["items"][0]["content"] == "transcript content"
        assert captured["body"]["items"][0]["context"] == "claude-code"

    def test_bank_id_encoded_in_retain_path(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["url"] = req.full_url
            return FakeResp({})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.retain("my::bank", "content")

        assert "my%3A%3Abank" in captured["url"]


class TestHindsightClientHealthCheck:
    def test_returns_true_on_200(self):
        c = HindsightClient("http://localhost:9077")
        with patch("urllib.request.urlopen", return_value=FakeResp({}, status=200)):
            with patch("time.sleep"):  # don't actually sleep
                assert c.health_check() is True

    def test_returns_false_after_retries(self):
        c = HindsightClient("http://localhost:9077")
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with patch("time.sleep"):
                assert c.health_check() is False

    def test_retries_on_failure(self):
        c = HindsightClient("http://localhost:9077")
        call_count = 0

        def flaky(*_a, **_kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("not yet")
            return FakeResp({}, status=200)

        with patch("urllib.request.urlopen", side_effect=flaky):
            with patch("time.sleep"):
                result = c.health_check()

        assert result is True
        assert call_count == 3


class TestHindsightClientSetBankMission:
    def test_patches_config_endpoint(self):
        c = HindsightClient("http://localhost:9077")
        captured = {}

        def fake_open(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.method
            captured["body"] = json.loads(req.data.decode())
            return FakeResp({})

        with patch("urllib.request.urlopen", side_effect=fake_open):
            c.set_bank_mission("my-bank", "I am Claude", retain_mission="Extract facts")

        assert captured["method"] == "PATCH"
        assert "my-bank" in captured["url"]
        assert captured["body"]["updates"]["reflect_mission"] == "I am Claude"
        assert captured["body"]["updates"]["retain_mission"] == "Extract facts"
