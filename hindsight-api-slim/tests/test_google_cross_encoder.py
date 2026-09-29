"""
Tests for Google Discovery Engine cross-encoder (Ranking REST API).

These tests cover:
1. Initialization (service account, ADC, missing project_id)
2. Predict (single query, multiple queries, batching, empty pairs, uninitialized)
3. Provider name
4. Factory function (create from env, validation errors)

The Ranking API is served by a real in-process HTTP server (``tests/aiohttp_stub.py``);
google-auth is mocked.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from hindsight_api.config import (
    HindsightConfig,
)
from hindsight_api.engine.cross_encoder import GoogleCrossEncoder, create_cross_encoder_from_env
from tests.aiohttp_stub import stub_server


def _make_rank_response(records: list[tuple[str, float]]) -> dict:
    """Build a JSON response matching the Discovery Engine REST API format."""
    return {"records": [{"id": rid, "score": score} for rid, score in records]}


class _RankStub:
    """Serves the given rank responses in order and records each request."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def handler(self, request: web.Request) -> web.StreamResponse:
        self.requests.append({"headers": dict(request.headers), "json": await request.json()})
        return web.json_response(self._responses.pop(0))


def _make_mock_credentials() -> MagicMock:
    """Create mock credentials with a valid token."""
    creds = MagicMock()
    creds.valid = True
    creds.token = "mock-token"
    return creds


async def _initialized_encoder(base_url: str, creds: MagicMock) -> GoogleCrossEncoder:
    """An initialized encoder whose rank URL points at the stub server."""
    encoder = GoogleCrossEncoder(project_id="test-project")
    with patch("google.auth.default", return_value=(creds, "p")):
        await encoder.initialize()
    encoder._rank_url = f"{base_url}/rank"
    return encoder


class TestGoogleCrossEncoder:
    """Unit tests for GoogleCrossEncoder with a stubbed Ranking API + mocked google-auth."""

    async def test_initialization_adc_success(self):
        """Test successful initialization with ADC (no service account key)."""
        mock_creds = _make_mock_credentials()

        encoder = GoogleCrossEncoder(project_id="test-project")

        with patch("google.auth.default", return_value=(mock_creds, "test-project")):
            await encoder.initialize()

        assert encoder._rank_url is not None
        assert encoder._credentials is mock_creds
        assert encoder.provider_name == "google"
        assert "test-project" in encoder._rank_url

    async def test_initialization_service_account(self):
        """Test initialization with service account key."""
        mock_creds = _make_mock_credentials()

        encoder = GoogleCrossEncoder(
            project_id="test-project",
            service_account_key="/path/to/key.json",
        )

        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            await encoder.initialize()

        assert encoder._rank_url is not None
        assert encoder._credentials is mock_creds

    async def test_initialization_idempotent(self):
        """Test that calling initialize() twice is a no-op."""
        mock_creds = _make_mock_credentials()
        encoder = GoogleCrossEncoder(project_id="test-project")

        with patch("google.auth.default", return_value=(mock_creds, "test-project")) as default:
            await encoder.initialize()
            await encoder.initialize()

        default.assert_called_once()

    async def test_predict_single_query(self):
        """Test prediction with a single query and multiple documents."""
        stub = _RankStub([_make_rank_response([("1", 0.95), ("0", 0.30)])])

        async with stub_server(stub.handler) as base_url:
            encoder = await _initialized_encoder(base_url, _make_mock_credentials())
            scores = await encoder.predict(
                [
                    ("What is AI?", "AI is artificial intelligence"),
                    ("What is AI?", "The sky is blue"),
                ]
            )

        assert len(scores) == 2
        assert scores[0] == 0.30  # id="0" -> index 0
        assert scores[1] == 0.95  # id="1" -> index 1
        assert len(stub.requests) == 1
        body = stub.requests[0]["json"]
        assert body["model"] == "semantic-ranker-default-004"
        assert body["query"] == "What is AI?"
        assert body["topN"] == 2
        assert [r["id"] for r in body["records"]] == ["0", "1"]

    async def test_predict_multiple_queries(self):
        """Test prediction with multiple distinct queries."""
        stub = _RankStub(
            [
                _make_rank_response([("0", 0.9), ("1", 0.1)]),
                _make_rank_response([("0", 0.8)]),
            ]
        )

        async with stub_server(stub.handler) as base_url:
            encoder = await _initialized_encoder(base_url, _make_mock_credentials())
            scores = await encoder.predict(
                [
                    ("Query A", "Doc A1"),
                    ("Query A", "Doc A2"),
                    ("Query B", "Doc B1"),
                ]
            )

        assert len(scores) == 3
        assert scores[0] == 0.9
        assert scores[1] == 0.1
        assert scores[2] == 0.8
        assert len(stub.requests) == 2

    async def test_predict_empty_pairs(self):
        """Test that empty pairs returns empty list."""
        mock_creds = _make_mock_credentials()
        encoder = GoogleCrossEncoder(project_id="test-project")

        with patch("google.auth.default", return_value=(mock_creds, "p")):
            await encoder.initialize()

        scores = await encoder.predict([])
        assert scores == []

    async def test_predict_not_initialized(self):
        """Test that predict raises if not initialized."""
        encoder = GoogleCrossEncoder(project_id="test-project")
        with pytest.raises(RuntimeError, match="not initialized"):
            await encoder.predict([("q", "d")])

    async def test_predict_batching(self):
        """Test that >200 records are split into batches."""
        stub = _RankStub(
            [
                _make_rank_response([(str(i), 0.5) for i in range(200)]),
                _make_rank_response([(str(i), 0.3) for i in range(50)]),
            ]
        )

        pairs = [("same query", f"doc {i}") for i in range(250)]
        async with stub_server(stub.handler) as base_url:
            encoder = await _initialized_encoder(base_url, _make_mock_credentials())
            scores = await encoder.predict(pairs)

        assert len(scores) == 250
        assert scores[:200] == [0.5] * 200
        assert scores[200:] == [0.3] * 50
        assert [len(r["json"]["records"]) for r in stub.requests] == [200, 50]

    async def test_auth_header_sent(self):
        """Test that Authorization header is sent with requests."""
        mock_creds = _make_mock_credentials()
        mock_creds.token = "test-bearer-token"
        stub = _RankStub([_make_rank_response([("0", 0.9)])])

        async with stub_server(stub.handler) as base_url:
            encoder = await _initialized_encoder(base_url, mock_creds)
            await encoder.predict([("q", "d")])

        assert stub.requests[0]["headers"]["Authorization"] == "Bearer test-bearer-token"
        mock_creds.refresh.assert_not_called()

    async def test_expired_token_is_refreshed_before_the_request(self):
        """Invalid credentials are refreshed (off the loop) and the fresh token is sent."""
        mock_creds = _make_mock_credentials()
        mock_creds.valid = False
        mock_creds.token = "stale-token"

        def refresh(_request: Any) -> None:
            mock_creds.token = "fresh-token"
            mock_creds.valid = True

        mock_creds.refresh.side_effect = refresh
        stub = _RankStub([_make_rank_response([("0", 0.9)])])

        async with stub_server(stub.handler) as base_url:
            encoder = await _initialized_encoder(base_url, mock_creds)
            await encoder.predict([("q", "d")])

        mock_creds.refresh.assert_called_once()
        assert stub.requests[0]["headers"]["Authorization"] == "Bearer fresh-token"

    def test_provider_name(self):
        assert GoogleCrossEncoder(project_id="p").provider_name == "google"

    def test_default_model(self):
        encoder = GoogleCrossEncoder(project_id="p")
        assert encoder.model == "semantic-ranker-default-004"

    def test_custom_model(self):
        encoder = GoogleCrossEncoder(project_id="p", model="semantic-ranker-fast-004")
        assert encoder.model == "semantic-ranker-fast-004"

    def test_default_location(self):
        encoder = GoogleCrossEncoder(project_id="p")
        assert encoder.location == "global"


class TestGoogleCrossEncoderFactory:
    """Tests for create_cross_encoder_from_env() with 'google' provider."""

    def _make_config(self, **overrides) -> HindsightConfig:
        from dataclasses import fields

        defaults = {}
        for f in fields(HindsightConfig):
            if f.type == "str":
                defaults[f.name] = ""
            elif f.type == "str | None":
                defaults[f.name] = None
            elif f.type == "int":
                defaults[f.name] = 0
            elif f.type == "int | None":
                defaults[f.name] = None
            elif f.type == "float":
                defaults[f.name] = 0.0
            elif f.type == "float | None":
                defaults[f.name] = None
            elif f.type == "bool":
                defaults[f.name] = False
            elif f.type == "list | None":
                defaults[f.name] = None
            elif str(f.type).startswith("list["):
                defaults[f.name] = []
            else:
                defaults[f.name] = None

        defaults["reranker_provider"] = "google"
        defaults["reranker_google_model"] = "semantic-ranker-default-004"
        defaults["reranker_google_project_id"] = "test-project"
        defaults["reranker_google_service_account_key"] = None

        defaults.update(overrides)
        return HindsightConfig(**defaults)

    def test_create_with_project_id(self):
        config = self._make_config()
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()
        assert isinstance(encoder, GoogleCrossEncoder)
        assert encoder.provider_name == "google"
        assert encoder.project_id == "test-project"
        assert encoder.service_account_key is None

    def test_create_with_service_account(self):
        config = self._make_config(reranker_google_service_account_key="/path/to/key.json")
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()
        assert isinstance(encoder, GoogleCrossEncoder)
        assert encoder.service_account_key == "/path/to/key.json"

    def test_create_missing_project_id(self):
        config = self._make_config(reranker_google_project_id=None)
        with patch("hindsight_api.config.get_config", return_value=config):
            with pytest.raises(ValueError, match="is required"):
                create_cross_encoder_from_env()

    def test_create_with_custom_model(self):
        config = self._make_config(reranker_google_model="semantic-ranker-fast-004")
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()
        assert encoder.model == "semantic-ranker-fast-004"
