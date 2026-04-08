"""
Tests for Google Discovery Engine cross-encoder (Ranking REST API).

These tests cover:
1. Initialization (service account, ADC, missing project_id)
2. Predict (single query, multiple queries, batching, empty pairs, uninitialized)
3. Provider name
4. Factory function (create from env, validation errors)
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from hindsight_api.config import (
    ENV_RERANKER_GOOGLE_PROJECT_ID,
    ENV_RERANKER_PROVIDER,
    HindsightConfig,
)
from hindsight_api.engine.cross_encoder import GoogleCrossEncoder, create_cross_encoder_from_env


def _make_rank_response(records: list[tuple[str, float]]) -> dict:
    """Build a JSON response matching the Discovery Engine REST API format."""
    return {"records": [{"id": rid, "score": score} for rid, score in records]}


def _make_mock_httpx_client(responses: list[dict] | None = None) -> MagicMock:
    """Create a mock httpx.Client that returns predefined responses."""
    mock_client = MagicMock(spec=httpx.Client)
    if responses:
        side_effects = []
        for resp_json in responses:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.json.return_value = resp_json
            mock_resp.raise_for_status.return_value = None
            side_effects.append(mock_resp)
        mock_client.post.side_effect = side_effects
    return mock_client


def _make_mock_credentials() -> MagicMock:
    """Create mock credentials with a valid token."""
    creds = MagicMock()
    creds.valid = True
    creds.token = "mock-token"
    return creds


class TestGoogleCrossEncoder:
    """Unit tests for GoogleCrossEncoder with mocked httpx + google-auth."""

    async def test_initialization_adc_success(self):
        """Test successful initialization with ADC (no service account key)."""
        mock_creds = _make_mock_credentials()

        encoder = GoogleCrossEncoder(project_id="test-project")

        with patch("google.auth.default", return_value=(mock_creds, "test-project")):
            await encoder.initialize()

        assert encoder._client is not None
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

        assert encoder._client is not None
        assert encoder._credentials is mock_creds

    async def test_initialization_idempotent(self):
        """Test that calling initialize() twice is a no-op."""
        mock_creds = _make_mock_credentials()
        encoder = GoogleCrossEncoder(project_id="test-project")

        with patch("google.auth.default", return_value=(mock_creds, "test-project")):
            await encoder.initialize()
            first_client = encoder._client
            await encoder.initialize()
            assert encoder._client is first_client

    async def test_predict_single_query(self):
        """Test prediction with a single query and multiple documents."""
        mock_creds = _make_mock_credentials()
        mock_client = _make_mock_httpx_client([
            _make_rank_response([("1", 0.95), ("0", 0.30)]),
        ])

        encoder = GoogleCrossEncoder(project_id="test-project")
        with patch("google.auth.default", return_value=(mock_creds, "p")):
            await encoder.initialize()
        encoder._client = mock_client

        scores = await encoder.predict([
            ("What is AI?", "AI is artificial intelligence"),
            ("What is AI?", "The sky is blue"),
        ])

        assert len(scores) == 2
        assert scores[0] == 0.30  # id="0" -> index 0
        assert scores[1] == 0.95  # id="1" -> index 1
        mock_client.post.assert_called_once()

    async def test_predict_multiple_queries(self):
        """Test prediction with multiple distinct queries."""
        mock_creds = _make_mock_credentials()
        mock_client = _make_mock_httpx_client([
            _make_rank_response([("0", 0.9), ("1", 0.1)]),
            _make_rank_response([("0", 0.8)]),
        ])

        encoder = GoogleCrossEncoder(project_id="test-project")
        with patch("google.auth.default", return_value=(mock_creds, "p")):
            await encoder.initialize()
        encoder._client = mock_client

        scores = await encoder.predict([
            ("Query A", "Doc A1"),
            ("Query A", "Doc A2"),
            ("Query B", "Doc B1"),
        ])

        assert len(scores) == 3
        assert scores[0] == 0.9
        assert scores[1] == 0.1
        assert scores[2] == 0.8
        assert mock_client.post.call_count == 2

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
        mock_creds = _make_mock_credentials()
        mock_client = _make_mock_httpx_client([
            _make_rank_response([(str(i), 0.5) for i in range(200)]),
            _make_rank_response([(str(i), 0.3) for i in range(50)]),
        ])

        encoder = GoogleCrossEncoder(project_id="test-project")
        with patch("google.auth.default", return_value=(mock_creds, "p")):
            await encoder.initialize()
        encoder._client = mock_client

        pairs = [("same query", f"doc {i}") for i in range(250)]
        scores = await encoder.predict(pairs)

        assert len(scores) == 250
        assert mock_client.post.call_count == 2

    async def test_auth_header_sent(self):
        """Test that Authorization header is sent with requests."""
        mock_creds = _make_mock_credentials()
        mock_creds.token = "test-bearer-token"
        mock_client = _make_mock_httpx_client([
            _make_rank_response([("0", 0.9)]),
        ])

        encoder = GoogleCrossEncoder(project_id="test-project")
        with patch("google.auth.default", return_value=(mock_creds, "p")):
            await encoder.initialize()
        encoder._client = mock_client

        await encoder.predict([("q", "d")])

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-bearer-token"

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
