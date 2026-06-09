"""Tests for HindsightProvider credential validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from provider.hindsight import HindsightProvider


def _provider() -> HindsightProvider:
    """Construct the provider without invoking the dify_plugin base init."""
    return HindsightProvider.__new__(HindsightProvider)


class TestValidateCredentials:
    def test_missing_api_url_raises(self):
        with pytest.raises(ToolProviderCredentialValidationError, match="API URL"):
            _provider()._validate_credentials({"api_url": "", "api_key": "hsk_x"})

    def test_health_ok_with_key(self):
        with patch("provider.hindsight.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            _provider()._validate_credentials({"api_url": "https://api.example.com", "api_key": "hsk_x"})
            assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer hsk_x"}

    def test_health_ok_without_key(self):
        with patch("provider.hindsight.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            _provider()._validate_credentials({"api_url": "http://localhost:8888"})
            assert mock_get.call_args.kwargs["headers"] == {}

    def test_401_raises_with_key_message(self):
        with patch("provider.hindsight.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=401)
            with pytest.raises(ToolProviderCredentialValidationError, match="API key"):
                _provider()._validate_credentials({"api_url": "https://api.example.com", "api_key": "bad"})

    def test_500_raises(self):
        with patch("provider.hindsight.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            with pytest.raises(ToolProviderCredentialValidationError, match="HTTP 500"):
                _provider()._validate_credentials({"api_url": "https://api.example.com", "api_key": "hsk_x"})

    def test_connection_error_raises_with_url(self):
        import requests

        with patch("provider.hindsight.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("boom")
            with pytest.raises(ToolProviderCredentialValidationError, match="Could not reach"):
                _provider()._validate_credentials({"api_url": "https://api.example.com", "api_key": "hsk_x"})
