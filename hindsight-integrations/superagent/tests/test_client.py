"""Unit tests for Hindsight-Superagent client resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hindsight_superagent import configure, reset_config
from hindsight_superagent._client import resolve_hindsight_client, resolve_safety_client


class TestResolveHindsightClient:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_returns_explicit_client(self) -> None:
        client = MagicMock()
        result = resolve_hindsight_client(client, None, None)
        assert result is client

    def test_creates_client_from_args(self) -> None:
        with patch("hindsight_superagent._client.Hindsight") as mock_cls:
            mock_cls.return_value = MagicMock()
            resolve_hindsight_client(None, "http://localhost:8888", "test-key")
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8888"
            assert call_kwargs["api_key"] == "test-key"
            assert "hindsight-superagent" in call_kwargs["user_agent"]

    def test_creates_client_from_global_config(self) -> None:
        configure(hindsight_api_url="http://config:8888", api_key="config-key")
        with patch("hindsight_superagent._client.Hindsight") as mock_cls:
            mock_cls.return_value = MagicMock()
            resolve_hindsight_client(None, None, None)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "http://config:8888"
            assert call_kwargs["api_key"] == "config-key"

    def test_creates_client_without_api_key(self) -> None:
        with patch("hindsight_superagent._client.Hindsight") as mock_cls:
            mock_cls.return_value = MagicMock()
            resolve_hindsight_client(None, "http://localhost:8888", None)
            call_kwargs = mock_cls.call_args.kwargs
            assert "api_key" not in call_kwargs

    def test_defaults_to_cloud_url(self) -> None:
        with patch("hindsight_superagent._client.Hindsight") as mock_cls:
            mock_cls.return_value = MagicMock()
            resolve_hindsight_client(None, None, None)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "https://api.hindsight.vectorize.io"

    def test_explicit_args_override_config(self) -> None:
        configure(hindsight_api_url="http://config:8888", api_key="config-key")
        with patch("hindsight_superagent._client.Hindsight") as mock_cls:
            mock_cls.return_value = MagicMock()
            resolve_hindsight_client(None, "http://explicit:9999", "explicit-key")
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["base_url"] == "http://explicit:9999"
            assert call_kwargs["api_key"] == "explicit-key"


class TestResolveSafetyClient:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_returns_explicit_client(self) -> None:
        client = MagicMock()
        result = resolve_safety_client(client, None)
        assert result is client

    def test_creates_client_from_args(self) -> None:
        with patch("hindsight_superagent._client.create_client") as mock_fn:
            mock_fn.return_value = MagicMock()
            resolve_safety_client(None, "sa-test-key")
            mock_fn.assert_called_once_with(api_key="sa-test-key")

    def test_creates_client_from_global_config(self) -> None:
        configure(hindsight_api_url="http://localhost:8888", superagent_api_key="sa-config-key")
        with patch("hindsight_superagent._client.create_client") as mock_fn:
            mock_fn.return_value = MagicMock()
            resolve_safety_client(None, None)
            mock_fn.assert_called_once_with(api_key="sa-config-key")

    def test_creates_client_without_key(self) -> None:
        with patch("hindsight_superagent._client.create_client") as mock_fn:
            mock_fn.return_value = MagicMock()
            resolve_safety_client(None, None)
            mock_fn.assert_called_once_with()

    def test_explicit_key_overrides_config(self) -> None:
        configure(hindsight_api_url="http://localhost:8888", superagent_api_key="config-key")
        with patch("hindsight_superagent._client.create_client") as mock_fn:
            mock_fn.return_value = MagicMock()
            resolve_safety_client(None, "explicit-key")
            mock_fn.assert_called_once_with(api_key="explicit-key")
