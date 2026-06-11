"""Client resolution: cloud default + env/config overrides."""

from unittest.mock import MagicMock, patch

from hindsight_agent_framework._client import resolve_client
from hindsight_agent_framework.config import (
    DEFAULT_HINDSIGHT_API_URL,
    configure,
    reset_config,
)


def test_resolve_uses_cloud_default_when_nothing_supplied(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    reset_config()
    with patch("hindsight_agent_framework._client.Hindsight") as mock_cls:
        mock_cls.return_value = MagicMock()
        resolve_client(None, None, None)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == DEFAULT_HINDSIGHT_API_URL
        assert kwargs["user_agent"].startswith("hindsight-agent-framework/")
        assert "api_key" not in kwargs  # no key → omitted
    reset_config()


def test_resolve_honors_env_key(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_KEY", "env-key")
    reset_config()
    with patch("hindsight_agent_framework._client.Hindsight") as mock_cls:
        mock_cls.return_value = MagicMock()
        resolve_client(None, None, None)
        assert mock_cls.call_args.kwargs["api_key"] == "env-key"
    reset_config()


def test_resolve_uses_configured_url(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    configure(hindsight_api_url="http://localhost:8888", api_key="cfg-key")
    with patch("hindsight_agent_framework._client.Hindsight") as mock_cls:
        mock_cls.return_value = MagicMock()
        resolve_client(None, None, None)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "http://localhost:8888"
        assert kwargs["api_key"] == "cfg-key"
    reset_config()


def test_explicit_args_win(monkeypatch):
    configure(hindsight_api_url="http://configured", api_key="cfg")
    with patch("hindsight_agent_framework._client.Hindsight") as mock_cls:
        mock_cls.return_value = MagicMock()
        resolve_client(None, "http://explicit", "explicit-key")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == "http://explicit"
        assert kwargs["api_key"] == "explicit-key"
    reset_config()
