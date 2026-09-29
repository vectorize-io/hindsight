"""Tests for the shared helpers in tools/_client.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools._client import build_client, parse_tags


class TestParseTags:
    def test_none_returns_none(self):
        assert parse_tags(None) is None

    def test_empty_returns_none(self):
        assert parse_tags("") is None

    def test_whitespace_only_returns_none(self):
        assert parse_tags("  ,   , ") is None

    def test_single(self):
        assert parse_tags("vip") == ["vip"]

    def test_multiple_with_whitespace(self):
        assert parse_tags("a, b ,c") == ["a", "b", "c"]


class TestBuildClient:
    def test_with_api_key(self):
        with patch("tools._client.Hindsight") as mock_h:
            build_client({"api_url": "https://api.example.com", "api_key": "hsk_x"})
            mock_h.assert_called_once_with(base_url="https://api.example.com", timeout=30.0, api_key="hsk_x")

    def test_without_api_key(self):
        with patch("tools._client.Hindsight") as mock_h:
            build_client({"api_url": "http://localhost:8888"})
            mock_h.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0)

    def test_strips_trailing_slash(self):
        with patch("tools._client.Hindsight") as mock_h:
            build_client({"api_url": "https://api.example.com/"})
            assert mock_h.call_args.kwargs["base_url"] == "https://api.example.com"

    def test_empty_key_treated_as_missing(self):
        with patch("tools._client.Hindsight") as mock_h:
            build_client({"api_url": "http://localhost:8888", "api_key": ""})
            assert "api_key" not in mock_h.call_args.kwargs
