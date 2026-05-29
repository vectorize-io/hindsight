"""Tests for Codex provider base URL handling."""

from unittest.mock import patch

from hindsight_api.engine.providers.codex_llm import CodexLLM


def _make(base_url: str) -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("at", "acct")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value="rt"),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url=base_url,
            model="gpt-5.4-mini",
        )


def test_codex_uses_chatgpt_backend_when_base_url_empty():
    assert _make("").base_url == "https://chatgpt.com/backend-api"


def test_codex_ignores_inherited_openai_compatible_v1_base_url():
    assert _make("https://newapi.example.com/v1").base_url == "https://chatgpt.com/backend-api"


def test_codex_preserves_explicit_codex_backend_base_url_without_trailing_slash():
    assert _make("https://chatgpt.example.com/backend-api/").base_url == "https://chatgpt.example.com/backend-api"
