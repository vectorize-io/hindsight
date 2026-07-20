"""
Config wiring for per-bank attribution and the configurable OpenRouter rerank URL.

- HINDSIGHT_API_LLM_SEND_BANK_AS_USER (default off, opt-in bool)
- HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER (default off, opt-in bool)
- HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL (default = previously hardcoded URL)

Deterministic, no network.
"""

import os
from dataclasses import fields
from unittest.mock import patch

from hindsight_api.config import DEFAULT_RERANKER_OPENROUTER_BASE_URL, HindsightConfig
from hindsight_api.engine.bank_attribution import reranker_bank_attribution_headers
from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env
from hindsight_api.engine.memory_engine import _current_bank_id


def _restore_env(saved: dict[str, str | None]) -> None:
    from hindsight_api.config import clear_config_cache

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    clear_config_cache()


def _make_full_config(**overrides):
    """Build a complete HindsightConfig from type-based defaults plus overrides.

    Mirrors the helper in test_reranker_timeouts.py so we can exercise the
    factory without touching real env/config.
    """
    defaults: dict = {}
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
        else:
            defaults[f.name] = None
    defaults.update(overrides)
    return HindsightConfig(**defaults)


class TestSendBankAsUserConfig:
    def test_default_is_false(self):
        from hindsight_api.config import clear_config_cache

        saved = {"HINDSIGHT_API_LLM_SEND_BANK_AS_USER": os.environ.get("HINDSIGHT_API_LLM_SEND_BANK_AS_USER")}
        os.environ.pop("HINDSIGHT_API_LLM_SEND_BANK_AS_USER", None)
        clear_config_cache()
        try:
            assert HindsightConfig.from_env().llm_send_bank_as_user is False
        finally:
            _restore_env(saved)

    def test_true_enables(self):
        from hindsight_api.config import clear_config_cache

        saved = {"HINDSIGHT_API_LLM_SEND_BANK_AS_USER": os.environ.get("HINDSIGHT_API_LLM_SEND_BANK_AS_USER")}
        os.environ["HINDSIGHT_API_LLM_SEND_BANK_AS_USER"] = "true"
        clear_config_cache()
        try:
            assert HindsightConfig.from_env().llm_send_bank_as_user is True
        finally:
            _restore_env(saved)

    def test_reranker_bank_header_default_false_and_true(self):
        from hindsight_api.config import clear_config_cache

        key = "HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER"
        saved = {key: os.environ.get(key)}
        os.environ.pop(key, None)
        clear_config_cache()
        try:
            assert HindsightConfig.from_env().reranker_send_bank_as_header is False
            token = _current_bank_id.set("bank-disabled")
            try:
                assert reranker_bank_attribution_headers() == {}
            finally:
                _current_bank_id.reset(token)
            os.environ[key] = "true"
            clear_config_cache()
            assert HindsightConfig.from_env().reranker_send_bank_as_header is True
            assert reranker_bank_attribution_headers() == {}
            token = _current_bank_id.set("bank-configured")
            try:
                assert reranker_bank_attribution_headers() == {"X-Hindsight-Bank-Id": "bank-configured"}
            finally:
                _current_bank_id.reset(token)
        finally:
            _restore_env(saved)


class TestRerankerOpenRouterBaseUrlConfig:
    def test_default_matches_previously_hardcoded_url(self):
        from hindsight_api.config import clear_config_cache

        saved = {
            "HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL": os.environ.get("HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL")
        }
        os.environ.pop("HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL", None)
        clear_config_cache()
        try:
            config = HindsightConfig.from_env()
            assert config.reranker_openrouter_base_url == DEFAULT_RERANKER_OPENROUTER_BASE_URL
            assert config.reranker_openrouter_base_url == "https://openrouter.ai/api/v1/rerank"
        finally:
            _restore_env(saved)

    def test_env_override_is_read(self):
        from hindsight_api.config import clear_config_cache

        saved = {
            "HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL": os.environ.get("HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL")
        }
        os.environ["HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL"] = "https://gateway.internal/v1/rerank"
        clear_config_cache()
        try:
            assert HindsightConfig.from_env().reranker_openrouter_base_url == "https://gateway.internal/v1/rerank"
        finally:
            _restore_env(saved)

    def test_factory_threads_configured_base_url_into_cross_encoder(self):
        """create_cross_encoder_from_env() honors the configured OpenRouter rerank URL."""
        config = _make_full_config(
            reranker_provider="openrouter",
            reranker_openrouter_api_key="k",
            reranker_openrouter_model="cohere/rerank-v3.5",
            reranker_openrouter_base_url="https://gateway.internal/v1/rerank",
            reranker_openrouter_timeout=60.0,
        )
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()
        assert encoder.base_url == "https://gateway.internal/v1/rerank"
