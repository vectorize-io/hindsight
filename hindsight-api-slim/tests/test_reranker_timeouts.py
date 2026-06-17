"""
Per-provider HTTP timeout wiring for remote rerankers.

Covers issue #1807: each HTTP-based reranker provider must accept a configurable
timeout via env var so workloads with large batches don't hit the previously
hardcoded 60s ceiling.
"""

from dataclasses import fields
from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env


def _make_config(**overrides) -> HindsightConfig:
    """Build a HindsightConfig with field-type-appropriate zero values plus overrides."""
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


@pytest.mark.parametrize(
    "provider, extra, timeout_field, attr_path",
    [
        (
            "cohere",
            {"reranker_cohere_api_key": "k", "reranker_cohere_base_url": "https://example/rerank"},
            "reranker_cohere_timeout",
            ("_http_client", "timeout"),
        ),
        (
            "openrouter",
            {
                "reranker_openrouter_api_key": "k",
                "reranker_openrouter_base_url": "https://openrouter.ai/api/v1/rerank",
            },
            "reranker_openrouter_timeout",
            ("_http_client", "timeout"),
        ),
        (
            "zeroentropy",
            {"reranker_zeroentropy_api_key": "k"},
            "reranker_zeroentropy_timeout",
            ("_client", "timeout"),
        ),
        (
            "siliconflow",
            {
                "reranker_siliconflow_api_key": "k",
                "reranker_siliconflow_base_url": "https://api.siliconflow.cn/v1",
            },
            "reranker_siliconflow_timeout",
            ("_client", "timeout"),
        ),
        (
            "alibaba",
            {"reranker_alibaba_api_key": "k"},
            "reranker_alibaba_timeout",
            ("_client", "timeout"),
        ),
        (
            "litellm",
            {"reranker_litellm_api_base": "http://localhost:4000"},
            "reranker_litellm_timeout",
            ("timeout",),
        ),
        (
            "litellm-sdk",
            {"reranker_litellm_sdk_api_key": "k"},
            "reranker_litellm_sdk_timeout",
            ("timeout",),
        ),
        (
            "google",
            {"reranker_google_project_id": "test-project"},
            "reranker_google_timeout",
            ("timeout",),
        ),
    ],
)
def test_factory_threads_per_provider_timeout(provider, extra, timeout_field, attr_path):
    """Env-configured timeout reaches the provider instance (or its inner HTTP client)."""
    custom_timeout = 300.0
    config = _make_config(
        reranker_provider=provider,
        **{timeout_field: custom_timeout},
        **extra,
    )
    with patch("hindsight_api.config.get_config", return_value=config):
        encoder = create_cross_encoder_from_env()

    obj = encoder
    for part in attr_path:
        obj = getattr(obj, part)
    assert obj == custom_timeout


def test_default_timeouts_preserve_60s_behavior():
    """Unset env keeps the previously hardcoded 60.0s default for HTTP providers."""
    config = HindsightConfig.from_env()
    assert config.reranker_cohere_timeout == 60.0
    assert config.reranker_openrouter_timeout == 60.0
    assert config.reranker_zeroentropy_timeout == 60.0
    assert config.reranker_siliconflow_timeout == 60.0
    assert config.reranker_alibaba_timeout == 60.0
    assert config.reranker_litellm_timeout == 60.0
    assert config.reranker_litellm_sdk_timeout == 60.0
    assert config.reranker_google_timeout == 60.0
