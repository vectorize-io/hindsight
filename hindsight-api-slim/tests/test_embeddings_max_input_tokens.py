"""Provider-agnostic embedding input truncation.

The cap lives at the single choke point (`generate_embeddings_batch`) so every
provider and every call path (retain, recall queries, consolidation, import) gets
identical, model-agnostic truncation before any backend's `encode()` runs.

Config is exposed as the generic `HINDSIGHT_API_EMBEDDINGS_MAX_INPUT_TOKENS`, with the
old LiteLLM-SDK-specific `HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS` kept as
a deprecated alias for backward compatibility.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hindsight_api.config import (
    ENV_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS,
    ENV_EMBEDDINGS_MAX_INPUT_TOKENS,
    HindsightConfig,
)
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.token_encoding import get_token_encoding


class _FakeBackend:
    """Records the texts each `encode_*` call actually receives."""

    provider_name = "fake"

    def __init__(self, dimension: int = 3) -> None:
        self._dimension = dimension
        self.received: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.received = list(texts)
        return [[0.1] * self._dimension for _ in texts]

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        return self.encode_documents(texts)


def _patch_cap(value: int | None):
    return patch.object(
        embedding_utils,
        "get_config",
        return_value=SimpleNamespace(embeddings_max_input_tokens=value),
    )


class TestTruncateInputs:
    def test_truncates_oversized_and_warns(self, caplog):
        backend = _FakeBackend()
        long_text = "word " * 500  # far more than 50 tokens
        with caplog.at_level(logging.WARNING):
            result = embedding_utils._truncate_inputs([long_text], 50, backend)

        assert len(get_token_encoding().encode(result[0])) <= 50
        assert result[0] != long_text
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "truncated" in r.message]
        assert warnings, "expected a truncation warning"
        # The warning names the generic env var (so operators know which knob to turn)
        # and the provider, not a model-specific message.
        assert ENV_EMBEDDINGS_MAX_INPUT_TOKENS in warnings[0].getMessage()
        assert "fake" in warnings[0].getMessage()

    def test_short_input_untouched_no_warning(self, caplog):
        backend = _FakeBackend()
        with caplog.at_level(logging.WARNING):
            result = embedding_utils._truncate_inputs(["short text"], 50, backend)

        assert result == ["short text"]
        assert not any("truncated" in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestGenerateEmbeddingsBatch:
    async def test_cap_applied_before_backend(self):
        backend = _FakeBackend()
        long_text = "word " * 500
        with _patch_cap(50):
            await embedding_utils.generate_embeddings_batch(backend, [long_text])

        assert backend.received, "backend was never called"
        assert len(get_token_encoding().encode(backend.received[0])) <= 50
        assert backend.received[0] != long_text

    async def test_no_cap_passes_verbatim(self):
        backend = _FakeBackend()
        long_text = "word " * 500
        with _patch_cap(None):
            await embedding_utils.generate_embeddings_batch(backend, [long_text])

        assert backend.received == [long_text]


class TestConfigWiring:
    def test_generic_env_parsed(self, monkeypatch):
        monkeypatch.setenv(ENV_EMBEDDINGS_MAX_INPUT_TOKENS, "8192")
        monkeypatch.delenv(ENV_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS, raising=False)
        assert HindsightConfig.from_env().embeddings_max_input_tokens == 8192

    def test_deprecated_litellm_alias_still_works(self, monkeypatch):
        monkeypatch.delenv(ENV_EMBEDDINGS_MAX_INPUT_TOKENS, raising=False)
        monkeypatch.setenv(ENV_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS, "4096")
        assert HindsightConfig.from_env().embeddings_max_input_tokens == 4096

    def test_generic_takes_precedence_over_alias(self, monkeypatch):
        monkeypatch.setenv(ENV_EMBEDDINGS_MAX_INPUT_TOKENS, "8192")
        monkeypatch.setenv(ENV_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS, "4096")
        assert HindsightConfig.from_env().embeddings_max_input_tokens == 8192

    def test_default_is_disabled(self, monkeypatch):
        monkeypatch.delenv(ENV_EMBEDDINGS_MAX_INPUT_TOKENS, raising=False)
        monkeypatch.delenv(ENV_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS, raising=False)
        assert HindsightConfig.from_env().embeddings_max_input_tokens is None
