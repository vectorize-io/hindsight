"""Provider-agnostic embedding input truncation.

The cap lives at the single choke point (`generate_embeddings_batch`) so every
provider and every call path (retain, recall queries, consolidation, import) gets
identical, model-agnostic truncation before any backend's `encode()` runs.

Config is exposed as the generic `HINDSIGHT_API_EMBEDDINGS_MAX_INPUT_TOKENS`, with the
old LiteLLM-SDK-specific `HINDSIGHT_API_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS` kept as
a deprecated alias for backward compatibility. It defaults to 8192 — the input limit of
essentially every remote embedding model — because leaving it off let a knowledge page
sized to its own 8192-token generation budget fail its refresh permanently (#4165); 0
opts out.
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
from hindsight_api.engine.token_encoding import count_tokens


class _FakeBackend:
    """Records the texts each `encode_*` call actually receives."""

    provider_name = "fake"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self, dimension: int = 3) -> None:
        self._dimension = dimension
        self.received: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.received = list(texts)
        return [[0.1] * self._dimension for _ in texts]

    async def encode_query(self, texts: list[str]) -> list[list[float]]:
        return await self.encode_documents(texts)


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
            result = embedding_utils._truncate_inputs([long_text], 50, backend, "document")

        assert count_tokens(result[0]) <= 50
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
            result = embedding_utils._truncate_inputs(["short text"], 50, backend, "document")

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
        assert count_tokens(backend.received[0]) <= 50
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

    def test_default_is_the_common_model_limit(self, monkeypatch):
        monkeypatch.delenv(ENV_EMBEDDINGS_MAX_INPUT_TOKENS, raising=False)
        monkeypatch.delenv(ENV_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS, raising=False)
        assert HindsightConfig.from_env().embeddings_max_input_tokens == 8192

    def test_zero_opts_out(self, monkeypatch):
        """0 is the only way to send text uncapped now that the default is a real limit."""
        monkeypatch.setenv(ENV_EMBEDDINGS_MAX_INPUT_TOKENS, "0")
        monkeypatch.delenv(ENV_EMBEDDINGS_LITELLM_SDK_MAX_INPUT_TOKENS, raising=False)
        assert HindsightConfig.from_env().embeddings_max_input_tokens is None


class _PrefixedBackend(_FakeBackend):
    """An asymmetric model whose instruction `Embeddings._encode_prefixed` glues on
    AFTER the cap has been applied."""

    query_prefix = "query: "
    passage_prefix = "passage: "

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return await super().encode_documents([f"{self.passage_prefix}{t}" for t in texts])

    async def encode_query(self, texts: list[str]) -> list[list[float]]:
        return await super().encode_documents([f"{self.query_prefix}{t}" for t in texts])


@pytest.mark.asyncio
class TestFinalPayloadFitsTheCap:
    """What must fit the model's limit is the string that goes over the wire."""

    @pytest.mark.parametrize("input_type", ["document", "query"])
    async def test_prefix_is_charged_against_the_budget(self, input_type):
        backend = _PrefixedBackend()
        with _patch_cap(50):
            await embedding_utils.generate_embeddings_batch(backend, ["word " * 500], input_type=input_type)

        # backend.received holds the prefixed text, i.e. the actual provider payload.
        assert count_tokens(backend.received[0]) <= 50

    async def test_page_name_plus_content_fits(self):
        """#4165: a knowledge page embeds `f"{name} {content}"`. Content sized to the
        cap plus the name used to arrive exactly one token over."""
        backend = _FakeBackend()
        content = "consolidated " * 9000
        while count_tokens(content) > 8192:
            content = content[: -len("consolidated ")]
        assert count_tokens(content) == 8192

        with _patch_cap(8192):
            await embedding_utils.generate_embeddings_batch(backend, [f"User working preferences {content}"])

        assert count_tokens(backend.received[0]) <= 8192


class _RejectsOversizeBackend(_FakeBackend):
    """A provider that answers a too-large input with a permanent 4xx.

    Its own tokenizer counts more than ours, so a text cut to exactly the cap
    still arrives over its limit (#4331). It accepts anything at or below
    ``accepts_tokens``.
    """

    def __init__(self, accepts_tokens: int, status_code: int = 400) -> None:
        super().__init__()
        self.accepts_tokens = accepts_tokens
        self.status_code = status_code
        self.calls: list[list[str]] = []

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if any(count_tokens(text) > self.accepts_tokens for text in texts):
            error = Exception("The parameter is invalid. Please check again.")
            error.status_code = self.status_code
            raise error
        return await super().encode_documents(texts)


class TestOversizeRejectionRetry:
    @pytest.mark.asyncio
    async def test_retries_once_at_half_the_budget(self, caplog):
        # Accepts half the cap, refuses the cap: the shape of a provider whose
        # tokenizer runs ahead of ours.
        backend = _RejectsOversizeBackend(accepts_tokens=25)
        long_text = "word " * 500

        with _patch_cap(50), caplog.at_level(logging.WARNING):
            embeddings = await embedding_utils.generate_embeddings_batch(backend, [long_text])

        assert len(embeddings) == 1
        assert len(backend.calls) == 2, "expected exactly one retry"
        assert count_tokens(backend.calls[0][0]) > 25
        assert count_tokens(backend.calls[1][0]) <= 25
        assert any("retrying once at" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_does_not_retry_an_input_the_cap_did_not_touch(self):
        backend = _RejectsOversizeBackend(accepts_tokens=0)

        with _patch_cap(50), pytest.raises(Exception, match="Failed to generate batch embeddings"):
            await embedding_utils.generate_embeddings_batch(backend, ["short text"])

        assert len(backend.calls) == 1, "nothing was truncated, so there is nothing to shrink"

    @pytest.mark.asyncio
    async def test_does_not_retry_a_rejection_that_is_not_about_size(self):
        backend = _RejectsOversizeBackend(accepts_tokens=25, status_code=401)
        long_text = "word " * 500

        with _patch_cap(50), pytest.raises(Exception, match="Failed to generate batch embeddings"):
            await embedding_utils.generate_embeddings_batch(backend, [long_text])

        assert len(backend.calls) == 1, "an auth failure is not fixed by a smaller input"

    @pytest.mark.asyncio
    async def test_keeps_the_provider_error_as_the_cause(self):
        backend = _RejectsOversizeBackend(accepts_tokens=0, status_code=401)

        with _patch_cap(50), pytest.raises(Exception) as excinfo:
            await embedding_utils.generate_embeddings_batch(backend, ["short text"])

        assert excinfo.value.__cause__ is not None
        assert getattr(excinfo.value.__cause__, "status_code", None) == 401
