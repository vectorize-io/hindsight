"""
Tests for HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE config wiring.

Regression test for issue #1142: `OpenAIEmbeddings` hardcoded `batch_size=100` is
incompatible with OpenAI-compatible providers that enforce stricter per-request
limits (e.g. DashScope / Aliyun Tongyi cap at 10). Users must be able to override
the batch size via env var so `encode()` splits into smaller chunks.
"""

import json
import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Save/restore env vars touched by these tests."""
    from hindsight_api.config import clear_config_cache

    env_vars_to_save = [
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS",
        "HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY",
        "HINDSIGHT_API_LLM_API_KEY",
        "HINDSIGHT_API_LLM_PROVIDER",
    ]

    original_values = {key: os.environ.get(key) for key in env_vars_to_save}

    clear_config_cache()

    yield

    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_default_openai_batch_size_is_100():
    """Default batch size is 100 when env var unset (preserves legacy behavior)."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE", None)

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_batch_size == 100


def test_openai_batch_size_env_var_is_read():
    """HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE overrides the default."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "10"

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_batch_size == 10


def test_openai_dimensions_env_var_is_read():
    """HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS requests reduced OpenAI output dims."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS"] = "384"

    config = HindsightConfig.from_env()
    assert config.embeddings_openai_dimensions == 384


def test_openai_embeddings_provider_uses_configured_batch_size():
    """create_embeddings_from_env() propagates config to OpenAIEmbeddings for 'openai' provider."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openai"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"] = "sk-test"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "10"

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.batch_size == 10


def test_openrouter_provider_uses_configured_batch_size():
    """'openrouter' provider also honors the shared batch-size config (both paths use OpenAIEmbeddings)."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openrouter"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY"] = "sk-or-test"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "8"

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.batch_size == 8


def test_openai_codex_provider_uses_codex_oauth_token_and_configured_batch_size(tmp_path, monkeypatch):
    """'openai-codex' embeddings reuse Codex OAuth auth without a separate API key."""
    from hindsight_api.engine.embeddings import CodexOAuthEmbeddings, create_embeddings_from_env

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "codex-oauth-token-test",
                    "account_id": "acct-test",
                },
            }
        )
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "openai-codex"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"] = "text-embedding-3-small"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "7"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS"] = "384"

    embeddings = create_embeddings_from_env()
    assert isinstance(embeddings, CodexOAuthEmbeddings)
    assert embeddings.provider_name == "openai-codex"
    assert embeddings.model == "text-embedding-3-small"
    assert embeddings.base_url == "https://api.openai.com/v1"
    assert embeddings.api_key == "codex-oauth-token-test"
    assert embeddings.batch_size == 7
    assert embeddings.dimensions == 384


def test_zero_batch_size_is_rejected():
    """Zero would cause `range(0, N, 0)` to crash at runtime — fail fast at config load."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "0"

    with pytest.raises(ValueError, match="must be >= 1"):
        HindsightConfig.from_env()


def test_negative_batch_size_is_rejected():
    """Negative values would silently skip batching — reject at config load."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "-5"

    with pytest.raises(ValueError, match="must be >= 1"):
        HindsightConfig.from_env()


def test_non_numeric_batch_size_is_rejected():
    """Non-integer strings are rejected with a clear error pointing at the env var name."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"] = "not-a-number"

    with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE"):
        HindsightConfig.from_env()


def test_openai_encode_splits_on_configured_batch_size(monkeypatch):
    """encode() sends multiple upstream requests when len(texts) > batch_size."""
    from types import SimpleNamespace

    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small", batch_size=10)

    calls: list[int] = []

    def fake_create(*, model, input):
        calls.append(len(input))
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.0] * 1536) for i in range(len(input))])

    emb._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    emb._dimension = 1536

    vectors = emb.encode(["x"] * 25)

    assert calls == [10, 10, 5]
    assert len(vectors) == 25


def test_openai_encode_passes_configured_dimensions():
    """OpenAI embeddings requests include the optional dimensions parameter when configured."""
    from types import SimpleNamespace

    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(
        api_key="sk-test",
        model="text-embedding-3-small",
        batch_size=10,
        dimensions=384,
    )

    calls: list[int | None] = []

    def fake_create(*, model, input, dimensions=None):
        calls.append(dimensions)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[0.0] * 384) for i in range(len(input))])

    emb._client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    emb._dimension = 384

    vectors = emb.encode(["x"] * 2)

    assert calls == [384]
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
