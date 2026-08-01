"""Wiring tests for the OrcaRouter provider.

OrcaRouter is an OpenAI-compatible routing gateway, so it rides the shared
``OpenAICompatibleLLM`` path like OpenRouter and Requesty. These tests pin the
things a gateway provider has to get right: the default base URL, the API-key
requirement, the namespaced model id, and the embeddings factory branch.
"""

import os

import pytest

from hindsight_api.config import PROVIDER_DEFAULT_MODELS
from hindsight_api.engine.llm_wrapper import LLMProvider, create_llm_provider
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"


@pytest.fixture
def embeddings_env():
    """Save/restore the env vars the embeddings factory reads."""
    from hindsight_api.config import clear_config_cache

    keys = [
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_ORCAROUTER_API_KEY",
        "HINDSIGHT_API_EMBEDDINGS_ORCAROUTER_MODEL",
        "HINDSIGHT_API_ORCAROUTER_API_KEY",
        "HINDSIGHT_API_LLM_API_KEY",
        "HINDSIGHT_API_LLM_PROVIDER",
    ]
    original = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    clear_config_cache()

    yield

    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    clear_config_cache()


def test_default_base_url():
    """An empty base_url falls back to the OrcaRouter gateway endpoint."""
    llm = OpenAICompatibleLLM(provider="orcarouter", api_key="sk-orca-test", base_url="", model="openai/gpt-4o-mini")
    assert llm.base_url == ORCAROUTER_BASE_URL
    assert str(llm._client.base_url).rstrip("/") == ORCAROUTER_BASE_URL


def test_explicit_base_url_wins():
    """A self-hosted or proxied gateway URL is used verbatim."""
    llm = OpenAICompatibleLLM(
        provider="orcarouter",
        api_key="sk-orca-test",
        base_url="https://gateway.internal/v1",
        model="openai/gpt-4o-mini",
    )
    assert llm.base_url == "https://gateway.internal/v1"


def test_api_key_required():
    """OrcaRouter is a cloud gateway, so a missing key fails fast."""
    with pytest.raises(ValueError, match="API key is required for orcarouter"):
        OpenAICompatibleLLM(provider="orcarouter", api_key="", base_url="", model="openai/gpt-4o-mini")


def test_namespaced_model_id_is_preserved():
    """Model ids keep their ``vendor/model`` namespace; the gateway routes on it."""
    llm = OpenAICompatibleLLM(
        provider="orcarouter", api_key="sk-orca-test", base_url="", model="anthropic/claude-opus-4.8"
    )
    assert llm.model == "anthropic/claude-opus-4.8"


def test_llm_provider_accepts_orcarouter():
    """The unified LLMProvider validates the id and resolves the same base URL."""
    provider = LLMProvider(provider="orcarouter", api_key="sk-orca-test", base_url="", model="openai/gpt-4o-mini")
    assert provider.base_url == ORCAROUTER_BASE_URL


def test_create_llm_provider_routes_to_openai_compatible():
    """The factory dispatches OrcaRouter to the OpenAI-compatible implementation."""
    llm = create_llm_provider(
        provider="orcarouter",
        api_key="sk-orca-test",
        base_url="",
        model="openai/gpt-4o-mini",
        reasoning_effort="low",
    )
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.base_url == ORCAROUTER_BASE_URL


def test_default_model_registered():
    """PROVIDER_DEFAULT_MODELS must carry an entry, otherwise the model fallback is wrong."""
    assert PROVIDER_DEFAULT_MODELS["orcarouter"] == "openai/gpt-4o-mini"


def test_embeddings_factory_uses_gateway_base_url(embeddings_env):
    """``HINDSIGHT_API_EMBEDDINGS_PROVIDER=orcarouter`` builds an OpenAI-compatible embedder."""
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "orcarouter"
    os.environ["HINDSIGHT_API_EMBEDDINGS_ORCAROUTER_API_KEY"] = "sk-orca-test"

    embedder = create_embeddings_from_env()

    assert isinstance(embedder, OpenAIEmbeddings)
    assert embedder.base_url == ORCAROUTER_BASE_URL
    assert embedder.model == "openai/text-embedding-3-small"


def test_embeddings_key_falls_back_to_llm_key(embeddings_env):
    """A single HINDSIGHT_API_LLM_API_KEY covers both chat and embeddings, as for Requesty."""
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "orcarouter"
    os.environ["HINDSIGHT_API_LLM_API_KEY"] = "sk-orca-shared"

    assert create_embeddings_from_env().api_key == "sk-orca-shared"


def test_embeddings_factory_requires_a_key(embeddings_env):
    """Without any usable key the factory raises instead of sending anonymous requests."""
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    os.environ["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "orcarouter"

    with pytest.raises(ValueError, match="ORCAROUTER_API_KEY"):
        create_embeddings_from_env()
