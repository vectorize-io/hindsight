"""The OpenAI embeddings startup probe retries like every other provider's (#6709).

A model outside MODEL_DIMENSIONS makes initialize() send one test embedding to learn
the dimension. A 429 there used to exit the API at startup.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.embeddings import OpenAIEmbeddings, RetryPolicy, _create_embeddings


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _ok_response() -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 7)])


async def _init_with(create: AsyncMock) -> OpenAIEmbeddings:
    emb = OpenAIEmbeddings(
        api_key="k",
        model="custom-gateway-model",
        retry_policy=RetryPolicy(max_retries=3, initial_backoff=0.0, max_backoff=0.0, budget_seconds=10.0),
    )
    client = MagicMock()
    client.embeddings.create = create
    with patch.object(OpenAIEmbeddings, "_loop_client", return_value=client):
        await emb.initialize()
    return emb


@pytest.mark.asyncio
async def test_probe_rides_out_429():
    create = AsyncMock(side_effect=[_StatusError(429), _StatusError(429), _ok_response()])
    emb = await _init_with(create)
    assert emb.dimension == 7
    assert create.await_count == 3


@pytest.mark.asyncio
async def test_probe_fails_fast_on_auth_error():
    create = AsyncMock(side_effect=_StatusError(401))
    with pytest.raises(_StatusError):
        await _init_with(create)
    assert create.await_count == 1


@pytest.mark.parametrize("provider", ["openai", "openrouter", "requesty", "hindsight", "openai-codex"])
def test_factory_passes_configured_policy(provider):
    config = MagicMock()
    config.embeddings_provider = provider
    config.embeddings_max_retries = 9
    config.embeddings_retry_budget = 42.0
    config.embeddings_openai_dimensions = None
    with patch("hindsight_api.engine.providers.codex_auth.CodexAuthManager.from_file"):
        emb = _create_embeddings(config)
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.retry_policy.max_retries == 9
    assert emb.retry_policy.budget_seconds == 42.0
