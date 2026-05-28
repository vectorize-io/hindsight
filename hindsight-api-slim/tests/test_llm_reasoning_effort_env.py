"""Tests for configuring LLM reasoning effort from environment variables."""

from hindsight_api.engine.cross_encoder import CrossEncoderModel
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.llm_wrapper import LLMProvider


class DummyEmbeddings(Embeddings):
    @property
    def provider_name(self) -> str:
        return "dummy"

    @property
    def dimension(self) -> int:
        return 1

    async def initialize(self) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class DummyCrossEncoder(CrossEncoderModel):
    @property
    def provider_name(self) -> str:
        return "dummy"

    async def initialize(self) -> None:
        pass

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [0.0] * len(pairs)


def _clear_config_cache() -> None:
    from hindsight_api.config import clear_config_cache

    clear_config_cache()


def test_llm_provider_from_env_uses_reasoning_effort(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_REASONING_EFFORT", "xhigh")
    _clear_config_cache()

    provider = LLMProvider.from_env()

    assert provider.reasoning_effort == "xhigh"
    assert provider._provider_impl.reasoning_effort == "xhigh"


def test_memory_engine_llm_configs_use_reasoning_effort_from_env(monkeypatch):
    from hindsight_api.engine.memory_engine import MemoryEngine

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_REASONING_EFFORT", "xhigh")
    _clear_config_cache()

    memory = MemoryEngine(
        db_url="postgresql://localhost/hindsight_test",
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=DummyEmbeddings(),
        cross_encoder=DummyCrossEncoder(),
        run_migrations=False,
        skip_llm_verification=True,
    )

    assert memory._llm_config.reasoning_effort == "xhigh"
    assert memory._retain_llm_config.reasoning_effort == "xhigh"
    assert memory._reflect_llm_config.reasoning_effort == "xhigh"
    assert memory._consolidation_llm_config.reasoning_effort == "xhigh"
    _clear_config_cache()
