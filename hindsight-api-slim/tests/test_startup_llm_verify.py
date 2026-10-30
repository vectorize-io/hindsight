"""Startup LLM verification is bounded and fail-open (#6136).

Deterministic: fake providers stand in for a busy, healthy or misconfigured backend.
"""

import asyncio
import time

import pytest

import hindsight_api.engine.memory_engine as memory_engine
from hindsight_api import MemoryEngine
from hindsight_api.config import LLM_STRATEGY_FAILOVER, LLMStrategyConfig, clear_config_cache
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.task_backend import SyncTaskBackend
from tests.conftest import _teardown_memory_engine


class _FakeLLM:
    def __init__(self, delay: float = 0.0, error: Exception | None = None, model: str = "m"):
        self.provider = "fake"
        self.model = model
        self.delay = delay
        self.error = error
        self.verified = False

    async def verify_connection(self) -> None:
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        self.verified = True


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


@pytest.fixture(autouse=True)
def _short_probe_timeout(monkeypatch):
    monkeypatch.setattr(memory_engine, "_LLM_PROBE_TIMEOUT_SECONDS", 0.1)


@pytest.mark.asyncio
async def test_busy_backend_does_not_stall_startup(caplog):
    busy, busy2, ok, broken = _FakeLLM(delay=30), _FakeLLM(delay=30), _FakeLLM(), _FakeLLM(error=RuntimeError("401"))

    start = time.monotonic()
    await memory_engine.verify_llm_connections(
        [("default", busy), ("retain", busy2), ("reflect", ok), ("consolidation", broken)]  # type: ignore[list-item]
    )

    # Bounded by one probe timeout, not by the backend, and not summed across configs.
    assert time.monotonic() - start < 1
    assert ok.verified
    warnings = _warnings(caplog)
    assert sum("no answer within" in w for w in warnings) == 2
    assert any("'consolidation'" in w and "401" in w for w in warnings)


@pytest.mark.asyncio
async def test_slow_failover_member_does_not_hide_healthy_primary(caplog):
    primary, slow_backup = _FakeLLM(model="primary"), _FakeLLM(delay=30, model="backup")
    chain = MultiLLMProvider([primary, slow_backup], LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))  # type: ignore[list-item]

    await memory_engine.verify_llm_connections([("default", chain)])

    assert primary.verified
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "fake/backup" in warnings[0] and "no answer within" in warnings[0]


@pytest.mark.asyncio
async def test_engine_starts_while_llm_backend_is_busy(
    pg0_db_url, embeddings, cross_encoder, query_analyzer, monkeypatch, caplog
):
    """The real startup path: a probe stuck behind a busy backend used to hold
    initialize() for the whole model-init budget and then fail it."""
    monkeypatch.setenv("HINDSIGHT_API_MODEL_INIT_TIMEOUT", "5")
    clear_config_cache()

    async def busy(self):
        await asyncio.sleep(60)

    monkeypatch.setattr(LLMConfig, "verify_connection", busy)
    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=False,
    )
    try:
        await mem.initialize()
        assert any("no answer within" in w for w in _warnings(caplog))
    finally:
        await _teardown_memory_engine(mem)
        clear_config_cache()
