"""
Tests for the internal recall configuration knobs used during mental model
refresh: recall_include_chunks, recall_max_tokens, recall_chunks_max_tokens.

These are exposed both as hierarchical config fields (env → tenant → bank)
and as overrides on a mental model's `trigger` JSONB field.
"""

import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.reflect.tools import tool_recall
from hindsight_api.engine.response_models import RecallResult as RecallResultModel
from hindsight_api.models import RequestContext


def _make_mock_engine():
    engine = MagicMock()
    engine.recall_async = AsyncMock(return_value=RecallResultModel(results=[], entities={}, chunks={}))
    return engine


@pytest.fixture
def mock_request_context():
    # internal=True bypasses the tenant extension, letting these unit tests
    # exercise engine methods without standing up auth.
    return RequestContext(internal=True)


class TestToolRecallIncludeChunks:
    """tool_recall must honor the include_chunks parameter (was hardcoded True)."""

    @pytest.mark.asyncio
    async def test_default_includes_chunks(self, mock_request_context):
        engine = _make_mock_engine()

        await tool_recall(engine, "bank-1", "q", mock_request_context)

        kwargs = engine.recall_async.call_args.kwargs
        assert kwargs["include_chunks"] is True

    @pytest.mark.asyncio
    async def test_include_chunks_false_propagates(self, mock_request_context):
        engine = _make_mock_engine()

        await tool_recall(engine, "bank-1", "q", mock_request_context, include_chunks=False)

        kwargs = engine.recall_async.call_args.kwargs
        assert kwargs["include_chunks"] is False

    @pytest.mark.asyncio
    async def test_max_chunk_tokens_propagates(self, mock_request_context):
        engine = _make_mock_engine()

        await tool_recall(
            engine, "bank-1", "q", mock_request_context, max_chunk_tokens=2500, max_tokens=512
        )

        kwargs = engine.recall_async.call_args.kwargs
        assert kwargs["max_chunk_tokens"] == 2500
        assert kwargs["max_tokens"] == 512


class TestRecallConfigFields:
    """Hierarchical config fields for internal recall."""

    def test_fields_exist_on_dataclass(self):
        from hindsight_api.config import HindsightConfig

        names = {f.name for f in dataclasses.fields(HindsightConfig)}
        assert "recall_include_chunks" in names
        assert "recall_max_tokens" in names
        assert "recall_chunks_max_tokens" in names

    def test_fields_are_configurable(self):
        from hindsight_api.config import HindsightConfig

        configurable = HindsightConfig.get_configurable_fields()
        assert "recall_include_chunks" in configurable
        assert "recall_max_tokens" in configurable
        assert "recall_chunks_max_tokens" in configurable

    def test_default_values(self):
        from hindsight_api.config import (
            DEFAULT_RECALL_CHUNKS_MAX_TOKENS,
            DEFAULT_RECALL_INCLUDE_CHUNKS,
            DEFAULT_RECALL_MAX_TOKENS,
        )

        assert DEFAULT_RECALL_INCLUDE_CHUNKS is True
        assert DEFAULT_RECALL_MAX_TOKENS == 2048
        assert DEFAULT_RECALL_CHUNKS_MAX_TOKENS == 1000

    def test_env_var_constants(self):
        from hindsight_api.config import (
            ENV_RECALL_CHUNKS_MAX_TOKENS,
            ENV_RECALL_INCLUDE_CHUNKS,
            ENV_RECALL_MAX_TOKENS,
        )

        assert ENV_RECALL_INCLUDE_CHUNKS == "HINDSIGHT_API_RECALL_INCLUDE_CHUNKS"
        assert ENV_RECALL_MAX_TOKENS == "HINDSIGHT_API_RECALL_MAX_TOKENS"
        assert ENV_RECALL_CHUNKS_MAX_TOKENS == "HINDSIGHT_API_RECALL_CHUNKS_MAX_TOKENS"

    @patch.dict(
        "os.environ",
        {
            "HINDSIGHT_API_RECALL_INCLUDE_CHUNKS": "false",
            "HINDSIGHT_API_RECALL_MAX_TOKENS": "777",
            "HINDSIGHT_API_RECALL_CHUNKS_MAX_TOKENS": "333",
        },
    )
    def test_from_env_reads_overrides(self):
        from hindsight_api.config import HindsightConfig

        config = HindsightConfig.from_env()
        assert config.recall_include_chunks is False
        assert config.recall_max_tokens == 777
        assert config.recall_chunks_max_tokens == 333


class TestMentalModelTriggerRecallFields:
    """MentalModelTrigger Pydantic model accepts the new override fields."""

    def test_trigger_accepts_new_fields(self):
        from hindsight_api.api.http import MentalModelTrigger

        trigger = MentalModelTrigger(
            include_chunks=False,
            recall_max_tokens=512,
            recall_chunks_max_tokens=0,
        )
        assert trigger.include_chunks is False
        assert trigger.recall_max_tokens == 512
        assert trigger.recall_chunks_max_tokens == 0

    def test_trigger_defaults_are_none(self):
        from hindsight_api.api.http import MentalModelTrigger

        trigger = MentalModelTrigger()
        assert trigger.include_chunks is None
        assert trigger.recall_max_tokens is None
        assert trigger.recall_chunks_max_tokens is None


class TestRefreshTriggerWiring:
    """Verify mental-model refresh forwards trigger overrides into reflect_async kwargs."""

    @pytest.mark.asyncio
    async def test_trigger_overrides_passed_to_reflect_async(self, mock_request_context):
        from hindsight_api.engine.memory_engine import MemoryEngine
        from hindsight_api.engine.response_models import ReflectResult

        engine = MemoryEngine.__new__(MemoryEngine)

        async def fake_get_mental_model(bank_id, mental_model_id, request_context):
            return {
                "id": mental_model_id,
                "source_query": "What do we know?",
                "tags": [],
                "trigger": {
                    "include_chunks": False,
                    "recall_max_tokens": 512,
                    "recall_chunks_max_tokens": 0,
                    "fact_types": ["world"],
                },
            }

        captured = {}

        async def fake_reflect_async(**kwargs):
            captured.update(kwargs)
            return ReflectResult(text="ok", based_on={})

        async def fake_update_mental_model(*args, **kwargs):
            return None

        engine.get_mental_model = fake_get_mental_model
        engine.reflect_async = fake_reflect_async
        engine.update_mental_model = fake_update_mental_model
        engine._operation_validator = None
        engine._tenant_extension = None

        await engine.refresh_mental_model(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=mock_request_context,
        )

        assert captured["recall_include_chunks"] is False
        assert captured["recall_max_tokens_override"] == 512
        assert captured["recall_chunks_max_tokens_override"] == 0
        assert captured["fact_types"] == ["world"]

    @pytest.mark.asyncio
    async def test_missing_trigger_fields_pass_none(self, mock_request_context):
        from hindsight_api.engine.memory_engine import MemoryEngine
        from hindsight_api.engine.response_models import ReflectResult

        engine = MemoryEngine.__new__(MemoryEngine)

        async def fake_get_mental_model(bank_id, mental_model_id, request_context):
            return {"id": mental_model_id, "source_query": "q", "tags": [], "trigger": {}}

        captured = {}

        async def fake_reflect_async(**kwargs):
            captured.update(kwargs)
            return ReflectResult(text="ok", based_on={})

        async def fake_update_mental_model(*args, **kwargs):
            return None

        engine.get_mental_model = fake_get_mental_model
        engine.reflect_async = fake_reflect_async
        engine.update_mental_model = fake_update_mental_model
        engine._operation_validator = None
        engine._tenant_extension = None

        await engine.refresh_mental_model(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=mock_request_context,
        )

        # When trigger fields are absent, None is forwarded so reflect_async falls back to bank/global config.
        assert captured["recall_include_chunks"] is None
        assert captured["recall_max_tokens_override"] is None
        assert captured["recall_chunks_max_tokens_override"] is None
