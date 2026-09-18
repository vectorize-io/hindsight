"""
Tests for the internal recall configuration knobs used during mental model
refresh: recall_include_chunks, recall_max_tokens, recall_chunks_max_tokens.

These are exposed both as hierarchical config fields (env → tenant → bank)
and as overrides on a mental model's `trigger` JSONB field.
"""

import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.reflect.tokenization import count_prompt_tokens
from hindsight_api.engine.reflect.tools import tool_recall
from hindsight_api.engine.response_models import ChunkInfo
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

        await tool_recall(engine, "bank-1", "q", mock_request_context, max_chunk_tokens=2500, max_tokens=512)

        kwargs = engine.recall_async.call_args.kwargs
        assert kwargs["max_chunk_tokens"] == 2500
        assert kwargs["max_tokens"] == 512


class TestToolRecallChunkBudget:
    """#4495: tool_recall's "chunks" payload must be bounded by max_chunk_tokens
    on the RENDERED payload, not just the raw text the recall engine already
    budgets -- a store-answered recall never goes through that budget loop at
    all, so the tool must be self-bounding regardless of the recall path."""

    @staticmethod
    def _engine_with_chunks(chunks: dict[str, ChunkInfo]):
        engine = MagicMock()
        engine.recall_async = AsyncMock(return_value=RecallResultModel(results=[], entities={}, chunks=chunks))
        return engine

    @pytest.mark.asyncio
    async def test_oversized_chunk_payload_is_bounded(self, mock_request_context):
        """A recall path that ignores the budget (e.g. an unbounded store
        response) must still come back under max_chunk_tokens once rendered."""
        chunks = {f"c{i}": ChunkInfo(chunk_text="x" * 4000, chunk_index=i) for i in range(20)}
        engine = self._engine_with_chunks(chunks)

        result = await tool_recall(engine, "bank-1", "q", mock_request_context, max_chunk_tokens=1000)

        rendered = json.dumps(result["chunks"], indent=2, default=str, ensure_ascii=False)
        assert result["chunks"], "budget must not drop every chunk"
        assert count_prompt_tokens(rendered) <= 1000

    @pytest.mark.asyncio
    async def test_highest_relevance_chunks_are_kept(self, mock_request_context):
        """Chunks are walked in the order the recall result gives them
        (relevance order); dropped chunks must be a suffix, not an arbitrary
        subset."""
        chunks = {f"c{i}": ChunkInfo(chunk_text="x" * 4000, chunk_index=i) for i in range(20)}
        engine = self._engine_with_chunks(chunks)

        result = await tool_recall(engine, "bank-1", "q", mock_request_context, max_chunk_tokens=1000)

        assert "c0" in result["chunks"]
        assert "c19" not in result["chunks"]
        kept_indices = [int(k[1:]) for k in result["chunks"]]
        assert kept_indices == sorted(kept_indices), "surviving chunks must be a prefix of the original order"

    @pytest.mark.asyncio
    async def test_payload_within_budget_is_returned_untouched(self, mock_request_context):
        """The common case (chunks already fit) must be unaffected by the cap."""
        chunks = {f"c{i}": ChunkInfo(chunk_text="y" * 10, chunk_index=i) for i in range(3)}
        engine = self._engine_with_chunks(chunks)

        result = await tool_recall(engine, "bank-1", "q", mock_request_context, max_chunk_tokens=1000)

        assert set(result["chunks"]) == {"c0", "c1", "c2"}


class TestRecallConfigFields:
    """Hierarchical config fields for internal recall."""

    def test_fields_exist_on_dataclass(self):
        from hindsight_api.config import DEFAULT_BM25_MAX_QUERY_TERMS, HindsightConfig

        names = {f.name for f in dataclasses.fields(HindsightConfig)}
        assert "recall_include_chunks" in names
        assert "recall_max_tokens" in names
        assert "recall_chunks_max_tokens" in names
        assert "bm25_max_query_terms" in names
        assert HindsightConfig.__dataclass_fields__["bm25_max_query_terms"].default == DEFAULT_BM25_MAX_QUERY_TERMS

    def test_fields_are_configurable(self):
        from hindsight_api.config import HindsightConfig

        configurable = HindsightConfig.get_configurable_fields()
        assert "recall_include_chunks" in configurable
        assert "recall_max_tokens" in configurable
        assert "recall_chunks_max_tokens" in configurable

    def test_default_values(self):
        from hindsight_api.config import (
            DEFAULT_BM25_MAX_QUERY_TERMS,
            DEFAULT_RECALL_CHUNKS_MAX_TOKENS,
            DEFAULT_RECALL_INCLUDE_CHUNKS,
            DEFAULT_RECALL_MAX_TOKENS,
        )

        assert DEFAULT_RECALL_INCLUDE_CHUNKS is True
        assert DEFAULT_RECALL_MAX_TOKENS == 2048
        assert DEFAULT_RECALL_CHUNKS_MAX_TOKENS == 1000
        assert DEFAULT_BM25_MAX_QUERY_TERMS == 16

    def test_env_var_constants(self):
        from hindsight_api.config import (
            ENV_BM25_MAX_QUERY_TERMS,
            ENV_RECALL_CHUNKS_MAX_TOKENS,
            ENV_RECALL_INCLUDE_CHUNKS,
            ENV_RECALL_MAX_TOKENS,
        )

        assert ENV_RECALL_INCLUDE_CHUNKS == "HINDSIGHT_API_RECALL_INCLUDE_CHUNKS"
        assert ENV_RECALL_MAX_TOKENS == "HINDSIGHT_API_RECALL_MAX_TOKENS"
        assert ENV_RECALL_CHUNKS_MAX_TOKENS == "HINDSIGHT_API_RECALL_CHUNKS_MAX_TOKENS"
        assert ENV_BM25_MAX_QUERY_TERMS == "HINDSIGHT_API_BM25_MAX_QUERY_TERMS"

    @patch.dict(
        "os.environ",
        {
            "HINDSIGHT_API_RECALL_INCLUDE_CHUNKS": "false",
            "HINDSIGHT_API_RECALL_MAX_TOKENS": "777",
            "HINDSIGHT_API_RECALL_CHUNKS_MAX_TOKENS": "333",
            "HINDSIGHT_API_BM25_MAX_QUERY_TERMS": "24",
        },
    )
    def test_from_env_reads_overrides(self):
        from hindsight_api.config import HindsightConfig

        config = HindsightConfig.from_env()
        assert config.recall_include_chunks is False
        assert config.recall_max_tokens == 777
        assert config.recall_chunks_max_tokens == 333
        assert config.bm25_max_query_terms == 24

    @patch.dict("os.environ", {"HINDSIGHT_API_BM25_MAX_QUERY_TERMS": "-1"})
    def test_from_env_rejects_negative_bm25_max_query_terms(self):
        from hindsight_api.config import HindsightConfig

        with pytest.raises(ValueError, match="HINDSIGHT_API_BM25_MAX_QUERY_TERMS must be >= 0"):
            HindsightConfig.from_env()


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
        from datetime import datetime, timezone

        from hindsight_api.engine.memory_engine import MemoryEngine, _MentalModelScopeWatermark
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
        # DB-time refresh watermark — stub so this mock test doesn't reach a real
        # pool (matches the other collaborator stubs above).
        engine._mental_model_refresh_cutoff = AsyncMock(return_value=datetime(2026, 1, 1, tzinfo=timezone.utc))
        # A scope with a memory in it: the reading this returns is also what decides
        # whether the refresh has anything to reflect over (#3875), so a stub saying
        # "empty" would skip the reflect call these tests assert on.
        engine._mental_model_scope_watermark = AsyncMock(
            return_value=_MentalModelScopeWatermark(
                newest_in_scope=datetime(2025, 12, 1, tzinfo=timezone.utc), watermark=None
            )
        )

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
        from datetime import datetime, timezone

        from hindsight_api.engine.memory_engine import MemoryEngine, _MentalModelScopeWatermark
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
        # DB-time refresh watermark — stub so this mock test doesn't reach a real
        # pool (matches the other collaborator stubs above).
        engine._mental_model_refresh_cutoff = AsyncMock(return_value=datetime(2026, 1, 1, tzinfo=timezone.utc))
        # A scope with a memory in it: the reading this returns is also what decides
        # whether the refresh has anything to reflect over (#3875), so a stub saying
        # "empty" would skip the reflect call these tests assert on.
        engine._mental_model_scope_watermark = AsyncMock(
            return_value=_MentalModelScopeWatermark(
                newest_in_scope=datetime(2025, 12, 1, tzinfo=timezone.utc), watermark=None
            )
        )

        await engine.refresh_mental_model(
            bank_id="bank-1",
            mental_model_id="mm-1",
            request_context=mock_request_context,
        )

        # When trigger fields are absent, None is forwarded so reflect_async falls back to bank/global config.
        assert captured["recall_include_chunks"] is None
        assert captured["recall_max_tokens_override"] is None
        assert captured["recall_chunks_max_tokens_override"] is None
