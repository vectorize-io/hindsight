"""
Tests for the internal recall configuration knobs used during mental model
refresh: recall_include_chunks, recall_max_tokens, recall_chunks_max_tokens.

These are exposed both as hierarchical config fields (env → tenant → bank)
and as overrides on a mental model's `trigger` JSONB field.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.reflect.tools import tool_recall
from hindsight_api.engine.response_models import RecallResult as RecallResultModel
from hindsight_api.models import RequestContext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured,overrides,tool_limits,expected",
    [
        ({}, {}, {}, (2048, 1000)),
        ({"recall_max_tokens": 8192, "recall_chunks_max_tokens": 3072}, {}, {}, (8192, 3072)),
        ({"recall_max_tokens": 512, "recall_chunks_max_tokens": 0}, {}, {}, (512, 0)),
        (
            {"recall_max_tokens": 8192, "recall_chunks_max_tokens": 3072},
            {"recall_max_tokens_override": 512, "recall_chunks_max_tokens_override": 0},
            {},
            (512, 0),
        ),
        (
            {"recall_max_tokens": 512, "recall_chunks_max_tokens": 0},
            {},
            {"max_tokens": "None", "max_chunk_tokens": "null"},
            (512, 0),
        ),
        (
            {"recall_max_tokens": 512, "recall_chunks_max_tokens": 0},
            {},
            {"max_tokens": 0, "max_chunk_tokens": 0},
            (512, 0),
        ),
        ({"recall_max_tokens": 512, "recall_chunks_max_tokens": 0}, {}, {"max_tokens": "4096"}, (4096, 0)),
        ({"recall_max_tokens": 512, "recall_chunks_max_tokens": 0}, {}, {"max_chunk_tokens": 2048}, (512, 2048)),
        (
            {"recall_max_tokens": 512, "recall_chunks_max_tokens": 0},
            {},
            {"max_tokens": 100, "max_chunk_tokens": 100},
            (1000, 1000),
        ),
    ],
)
async def test_reflect_dispatch_uses_resolved_recall_defaults(
    configured: dict[str, int],
    overrides: dict[str, int],
    tool_limits: dict[str, int | str],
    expected: tuple[int, int],
) -> None:
    """Exercise engine -> agent -> real recall tool, not just closure defaults (#4239)."""
    from hindsight_api.engine.memory_engine import MemoryEngine
    from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, MemoryFact

    # Stub storage/LLM transport, but keep the complete reflect dispatch path. The
    # prior trigger tests stopped before the agent replaced these settings.
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    engine._resolve_fuzzy_tag_groups = AsyncMock(return_value=None)
    engine._get_backend = AsyncMock()
    engine.get_bank_profile = AsyncMock(return_value={"name": "Test", "mission": "Testing"})
    engine.get_bank_freshness = AsyncMock(return_value={})
    engine.list_directives = AsyncMock(return_value=SimpleNamespace(items=[]))
    engine._config_resolver = MagicMock()
    engine._config_resolver.get_bank_config = AsyncMock(return_value=configured)
    engine._config_resolver.resolve_full_config = AsyncMock(return_value=SimpleNamespace(llm_output_language=None))
    engine.recall_async = AsyncMock(
        return_value=RecallResultModel(results=[MemoryFact(id="mem-1", text="The deploy is ready.", fact_type="world")])
    )
    llm = MagicMock()
    llm._provider_impl = None
    llm.call_with_tools = AsyncMock(
        side_effect=[
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "deploy", **tool_limits})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="done", arguments={"answer": "Ready.", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]
    )
    engine._reflect_llm_config = MagicMock(provider="test")
    engine._reflect_llm_config.with_config.return_value = llm

    result = await engine.reflect_async(
        bank_id="bank-1",
        query="Deploy status?",
        request_context=RequestContext(internal=True),
        fact_types=["world"],
        exclude_mental_models=True,
        _skip_span=True,
        **overrides,
    )

    assert result.text == "Ready."
    engine.recall_async.assert_awaited_once()
    actual = engine.recall_async.await_args.kwargs
    assert (actual["max_tokens"], actual["max_chunk_tokens"]) == expected
    assert actual["request_context"].internal is True
    assert result.based_on["world"][0].id == "mem-1"


def test_recall_schema_defaults_are_scoped_to_one_reflect() -> None:
    from hindsight_api.engine.reflect.tools_schema import TOOL_RECALL, get_reflect_tools

    original = TOOL_RECALL["function"]["parameters"]["properties"]["max_tokens"]["description"]
    custom = get_reflect_tools(recall_max_tokens=512, recall_chunks_max_tokens=0)
    regular = get_reflect_tools()
    custom_recall = next(t for t in custom if t["function"]["name"] == "recall")
    regular_recall = next(t for t in regular if t["function"]["name"] == "recall")
    custom_properties = custom_recall["function"]["parameters"]["properties"]
    assert "default 512" in custom_properties["max_tokens"]["description"]
    assert "default 0" in custom_properties["max_chunk_tokens"]["description"]
    assert regular_recall["function"]["parameters"]["properties"]["max_tokens"]["description"] == original
    assert TOOL_RECALL["function"]["parameters"]["properties"]["max_tokens"]["description"] == original


def test_recall_trace_summary_uses_the_dispatch_defaults() -> None:
    from hindsight_api.engine.reflect.agent import _summarize_input

    assert (
        _summarize_input("recall", {"query": "deploy"}, recall_max_tokens=512, recall_chunks_max_tokens=0)
        == "(query='deploy', max_tokens=512, max_chunk_tokens=0)"
    )


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
