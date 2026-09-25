"""Deterministic boundaries of the opt-in source window (issue #2551)."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import HindsightConfig, validate_retain_context_chars
from hindsight_api.config_resolver import apply_strategy
from hindsight_api.engine.llm_interface import OutputTooLongError
from hindsight_api.engine.memory_engine import _iter_raw_sub_batches
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.retain.orchestrator import _build_retain_params
from hindsight_api.engine.retain.source_context import contextual_chunks, extend_source_context
from hindsight_api.engine.retain.types import RetainContent


def config(**overrides) -> HindsightConfig:
    return replace(HindsightConfig.from_env(), **overrides)


def test_bounded_ordered_source_excludes_target_and_future() -> None:
    windows = list(contextual_chunks(["one", "two", "three", "four"], 8))
    assert [c.text for c in windows] == ["one", "two", "three", "four"]
    assert [c.previous_source for c in windows] == ["", "one", "one\n\ntwo", "o\n\nthree"]
    assert all(len(c.previous_source) <= 8 for c in windows)
    assert all(c.previous_source == "" for c in contextual_chunks(["one", "two"], 0, "seed"))


def test_prior_attachments_are_not_loaded_or_numbered() -> None:
    text = extend_source_context("", "See ⟦hs-att:012345abcdef⟧ before choosing.", 100)
    assert text == "See [attachment omitted] before choosing."


@pytest.mark.parametrize("value", [-1, 32001, True, 1.5, "100"])
def test_invalid_budgets_rejected_in_named_strategies(value) -> None:
    with pytest.raises(ValueError, match="retain_context_chars"):
        apply_strategy(config(retain_strategies={"chat": {"retain_context_chars": value}}), "chat")


@pytest.mark.parametrize("value", [0, 1, 32000])
def test_valid_budgets(value: int) -> None:
    validate_retain_context_chars(value)
    assert (
        apply_strategy(config(retain_strategies={"chat": {"retain_context_chars": value}}), "chat").retain_context_chars
        == value
    )


def test_source_window_never_becomes_replay_context() -> None:
    params, _ = _build_retain_params(
        [
            {
                "content": "target",
                "context": "speaker identity",
                "_previous_source": "prior",
                "_retain_context_chars": 99,
            }
        ],
        context_chars=100,
    )
    assert params == {"context": "speaker identity", "_retain_context_chars": 100}
    disabled, _ = _build_retain_params([params])
    assert disabled == {"context": "speaker identity"}


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 800])
async def test_extraction_preserves_source_and_item_boundaries(monkeypatch, budget: int) -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/retain_context_options.json").read_text())["A"]
    text = json.dumps([fixture["prior"], fixture["target"]])
    extract = AsyncMock(return_value=([], TokenUsage()))
    monkeypatch.setattr(fact_extraction, "_extract_facts_with_auto_split", extract)
    result = await fact_extraction.extract_facts_from_contents(
        [RetainContent(content=text), RetainContent(content="unrelated document")],
        object(),
        config(retain_chunk_size=800, retain_context_chars=budget, retain_batch_enabled=False),
    )
    calls = [c.kwargs for c in extract.call_args_list]
    assert len(calls) == 3
    assert calls[0]["previous_source"] == ""
    assert calls[1]["previous_source"] == (calls[0]["chunk"] if budget else "")
    assert calls[2]["previous_source"] == ""
    assert [c.chunk_text for c in result.chunks] == [c["chunk"] for c in calls]
    assert [c.chunk_index for c in result.chunks] == [0, 1, 2]


@pytest.mark.asyncio
async def test_retry_right_half_sees_left_source(monkeypatch) -> None:
    extract = AsyncMock(side_effect=[OutputTooLongError("split"), ([], TokenUsage()), ([], TokenUsage())])
    monkeypatch.setattr(fact_extraction, "_extract_facts_from_chunk", extract)
    await fact_extraction._extract_facts_with_auto_split(
        chunk=json.dumps([{"content": "left " * 60}, {"content": "right " * 60}]),
        chunk_index=3,
        total_chunks=5,
        event_date=None,
        context="identity",
        llm_config=object(),
        config=config(retain_context_chars=100),
        previous_source="earlier",
    )
    parent, left, right = [c.kwargs for c in extract.call_args_list]
    assert left["previous_source"] == parent["previous_source"] == "earlier"
    assert right["previous_source"] == left["chunk"][-100:]
    assert all(c.kwargs["chunk_index"] == 3 for c in extract.call_args_list)


def test_oversized_slices_keep_context_without_storing_overlap() -> None:
    text = json.dumps([{"role": "user", "content": f"turn {i} " + "words " * 20} for i in range(8)])
    slices = list(
        _iter_raw_sub_batches(
            [{"content": text, "document_id": "chat"}],
            60,
            chunk_size=200,
            max_attachments_per_chunk=8,
            context_chars=400,
        )
    )
    assert len(slices) > 1
    seen = ""
    for part in slices:
        item = part.contents[0]
        assert item["_previous_source"] == seen
        assert part.full_document_body == text
        seen = extend_source_context(seen, item["content"], 400)


@pytest.mark.parametrize("shape", ["text", "json"])
@pytest.mark.parametrize("chunks_per_slice", [1, 2, 3])
def test_internal_batching_preserves_native_source_windows(shape: str, chunks_per_slice: int) -> None:
    from hindsight_api.engine.memory_engine import count_tokens

    turns = [
        "Option 2 means Monday with Theo. " + "alpha " * 10,
        "A neutral paragraph. " + "beta " * 10,
        "I select option 2. " + "gamma " * 10,
    ]
    text = (
        turns[0] + "\n" * 300 + turns[1] + "\n\n" + turns[2]
        if shape == "text"
        else json.dumps([{"role": "user", "content": turn} for turn in turns])
    )
    native = fact_extraction.chunk_text(text, 100, structured_chunk_size=200)
    assert len(native) == 3
    parts = list(
        _iter_raw_sub_batches(
            [{"content": text, "document_id": "chat"}],
            sum(count_tokens(chunk) for chunk in native[:chunks_per_slice]),
            chunk_size=100,
            structured_chunk_size=200,
            max_attachments_per_chunk=8,
            context_chars=250,
        )
    )
    actual = []
    for part in parts:
        item = part.contents[0]
        chunks = fact_extraction.chunk_text(item["content"], 100, structured_chunk_size=200)
        assert part.chunk_count == len(chunks)
        if part.full_document_body is not None:
            assert part.full_document_body == text
        actual.extend(contextual_chunks(chunks, 250, item.get("_previous_source", "")))
    assert [chunk.text for chunk in actual] == native
    assert actual == list(contextual_chunks(native, 250))


def test_prompt_marks_supporting_source_separately() -> None:
    parts = fact_extraction.build_chunk_prompt_parts(
        config(), chunk="choose option 2", context="Mira is speaking", previous_source="Option 2: Friday / Nia"
    )
    assert "PREVIOUS SOURCE (reference context only):\nOption 2: Friday / Nia" in parts.user_message
    assert parts.user_message.endswith("Content:\nchoose option 2")
    baseline = fact_extraction.build_chunk_prompt_parts(config(), chunk="choose option 2")
    assert "PREVIOUS SOURCE" not in baseline.user_message


@pytest.mark.asyncio
async def test_batch_prompts_share_the_same_windows(monkeypatch) -> None:
    impl = SimpleNamespace(
        provider="openai",
        model="test-model",
        openai_service_tier=None,
        submit_batch=AsyncMock(side_effect=RuntimeError("captured submission")),
    )
    llm = SimpleNamespace(batch_provider_impl=AsyncMock(return_value=impl))
    fixture = json.loads((Path(__file__).parent / "fixtures/retain_context_options.json").read_text())["A"]
    text = json.dumps([fixture["prior"], fixture["target"]])
    with pytest.raises(RuntimeError, match="captured submission"):
        await fact_extraction.extract_facts_from_contents_batch_api(
            [RetainContent(content=text), RetainContent(content="unrelated")],
            llm,
            config(retain_chunk_size=800, retain_context_chars=800),
        )
    requests = impl.submit_batch.call_args.args[0]
    prompts = [r["body"]["messages"][-1]["content"] for r in requests]
    assert len(prompts) == 3
    assert "PREVIOUS SOURCE" not in prompts[0]
    assert "PREVIOUS SOURCE" in prompts[1]
    assert "Friday with Nia" in prompts[1]
    assert prompts[1].endswith("Content:\n" + json.dumps([fixture["target"]]))
    assert "PREVIOUS SOURCE" not in prompts[2]


def test_bank_template_keeps_context_window_setting() -> None:
    from hindsight_api.api.http import BankTemplateConfig

    template = BankTemplateConfig.model_validate({"retain_context_chars": 800})
    assert template.get_config_updates() == {"retain_context_chars": 800}


@pytest.mark.parametrize("value", ["-1", "32001", "not-an-integer"])
def test_invalid_environment_budget_fails_at_startup(monkeypatch, value: str) -> None:
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_CONTEXT_CHARS", value)
    with pytest.raises(ValueError):
        HindsightConfig.from_env()


@pytest.mark.asyncio
async def test_batch_resume_keeps_implicit_date_but_rejects_changed_context(monkeypatch) -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager
    from datetime import datetime, timezone

    from hindsight_api.engine import db_utils

    conn = SimpleNamespace(fetchrow=AsyncMock(return_value=None), execute=AsyncMock())

    @asynccontextmanager
    async def connection(pool: object) -> AsyncIterator[SimpleNamespace]:
        yield conn

    monkeypatch.setattr(db_utils, "acquire_with_retry", connection)
    impl = SimpleNamespace(
        provider="openai",
        model="test-model",
        openai_service_tier=None,
        batch_account_key="test-account",
        submit_batch=AsyncMock(return_value={"batch_id": "saved-batch"}),
        get_batch_status=AsyncMock(side_effect=RuntimeError("polling reached")),
    )
    llm = SimpleNamespace(batch_provider_impl=AsyncMock(return_value=impl))
    original = RetainContent(
        content="target",
        previous_source="option 2 means Friday",
        event_date=datetime(2026, 9, 24, tzinfo=timezone.utc),
        event_date_is_default=True,
    )
    cfg = config(retain_context_chars=100)
    with pytest.raises(RuntimeError, match="polling reached"):
        await fact_extraction.extract_facts_from_contents_batch_api(
            [original],
            llm,
            cfg,
            pool=object(),
            operation_id="operation",
        )
    saved_state = json.loads(conn.execute.call_args.args[1])
    conn.fetchrow.return_value = {"result_metadata": saved_state}
    resumed = replace(original, event_date=datetime(2026, 9, 25, tzinfo=timezone.utc))
    with pytest.raises(RuntimeError, match="polling reached"):
        await fact_extraction.extract_facts_from_contents_batch_api(
            [resumed],
            llm,
            cfg,
            pool=object(),
            operation_id="operation",
        )
    assert resumed.event_date == original.event_date
    impl.submit_batch.assert_awaited_once()
    for changed in (
        replace(resumed, previous_source="option 2 means Monday"),
        replace(resumed, event_date=datetime(2026, 9, 26, tzinfo=timezone.utc), event_date_is_default=False),
    ):
        with pytest.raises(RuntimeError, match="batch prompts changed"):
            await fact_extraction.extract_facts_from_contents_batch_api(
                [changed],
                llm,
                cfg,
                pool=object(),
                operation_id="operation",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"_retain_context_chars": 800}, {"context": "ordinary"}, {}, None])
async def test_store_owned_delta_decodes_retain_params(monkeypatch, params) -> None:
    from hindsight_api.engine import memories
    from hindsight_api.engine.memories.base import document_record_metadata
    from hindsight_api.engine.retain import orchestrator

    store = SimpleNamespace(
        store_owned_for=lambda bank_id: True,
        get_document_record=AsyncMock(
            return_value={
                "metadata": document_record_metadata(params),
                "chunk_hashes": [hashlib.sha256(b"unchanged").hexdigest()],
            }
        ),
    )
    monkeypatch.setattr(memories, "get_memories", lambda: store)
    metadata_only = AsyncMock(return_value=orchestrator.RetainBatchResult([[]], TokenUsage(), 0))
    monkeypatch.setattr(orchestrator, "_delta_metadata_only", metadata_only)
    result = await orchestrator._try_delta_retain(
        pool=None,
        embeddings_model=None,
        llm_config=None,
        entity_resolver=None,
        format_date_fn=None,
        bank_id="bank",
        contents_dicts=[{"content": "unchanged"}],
        contents=[RetainContent(content="unchanged")],
        config=config(),
        document_id="doc",
        fact_type_override=None,
        document_tags=None,
        log_buffer=[],
        start_time=0,
        operation_id=None,
        schema=None,
        outbox_callback=None,
    )
    if params and params.get("_retain_context_chars"):
        assert result is None  # Disable transition must re-extract.
        metadata_only.assert_not_awaited()
    else:
        assert result is metadata_only.return_value  # Ordinary retains still reuse unchanged chunks.
        metadata_only.assert_awaited_once()
    store.get_document_record.assert_awaited_once_with(bank_id="bank", document_id="doc", include_text=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["block", "redact"])
async def test_internal_slice_context_is_screened_before_extraction(monkeypatch, action: str) -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from hindsight_api.engine import memories
    from hindsight_api.engine.retain import orchestrator
    from hindsight_api.extensions.memory_defense import DefenseAction, DefenseDecision

    @asynccontextmanager
    async def connection(pool: object) -> AsyncIterator[object]:
        yield object()

    monkeypatch.setattr(orchestrator, "acquire_with_retry", connection)
    store = SimpleNamespace(assert_writable=AsyncMock(), store_owned_for=lambda bank_id: True)
    monkeypatch.setattr(memories, "get_memories", lambda: store)
    streaming = AsyncMock(side_effect=RuntimeError("capture windows"))
    monkeypatch.setattr(orchestrator, "_streaming_retain_batch", streaming)
    defense = SimpleNamespace(
        screen=AsyncMock(
            side_effect=[
                DefenseDecision(action=DefenseAction(action), redacted_content="safe prior"),
                DefenseDecision(action=DefenseAction.ALLOW),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="capture windows"):
        await orchestrator.retain_batch(
            pool=None,
            embeddings_model=None,
            llm_config=None,
            entity_resolver=None,
            format_date_fn=None,
            bank_id="bank",
            document_id="doc",
            contents_dicts=[{"content": "target", "_previous_source": "sensitive prior"}],
            config=config(retain_context_chars=800, memory_defense={"enabled": True}),
            memory_defense_extension=defense,
        )
    assert streaming.call_args.kwargs["previous_sources"] == ([""] if action == "block" else ["safe prior"])
