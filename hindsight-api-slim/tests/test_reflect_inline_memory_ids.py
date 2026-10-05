"""Deterministic regression for #5166, independent of alias expansion (#4878).

Only the synthetic source citation is constrained here; ordinary UUID prose is
preserved. Unretrieved citations retain their IDs with an explicit warning.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.models import ReflectAgentResult
from hindsight_api.engine.reflect.structured_doc import render_document
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCall, LLMToolCallResult

REAL_ID = "12345678-1111-4111-8111-111111111111"
FORGED_ID = "12345678-2222-4222-8222-222222222222"


async def _scripted_reflect(
    text: str, document_mode: bool, *, forced: bool = False, rewrite: bool = False
) -> ReflectAgentResult:
    # These dictionaries are the external tool-call JSON, as in test_reflect_agent.
    output = (
        {"document": {"sections": [{"heading": "", "level": 2, "blocks": [text]}]}}
        if document_mode
        else {"answer": text}
    )
    output["memory_ids"] = ["f1", FORGED_ID]
    if rewrite:
        original = f"Source: `{REAL_ID}`\n\n" + "An overlong answer. " * 100
        output = (
            {"document": {"sections": [{"heading": "", "level": 2, "blocks": [original]}]}}
            if document_mode
            else {"answer": original}
        )
        output["memory_ids"] = ["f1"]
    llm = MagicMock()
    llm.call = (
        AsyncMock(return_value=LLMCallResult(content=text))
        if forced or rewrite
        else AsyncMock(side_effect=AssertionError("Unexpected synthesis or rewrite"))
    )
    llm.call_with_tools = AsyncMock(
        side_effect=[
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="recall", name="recall", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            *(
                [LLMToolCallResult(content="", finish_reason="stop")] * 2
                if forced
                else [
                    LLMToolCallResult(
                        tool_calls=[LLMToolCall(id="done", name="done", arguments=output)],
                        finish_reason="tool_calls",
                    )
                ]
            ),
        ]
    )
    recall = AsyncMock(return_value={"memories": [{"id": REAL_ID, "text": "A synthetic fact."}]})
    unused = AsyncMock(side_effect=AssertionError("Unexpected retrieval tool"))
    result = await run_reflect_agent(
        llm_config=llm,
        bank_id="synthetic-inline-citation-bank",
        query="q",
        bank_profile={"name": "Synthetic", "mission": "Testing"},
        search_mental_models_fn=unused,
        read_mental_models_fn=unused,
        search_observations_fn=unused,
        recall_fn=recall,
        expand_fn=unused,
        include_observations=False,
        max_iterations=4,
        answer_as_document=document_mode,
        max_tokens=20 if rewrite else None,
    )
    recall.assert_awaited_once()
    unused.assert_not_called()
    if forced or rewrite:
        llm.call.assert_awaited_once()
    else:
        llm.call.assert_not_called()
    if rewrite:
        assert REAL_ID in str(llm.call.await_args.kwargs["messages"])
        assert FORGED_ID not in str(llm.call.await_args.kwargs["messages"])
    assert llm.call_with_tools.await_count == (3 if forced else 2)
    # Structured provenance already resolves f1 and filters the invented ID.
    assert result.used_memory_ids == ([] if forced else [REAL_ID])
    assert result.tool_trace[0].output["memories"][0]["id"] == REAL_ID
    if document_mode:
        assert result.document is not None
        assert render_document(result.document).strip() == result.text
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("document_mode", [False, True], ids=["answer", "document"])
async def test_unretrieved_complete_uuid_is_not_returned_as_a_verified_source(document_mode: bool) -> None:
    citation = f"Source: `{FORGED_ID}`"
    result = await _scripted_reflect(citation, document_mode)

    # A shared eight-character prefix is not evidence for replacing an ID.
    assert REAL_ID not in result.text
    assert result.text == f"{citation} (unverified)"


@pytest.mark.asyncio
@pytest.mark.parametrize("document_mode", [False, True], ids=["answer", "document"])
async def test_valid_source_and_ordinary_uuid_prose_are_preserved(document_mode: bool) -> None:
    text = f"Source: `{REAL_ID}`\n\nThe external request UUID is `{FORGED_ID}`."
    result = await _scripted_reflect(text, document_mode)
    assert result.text == text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["forced", "rewrite", "document-rewrite"])
async def test_final_generation_is_checked_after_the_last_llm_call(path: str) -> None:
    citation = f"Source: `{FORGED_ID}`"
    result = await _scripted_reflect(
        citation, path == "document-rewrite", forced=path == "forced", rewrite=path != "forced"
    )
    assert result.text == f"{citation} (unverified)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "citation",
    [
        f"Source: {FORGED_ID}",
        f"[memory](/v1/default/banks/synthetic-inline-citation-bank/memories/{FORGED_ID})",
        f"[memory](https://example.test/v1/default/banks/synthetic-inline-citation-bank/memories/{FORGED_ID})",
        f"[memory](/v1/default/banks/another-bank/memories/{REAL_ID})",
    ],
)
async def test_explicit_memory_links_and_plain_source_ids_are_marked(citation: str) -> None:
    result = await _scripted_reflect(citation, False)
    assert result.text == f"{citation} (unverified)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"Source: `{FORGED_ID}` (unverified)",
        f"[memory](/v1/default/banks/synthetic-inline-citation-bank/memories/{REAL_ID})",
        f"The external request UUID is `{FORGED_ID}`.",
        f"[request](https://example.test/requests/{FORGED_ID})",
        f"```markdown\nSource: `{FORGED_ID}`\n```",
        f"An example: `[memory](/v1/default/banks/another-bank/memories/{FORGED_ID})`",
    ],
)
async def test_existing_markers_and_uuid_examples_are_left_alone(text: str) -> None:
    result = await _scripted_reflect(text, False)
    assert result.text == text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["markdown", "structured", "delta"])
async def test_refresh_persists_matching_checked_markdown_and_blocks(mode: str, request_context, monkeypatch) -> None:
    from datetime import datetime, timezone

    from hindsight_api.engine.memory_engine import MemoryEngine, _MentalModelScopeWatermark
    from hindsight_api.engine.reflect.structured_doc import StructuredDocument, split_markdown
    from hindsight_api.engine.response_models import ReflectResult, ToolCallTrace

    citation = f"Source: `{FORGED_ID}`"
    document = split_markdown(citation) if mode == "structured" else None
    model = {
        "id": "mm-inline-citation",
        "bank_id": "synthetic-inline-citation-bank",
        "name": "Synthetic sources",
        "source_query": "q",
        "content": "initial",
        "tags": None,
        "trigger": {},
    }
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._operation_validator = None
    engine._authenticate_tenant = AsyncMock(return_value=None)  # type: ignore[method-assign]
    engine.get_mental_model = AsyncMock(return_value=model)  # type: ignore[method-assign]
    engine.update_mental_model = AsyncMock(return_value=model)  # type: ignore[method-assign]
    engine.reflect_async = AsyncMock(  # type: ignore[method-assign]
        return_value=ReflectResult(
            text=citation,
            document=document,
            based_on={},
            tool_trace=[ToolCallTrace(tool="recall", input={}, output={"memories": [{"id": REAL_ID}]}, duration_ms=0)],
        )
    )
    expected = citation + " (unverified)"
    if mode == "delta":
        from contextlib import asynccontextmanager

        from hindsight_api.engine.reflect.delta_ops import AppendBlockOp, DeltaOperationList

        baseline = split_markdown(f"Source: `{REAL_ID}`")
        model["content"] = render_document(baseline)
        model["trigger"] = {"mode": "delta"}
        model["reflect_response"] = {"based_on": {"world": [{"id": REAL_ID, "text": "Old evidence."}]}}
        new_id = "87654321-1111-4111-8111-111111111111"
        engine.reflect_async.return_value = ReflectResult.model_validate(
            {
                "text": "New synthetic evidence.",
                "based_on": {"world": [{"id": new_id, "text": "New evidence."}]},
                "tool_trace": [
                    {"tool": "recall", "input": {}, "output": {"memories": [{"id": new_id}]}, "duration_ms": 0}
                ],
            }
        )
        conn = MagicMock()
        conn.fetchrow = AsyncMock(
            return_value={"last_refreshed_source_query": "q", "structured_content": baseline.model_dump()}
        )

        @asynccontextmanager
        async def connection(*args):
            yield conn

        engine._get_backend = AsyncMock()  # type: ignore[method-assign]
        engine._store_read_conn = connection  # type: ignore[method-assign]
        monkeypatch.setattr("hindsight_api.engine.memory_engine.acquire_with_retry", connection)
        store = MagicMock()
        store.live_memory_ids = AsyncMock(return_value={REAL_ID})
        monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)
        engine._config_resolver = MagicMock()
        engine._config_resolver.resolve_full_config = AsyncMock()
        engine._mental_model_refresh_llm_override = None
        engine._reflect_llm_config = MagicMock()
        operations = AsyncMock(
            return_value=DeltaOperationList(
                operations=[AppendBlockOp(section_id=baseline.sections[0].id, text=citation)]
            )
        )
        monkeypatch.setattr("hindsight_api.engine.reflect.delta_ops.request_delta_operations", operations)
        expected = f"Source: `{REAL_ID}`\n\n{expected}"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine._mental_model_refresh_cutoff = AsyncMock(return_value=now)  # type: ignore[method-assign]
    engine._mental_model_scope_watermark = AsyncMock(  # type: ignore[method-assign]
        return_value=_MentalModelScopeWatermark(newest_in_scope=now, watermark=now)
    )
    await engine.refresh_mental_model(
        bank_id=model["bank_id"], mental_model_id=model["id"], request_context=request_context
    )
    engine.update_mental_model.assert_awaited_once()
    persisted = engine.update_mental_model.await_args.kwargs
    assert persisted["content"].strip() == expected
    if mode == "delta":
        operations.assert_awaited_once()
        store.live_memory_ids.assert_awaited_once()
    stored = StructuredDocument.model_validate(persisted["structured_content"])
    assert render_document(stored) == persisted["content"]
    if document is not None:
        assert stored.sections[0].id == document.sections[0].id
        assert stored.sections[0].blocks[0].id == document.sections[0].blocks[0].id
        assert document.sections[0].blocks[0].text == citation, "do not mutate the supplied document"
