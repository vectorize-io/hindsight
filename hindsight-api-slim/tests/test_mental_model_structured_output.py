"""Structured output on mental-model refresh (``trigger.response_schema``).

The parsed ``structured_output`` is always derived from the FINAL stored content,
not from reflect's answer. This matters in delta mode: reflect there only sees
facts created since the last refresh, so its answer reflects just the delta —
while the stored content is the delta-*merged* document. Extracting from the
final content keeps the structured view consistent with the markdown in both
modes. These are deterministic tests: reflect_async, the delta-ops LLM call, and
the structured-output extractor are all mocked (no real LLM).
"""

import types
import uuid

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.memory_engine import MentalModelRefreshError
from hindsight_api.engine.reflect import agent as reflect_agent
from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
from hindsight_api.engine.response_models import ReflectResult

_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def _canned_reflect_result(text: str, facts: list[dict] | None = None) -> ReflectResult:
    return ReflectResult.model_validate(
        {
            "text": text,
            "based_on": {
                "observation": facts or [],
                "world": [],
                "experience": [],
                "mental-models": [],
                "directives": [],
            },
        }
    )


def _patch_structured_output(monkeypatch, returns: dict) -> list[str]:
    """Patch _generate_structured_output; record the content it was asked to parse."""
    calls: list[str] = []

    async def fake(answer, response_schema, llm_config, reflect_id, max_tokens=None):
        calls.append(answer)
        return types.SimpleNamespace(
            structured_output=returns,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            thoughts_tokens=0,
        )

    monkeypatch.setattr(reflect_agent, "_generate_structured_output", fake)
    return calls


class TestMentalModelStructuredOutput:
    async def test_full_mode_extracts_from_final_content(
        self, memory: MemoryEngine, request_context: RequestContext, monkeypatch
    ):
        """Full mode: structured output is parsed from the stored content, and reflect
        is not asked to do the extraction itself."""
        bank_id = f"test-mm-struct-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team",
            source_query="team?",
            content="# Team\n\nOriginal.",
            trigger={"mode": "full", "response_schema": _SCHEMA},
            request_context=request_context,
        )

        reflect_calls: list[dict] = []

        async def fake_reflect_async(**kwargs):
            reflect_calls.append(kwargs)
            return _canned_reflect_result("# Team\n\nRegenerated answer.")

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)
        so_calls = _patch_structured_output(monkeypatch, {"summary": "A team."})

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # reflect is no longer asked to derive structured_output.
        assert reflect_calls[0].get("response_schema") is None
        # Extraction runs against the final content (== the answer in full mode).
        assert so_calls == ["# Team\n\nRegenerated answer."]
        assert refreshed["reflect_response"]["structured_output"] == {"summary": "A team."}

    async def test_no_schema_no_structured_output(
        self, memory: MemoryEngine, request_context: RequestContext, monkeypatch
    ):
        """No trigger schema → no extraction call and no stored structured_output."""
        bank_id = f"test-mm-nostruct-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team",
            source_query="team?",
            content="# Team\n\nOriginal.",
            trigger={"mode": "full"},
            request_context=request_context,
        )

        async def fake_reflect_async(**kwargs):
            return _canned_reflect_result("# Team\n\nRegenerated.")

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)
        so_calls = _patch_structured_output(monkeypatch, {"summary": "x"})

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert so_calls == []
        assert "structured_output" not in refreshed["reflect_response"]

    async def test_delta_extracts_from_merged_content(
        self, memory: MemoryEngine, request_context: RequestContext, monkeypatch
    ):
        """Delta mode: structured output is parsed from the merged document, NOT from
        reflect's partial (delta-only) answer."""
        bank_id = f"test-mm-delta-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Doc",
            source_query="doc?",
            content="# Doc\n\n## Section A\n\nOriginal body.",
            trigger={"mode": "delta", "response_schema": _SCHEMA},
            request_context=request_context,
        )

        # reflect returns a partial answer plus a new fact (so the delta actually runs
        # rather than short-circuiting on "no new facts").
        async def fake_reflect_async(**kwargs):
            return _canned_reflect_result(
                "PARTIAL DELTA ANSWER",
                facts=[{"id": str(uuid.uuid4()), "text": "a new fact", "context": None}],
            )

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)

        # Delta-ops call returns no operations → the document is preserved and
        # re-rendered, so final_content is the merged doc (here: the original).
        async def fake_delta_call(**kwargs):
            return DeltaOperationList(operations=[])

        monkeypatch.setattr(memory._reflect_llm_config, "call", fake_delta_call)
        so_calls = _patch_structured_output(monkeypatch, {"summary": "whole document"})

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(so_calls) == 1
        parsed = so_calls[0]
        # The extractor saw the merged document, not reflect's delta-only answer.
        assert "PARTIAL DELTA ANSWER" not in parsed
        assert "Original body." in parsed
        assert refreshed["reflect_response"]["structured_output"] == {"summary": "whole document"}

    async def test_extraction_failure_fails_the_refresh(
        self, memory: MemoryEngine, request_context: RequestContext, monkeypatch
    ):
        """When a schema is configured but extraction yields nothing, the refresh
        raises instead of silently persisting content with no structured output —
        and the prior content is preserved for retry."""
        bank_id = f"test-mm-failloud-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Doc",
            source_query="doc?",
            content="# Doc\n\nOriginal.",
            trigger={"mode": "full", "response_schema": _SCHEMA},
            request_context=request_context,
        )

        async def fake_reflect_async(**kwargs):
            return _canned_reflect_result("# Doc\n\nBrand new content.")

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)
        # Extraction "fails": returns no structured output.
        _patch_structured_output(monkeypatch, None)

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        # The refresh was aborted, so the prior content is untouched.
        reloaded = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert reloaded["content"] == "# Doc\n\nOriginal."
        # …and the failure is auditable, like every other refresh failure: this
        # path used to raise without recording anything, so the only trace of it
        # was a log line.
        assert (reloaded.get("reflect_response") or {}).get("refresh_skipped") == "structured_output_failed"
