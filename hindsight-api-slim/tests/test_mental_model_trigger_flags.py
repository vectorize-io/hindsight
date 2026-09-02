"""Every trigger flag, checked on the delta leg.

``max_tokens`` looked wired up — it was read from the model, passed to reflect,
and enforced by a rewrite — and was still ignored for the document that actually
got stored, because in delta mode the thing it capped never becomes the document.
A flag can be honoured on one leg and silently dropped on the other, and reading
the code is how that was missed the first time.

So each flag here is exercised through a real delta refresh and asserted at its
destination: retrieval options at the reflect call, document options in the delta
prompt or the persisted row. Full mode is covered by the surrounding modules;
this one exists because delta is the leg where a flag goes to die.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from hindsight_api import MemoryEngine, RequestContext
from tests.test_mental_model_delta import patch_llm_call, patch_reflect  # noqa: F401 — fixtures

pytestmark = pytest.mark.asyncio

_APPEND_OP = [{"op": "append_block", "section_id": "ops", "text": "New note."}]
_FACTS = [{"id": "o1", "text": "a new fact about the API", "type": "observation", "context": None}]


async def _refresh_with_trigger(
    memory: MemoryEngine,
    request_context: RequestContext,
    patch_reflect,
    patch_llm_call,
    trigger: dict[str, Any],
    *,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Run one delta refresh and hand back everything worth asserting on."""
    bank_id = f"test-trigger-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)
    mm = await memory.create_mental_model(
        bank_id=bank_id,
        name="API Reference",
        source_query="Document the API",
        content="## Ops\n\nOriginal.\n",
        tags=["team:core"],
        max_tokens=max_tokens,
        trigger={"mode": "delta", **trigger},
        request_context=request_context,
    )
    # A real memory in the model's scope, retained after it: a refresh with nothing to
    # read skips the reflect loop entirely (#3875), and every assertion here is about
    # what reaches that call. Tagged to match, and written after the model's creation
    # timestamp, so it falls inside the delta window these tests run in.
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "The recall endpoint accepts a tags_match parameter."}],
        document_tags=["team:core"],
        request_context=request_context,
    )
    await memory.wait_for_background_tasks()
    reflect_calls = patch_reflect(memory, text="## Ops\n\nCandidate.\n", facts=_FACTS)
    llm_calls = patch_llm_call(memory, returns=_APPEND_OP)

    refreshed = await memory.refresh_mental_model(
        bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
    )
    stored = await memory.get_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
    await memory.delete_bank(bank_id, request_context=request_context)
    return {
        "refreshed": refreshed,
        "stored": stored,
        "reflect_kwargs": reflect_calls[0] if reflect_calls else {},
        "delta_prompt": llm_calls[0]["messages"][1]["content"] if llm_calls else "",
        "delta_calls": llm_calls,
    }


class TestRetrievalFlagsReachReflect:
    """Options that shape what the refresh retrieves.

    These ride on the synthesis call, which runs on both legs — so the contract
    is that the stored trigger arrives at ``reflect_async`` unchanged.
    """

    async def test_fact_types(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(
            memory, request_context, patch_reflect, patch_llm_call, {"fact_types": ["observation"]}
        )
        assert run["reflect_kwargs"]["fact_types"] == ["observation"]

    async def test_exclude_mental_models(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(
            memory, request_context, patch_reflect, patch_llm_call, {"exclude_mental_models": True}
        )
        assert run["reflect_kwargs"]["exclude_mental_models"] is True

    async def test_exclude_mental_model_ids(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(
            memory, request_context, patch_reflect, patch_llm_call, {"exclude_mental_model_ids": ["mm-other"]}
        )
        assert "mm-other" in run["reflect_kwargs"]["exclude_mental_model_ids"]

    async def test_a_model_never_feeds_on_itself(self, memory, request_context, patch_reflect, patch_llm_call):
        """Its own previous version is not evidence for its next one."""
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {})
        own_id = run["stored"]["id"]
        assert own_id in (run["reflect_kwargs"].get("exclude_mental_model_ids") or [])

    async def test_include_chunks(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(
            memory, request_context, patch_reflect, patch_llm_call, {"include_chunks": True}
        )
        assert run["reflect_kwargs"]["recall_include_chunks"] is True

    async def test_recall_max_tokens(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(
            memory, request_context, patch_reflect, patch_llm_call, {"recall_max_tokens": 1234}
        )
        assert run["reflect_kwargs"]["recall_max_tokens_override"] == 1234

    async def test_recall_chunks_max_tokens(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(
            memory, request_context, patch_reflect, patch_llm_call, {"recall_chunks_max_tokens": 777}
        )
        assert run["reflect_kwargs"]["recall_chunks_max_tokens_override"] == 777

    async def test_tags_match_applies_to_the_model_tags(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {"tags_match": "all"})
        assert run["reflect_kwargs"]["tags_match"] == "all"
        assert "team:core" in (run["reflect_kwargs"].get("tags") or [])

    async def test_tag_groups_override_flat_tags_entirely(self, memory, request_context, patch_reflect, patch_llm_call):
        """Documented priority, pinned because it looks like a bug from outside.

        Groups carry their own match mode, so a ``tags_match`` set alongside them
        is deliberately not forwarded — and the model's flat tags are dropped
        rather than intersected. Anyone reading only the trigger would expect
        ``tags_match`` to survive; it does not, and that is the rule.
        """
        run = await _refresh_with_trigger(
            memory,
            request_context,
            patch_reflect,
            patch_llm_call,
            {"tags_match": "all", "tag_groups": [{"tags": ["a", "b"], "match": "any"}]},
        )
        assert run["reflect_kwargs"]["tag_groups"]
        assert run["reflect_kwargs"]["tags"] is None
        assert run["reflect_kwargs"]["tags_match"] == "any"

    async def test_tagged_model_defaults_to_strict_isolation(
        self, memory, request_context, patch_reflect, patch_llm_call
    ):
        """No tags_match on a tagged model means all_strict, not 'any' — a model
        scoped to tags must not widen its own scope by default."""
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {})
        assert run["reflect_kwargs"]["tags_match"] == "all_strict"

    async def test_model_tags_scope_retrieval(self, memory, request_context, patch_reflect, patch_llm_call):
        """The model's own tags, not just the trigger's, scope what it reads."""
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {})
        assert "team:core" in (run["reflect_kwargs"].get("tags") or [])


class TestDocumentFlagsReachTheDeltaLeg:
    """Options that shape the stored document.

    This is the class that would have caught the ``max_tokens`` gap: in delta
    mode the document comes from operations, so anything about the document has
    to reach the operations call or the persist path — being passed to the
    synthesis is not enough.
    """

    async def test_max_tokens_reaches_the_delta_prompt(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {}, max_tokens=300)
        rr = run["refreshed"].get("reflect_response") or {}
        assert rr["document_budget"] == 300
        assert rr["document_tokens"] > 0

    async def test_response_schema_is_extracted_from_the_delta_result(
        self, memory, request_context, patch_reflect, patch_llm_call, monkeypatch
    ):
        """The projection must come from the document the delta produced."""
        seen: list[str] = []

        async def fake_structured_output(content_text, schema, llm_config, reflect_id, max_tokens):
            from hindsight_api.engine.reflect.models import StructuredOutputResult

            seen.append(content_text)
            return StructuredOutputResult(structured_output={"summary": "ok"})

        monkeypatch.setattr("hindsight_api.engine.reflect.agent._generate_structured_output", fake_structured_output)
        run = await _refresh_with_trigger(
            memory,
            request_context,
            patch_reflect,
            patch_llm_call,
            {"response_schema": {"type": "object", "properties": {"summary": {"type": "string"}}}},
        )
        rr = run["refreshed"].get("reflect_response") or {}
        assert rr["structured_output"] == {"summary": "ok"}
        # Extracted from the merged document, not from reflect's delta-only answer.
        assert seen and "New note." in seen[0]
        assert "Candidate." not in seen[0]

    async def test_keep_trace_records_the_delta_run(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {"keep_trace": True})
        rr = run["refreshed"].get("reflect_response") or {}
        assert rr.get("trace"), "keep_trace must record the run that produced this version"

    async def test_trace_is_absent_without_keep_trace(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {})
        rr = run["refreshed"].get("reflect_response") or {}
        assert not rr.get("trace")

    async def test_mode_delta_actually_applied(self, memory, request_context, patch_reflect, patch_llm_call):
        run = await _refresh_with_trigger(memory, request_context, patch_reflect, patch_llm_call, {})
        rr = run["refreshed"].get("reflect_response") or {}
        assert rr["delta_applied"] is True
        assert len(run["delta_calls"]) == 1


class TestTriggerRoundTrip:
    """A flag that does not survive being stored is not honoured either.

    This list is deliberately exhaustive over ``MentalModelTrigger``: it is the
    cheapest place for a flag added later to be noticed, and the failure it
    guards against — a field that round-trips as ``None`` — looks like the flag
    being ignored rather than like a storage bug.
    """

    async def test_every_flag_survives_create(self, memory, request_context):
        bank_id = f"test-trigger-rt-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        trigger = {
            "mode": "delta",
            "refresh_after_consolidation": True,
            "refresh_cron": "0 3 * * *",
            "fact_types": ["observation"],
            "exclude_mental_models": True,
            "exclude_mental_model_ids": ["mm-x"],
            "tags_match": "all",
            "tag_groups": [{"tags": ["a"], "match": "any"}],
            "include_chunks": True,
            "recall_max_tokens": 1234,
            "recall_chunks_max_tokens": 777,
            "response_schema": {"type": "object"},
            "keep_trace": True,
            # Gates *automatic* refreshes (#3621) rather than shaping one, so it is
            # honoured in the submit path and covered there; it is here because a
            # flag that does not survive being stored is not honoured either, and
            # this list is what keeps the audit complete as flags are added.
            "min_refresh_interval_seconds": 900,
        }
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Ops\n\nOriginal.\n",
            trigger=trigger,
            request_context=request_context,
        )
        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        for key, value in trigger.items():
            assert stored["trigger"].get(key) == value, f"{key} did not survive create"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_update_patches_the_trigger_instead_of_replacing_it(self, memory, request_context):
        """Changing when a model refreshes must not reset how it refreshes.

        ``update_mental_model`` overwrites the whole trigger column, so a caller that
        sends only the field it wants used to strip every flag it did not mention —
        the defect #3506 fixed for knowledge pages, on the endpoint every MCP agent
        goes through.
        """
        bank_id = f"test-trigger-patch-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Ops\n\nOriginal.\n",
            trigger={
                "mode": "delta",
                "fact_types": ["observation"],
                "recall_max_tokens": 1234,
                "refresh_after_consolidation": True,
            },
            request_context=request_context,
        )

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            trigger={"refresh_cron": "0 3 * * *"},
            request_context=request_context,
        )
        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored["trigger"]["refresh_cron"] == "0 3 * * *"
        assert stored["trigger"]["mode"] == "delta"
        assert stored["trigger"]["fact_types"] == ["observation"]
        assert stored["trigger"]["recall_max_tokens"] == 1234
        # Moving onto a schedule clears the auto-refresh: storing both would be a pair
        # the API itself rejects.
        assert "refresh_after_consolidation" not in stored["trigger"]

        # ...and back again.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            trigger={"refresh_after_consolidation": True},
            request_context=request_context,
        )
        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored["trigger"]["refresh_after_consolidation"] is True
        assert "refresh_cron" not in stored["trigger"]
        assert stored["trigger"]["recall_max_tokens"] == 1234

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_http_style_full_trigger_still_replaces(self, memory, request_context):
        """The merge must not turn the HTTP contract into a patch by accident.

        HTTP routes serialize the whole ``MentalModelTrigger``, defaults included, so
        every key is present and the merge is a replacement. A flag cleared through
        the API has to actually clear.
        """
        bank_id = f"test-trigger-replace-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Ops\n\nOriginal.\n",
            trigger={"mode": "delta", "recall_max_tokens": 1234, "keep_trace": True},
            request_context=request_context,
        )
        from hindsight_api.api.http import MentalModelTrigger

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            trigger=MentalModelTrigger(mode="full").model_dump(),
            request_context=request_context,
        )
        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored["trigger"]["mode"] == "full"
        assert stored["trigger"]["recall_max_tokens"] is None
        assert stored["trigger"]["keep_trace"] is False

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_patch_stating_both_refresh_triggers_is_rejected(self, memory, request_context):
        bank_id = f"test-trigger-excl-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Ops\n\nOriginal.\n",
            request_context=request_context,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            await memory.update_mental_model(
                bank_id=bank_id,
                mental_model_id=mm["id"],
                trigger={"refresh_after_consolidation": True, "refresh_cron": "0 3 * * *"},
                request_context=request_context,
            )
        await memory.delete_bank(bank_id, request_context=request_context)
