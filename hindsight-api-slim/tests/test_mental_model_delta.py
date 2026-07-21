"""Tests for delta-mode mental model refresh.

Delta mode performs a surgical update on the existing mental model content:
- Unchanged sections are preserved byte-for-byte.
- Stale content is removed.
- New content from observations/facts is added, preferably by extending existing sections.

Fallback rules:
- If the mental model has no existing content, delta falls back to a full regeneration.
- If the source_query has changed since the last refresh, delta falls back to a full regeneration.

This file contains two kinds of tests:

1. TestDeltaRefreshPlumbing: fast, deterministic tests that monkey-patch reflect_async
   and the LLM call to verify branching logic (fallback conditions, provenance tracking).

2. TestDeltaRefreshGeminiEval: real-LLM behavioral evals against Gemini. These are
   gated on HINDSIGHT_RUN_GEMINI_EVALS=1 (plus a Gemini API key) because they cost
   money/time and require network access. They verify the actual quality of delta
   updates — format preservation, surgical edits, observation-grounding.
"""

import asyncio
import json
import os
import uuid
from typing import Any

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.response_models import ReflectResult
from hindsight_api.engine.retain import embedding_utils


def _canned_reflect_result(text: str, facts: list[dict] | None = None) -> ReflectResult:
    """Build a minimal ReflectResult for monkey-patching reflect_async."""
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


@pytest.fixture
def patch_reflect(monkeypatch):
    """Helper that patches memory.reflect_async to return a canned result and records the call.

    Usage:
        calls = patch_reflect(memory, text="hello", facts=[...])
        await memory.refresh_mental_model(...)
        assert len(calls) == 1
    """

    def _install(memory: MemoryEngine, *, text: str, facts: list[dict] | None = None):
        calls: list[dict] = []

        async def fake_reflect_async(**kwargs):
            calls.append(kwargs)
            return _canned_reflect_result(text, facts)

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)
        return calls

    return _install


@pytest.fixture
def patch_llm_call(monkeypatch):
    """Patch the reflect LLM config's ``.call()`` used for the structured delta call.

    The structured-delta path passes ``response_format=DeltaOperationList``, so the
    LLM returns a Pydantic instance.  Each invocation of ``patch_llm_call`` installs
    a single canned response, in any of these shapes:

    - ``DeltaOperationList`` instance → returned as-is
    - ``[]`` (empty list) → no operations (this is the no-change case)
    - ``[{"op": "...", ...}, ...]`` → wrapped into ``{"operations": [...]}``
    - ``{"operations": [...]}`` → validated directly
    """
    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

    def _to_op_list(resp: Any) -> DeltaOperationList:
        if isinstance(resp, DeltaOperationList):
            return resp
        if isinstance(resp, dict):
            if "operations" in resp:
                return DeltaOperationList.model_validate(resp)
            # Treat a bare op dict as a one-op list for ergonomics.
            return DeltaOperationList.model_validate({"operations": [resp]})
        if isinstance(resp, list):
            return DeltaOperationList.model_validate({"operations": resp})
        if isinstance(resp, str):
            # Tests that expect *no* call ever still install a sentinel; treat as no-op.
            return DeltaOperationList()
        raise TypeError(f"unsupported canned LLM response: {type(resp)!r}")

    def _install(memory: MemoryEngine, *, returns):
        calls: list[dict] = []
        canned = _to_op_list(returns)

        async def fake_call(*, messages, **kwargs):
            calls.append({"messages": messages, **kwargs})
            return canned

        monkeypatch.setattr(memory._reflect_llm_config, "call", fake_call)
        return calls

    return _install


class TestDeltaRefreshPlumbing:
    """Deterministic tests that verify the branching/plumbing of delta-mode refresh."""

    async def test_full_mode_does_not_call_delta_merge(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """When trigger.mode='full', no second LLM call for delta merge occurs."""
        bank_id = f"test-delta-full-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nOriginal content.",
            trigger={"mode": "full"},
            request_context=request_context,
        )

        patch_reflect(memory, text="# Team\n\nRegenerated from scratch.")
        llm_calls = patch_llm_call(memory, returns="should-not-be-called")

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert refreshed is not None
        assert refreshed["content"] == "# Team\n\nRegenerated from scratch."
        assert len(llm_calls) == 0, "Delta merge LLM call must not happen in full mode"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_mode_empty_content_falls_back_to_full(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """When the mental model has no existing content there is nothing to anchor
        a surgical edit on, so delta falls back to full regeneration. The user's
        candidate from reflect_async is used verbatim.
        """
        bank_id = f"test-delta-empty-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="",  # no existing content
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        patch_reflect(memory, text="# Team\n\nFull fresh synthesis.")
        llm_calls = patch_llm_call(memory, returns=[])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert refreshed["content"] == "# Team\n\nFull fresh synthesis."
        assert len(llm_calls) == 0  # delta path skipped entirely
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is not True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_mode_pending_placeholder_falls_back_to_full(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """The async creation placeholder is not a real delta baseline.

        A first refresh for a newly-created model must do a full recall over
        pre-existing facts instead of scoping recall to last_refreshed_at.
        """
        bank_id = f"test-delta-placeholder-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Backend Overview",
            source_query="What is the backend architecture?",
            content="Generating content...",
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        reflect_calls = patch_reflect(memory, text="# Backend\n\nFull fresh synthesis.")
        llm_calls = patch_llm_call(memory, returns="should-not-be-called")

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert refreshed["content"] == "# Backend\n\nFull fresh synthesis."
        assert len(llm_calls) == 0
        assert "created_after" not in reflect_calls[0]
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is not True
        assert rr.get("delta_skipped_reason") is None

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_mode_source_query_change_falls_back_to_full(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """If source_query changes after a refresh, the next delta run must do a full rewrite."""
        bank_id = f"test-delta-query-change-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nBaseline.",
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # First refresh: establishes last_refreshed_source_query.
        patch_reflect(memory, text="# Team\n\nFirst pass.")
        patch_llm_call(memory, returns="unused-first")
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        # Now change the source_query — a genuine topic shift.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            source_query="Tell me about customers instead",
            request_context=request_context,
        )

        # Second refresh under the new query must do a FULL rewrite, not a delta merge.
        patch_reflect(memory, text="# Customers\n\nBrand new topic.")
        llm_calls = patch_llm_call(memory, returns="should-not-be-called")

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert refreshed["content"] == "# Customers\n\nBrand new topic."
        assert len(llm_calls) == 0, "Source-query change must bypass the delta merge"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_no_new_facts_records_visible_scope_state(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """A successful no-op must suppress the same visible stale set without moving its watermark."""
        bank_id = f"test-delta-watermark-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Preferences\n\nThe user prefers concise answers.\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="User Preferences",
            source_query="What are the user's durable collaboration preferences?",
            content=existing,
            trigger={"mode": "delta", "refresh_cron": "* * * * *"},
            request_context=request_context,
        )

        # Reproduce an established model whose cron is overdue, then add a fresh
        # but topic-irrelevant fact. The coarse staleness query sees this row while
        # the reflect agent correctly returns no supporting facts for the model.
        assert memory._pool is not None
        stale_fact_id = uuid.uuid4()
        async with memory._pool.acquire() as conn:
            before = await conn.fetchval(
                """
                UPDATE mental_models
                SET last_refreshed_at = NOW() - INTERVAL '1 day',
                    last_refreshed_source_query = source_query
                WHERE bank_id = $1 AND id = $2
                RETURNING last_refreshed_at
                """,
                bank_id,
                mm["id"],
            )
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at)
                VALUES ($1, $2, 'The build server uses Linux.', 'world', ARRAY[]::varchar[], NOW(), NOW())
                """,
                stale_fact_id,
                bank_id,
            )
            stale_row = await conn.fetchrow(
                "SELECT id, tags, trigger, last_refreshed_at FROM mental_models WHERE bank_id = $1 AND id = $2",
                bank_id,
                mm["id"],
            )
            assert stale_row is not None
            assert await memory.compute_mental_model_is_stale(conn, bank_id, stale_row) is True

        patch_reflect(memory, text="No relevant preference changes.", facts=[])
        delta_llm_calls = patch_llm_call(memory, returns="should-not-be-called")

        async def fail_embedding_generation(*args, **kwargs):
            raise AssertionError("A no-op delta refresh must not regenerate the embedding")

        monkeypatch.setattr(embedding_utils, "generate_embeddings_batch", fail_embedding_generation)

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )

        assert refreshed is not None
        assert refreshed["content"] == existing
        assert len(delta_llm_calls) == 0
        assert (refreshed.get("reflect_response") or {}).get("delta_skipped_reason") == "no_new_facts"

        async with memory._pool.acquire() as conn:
            mm_row = await conn.fetchrow(
                """
                SELECT id, name, tags, trigger, source_query, reflect_response,
                       last_refreshed_at, max_tokens
                FROM mental_models WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                mm["id"],
            )
            assert mm_row is not None
            after = mm_row["last_refreshed_at"]
            is_stale = await memory.compute_mental_model_is_stale(conn, bank_id, mm_row)
            partial_row = await conn.fetchrow(
                "SELECT id, tags, trigger, last_refreshed_at FROM mental_models WHERE bank_id = $1 AND id = $2",
                bank_id,
                mm["id"],
            )
            assert partial_row is not None
            assert await memory.compute_mental_model_is_stale(conn, bank_id, partial_row) is False
            history_count = await conn.fetchval(
                "SELECT COUNT(*) FROM mental_model_history WHERE bank_id = $1 AND mental_model_id = $2",
                bank_id,
                mm["id"],
            )
        assert after == before
        assert is_stale is False
        assert history_count == 0
        scope_state = (refreshed.get("reflect_response") or {}).get("delta_scope_state")
        assert scope_state is not None
        assert scope_state["count"] == 1

        # Every option that changes refresh strategy or visible evidence must
        # invalidate the token even when the currently matching rows are unchanged.
        base_row = dict(mm_row)
        base_trigger = base_row["trigger"]
        if isinstance(base_trigger, str):
            base_trigger = json.loads(base_trigger)
        else:
            base_trigger = dict(base_trigger)
        trigger_changes = [
            {"mode": "full"},
            {"include_chunks": False},
            {"recall_max_tokens": 1234},
            {"recall_chunks_max_tokens": 567},
            {"exclude_mental_models": True},
            {"exclude_mental_model_ids": ["another-model"]},
            {"fact_types": ["experience"]},
            {"tag_groups": [{"tags": ["scope:other"], "match": "any"}]},
        ]
        async with memory._pool.acquire() as conn:
            for change in trigger_changes:
                candidate = dict(base_row)
                candidate["trigger"] = {**base_trigger, **change}
                candidate_state = await memory._compute_mental_model_scope_state(conn, bank_id, candidate)
                assert candidate_state is not None
                assert candidate_state.as_dict() != scope_state
            for key, value in (
                ("name", "Changed model name"),
                ("source_query", "A changed source query"),
                ("max_tokens", (base_row["max_tokens"] or 0) + 1),
                ("tags", ["scope:other"]),
            ):
                candidate = {**base_row, key: value}
                candidate_state = await memory._compute_mental_model_scope_state(conn, bank_id, candidate)
                assert candidate_state is not None
                assert candidate_state.as_dict() != scope_state

            stored_response = base_row["reflect_response"] or {}
            if isinstance(stored_response, str):
                stored_response = json.loads(stored_response)
            else:
                stored_response = dict(stored_response)

            legacy_row = dict(base_row)
            legacy_response = dict(stored_response)
            legacy_response["delta_scope_state"] = {**scope_state, "version": 2}
            legacy_row["reflect_response"] = legacy_response
            assert (
                await memory.compute_mental_model_is_stale(
                    conn,
                    bank_id,
                    legacy_row,
                    request_context,
                )
                is True
            )

            malformed_scope_states = [
                {**scope_state, "version": "3"},
                {**scope_state, "version": True},
                {**scope_state, "count": str(scope_state["count"])},
                {**scope_state, "count": True},
                {**scope_state, "count": -1},
                {**scope_state, "since": None},
                {**scope_state, "since": scope_state["since"].replace("Z", "+00:00")},
                {**scope_state, "sha256": 123},
                {**scope_state, "sha256": f"g{scope_state['sha256'][1:]}"},
                {key: value for key, value in scope_state.items() if key != "sha256"},
                {**scope_state, "extra": "unexpected"},
            ]
            for malformed_scope_state in malformed_scope_states:
                malformed_row = dict(base_row)
                malformed_response = dict(stored_response)
                malformed_response["delta_scope_state"] = malformed_scope_state
                malformed_row["reflect_response"] = malformed_response
                assert (
                    await memory.compute_mental_model_is_stale(
                        conn,
                        bank_id,
                        malformed_row,
                        request_context,
                    )
                    is True
                )

            def fail_nested_acquire(*args, **kwargs):
                raise AssertionError("Scope-state config resolution must reuse the held DB connection")

            with monkeypatch.context() as nested_patch:
                nested_patch.setattr(memory._config_resolver._backend, "acquire", fail_nested_acquire)
                same_scope_state = await memory._compute_mental_model_scope_state(
                    conn,
                    bank_id,
                    base_row,
                    request_context,
                )
            assert same_scope_state is not None
            assert same_scope_state.as_dict() == scope_state

        submitted: list[str] = []

        async def record_submit(
            *, bank_id: str, mental_model_id: str, request_context: RequestContext
        ) -> dict[str, str]:
            submitted.append(mental_model_id)
            return {"operation_id": str(uuid.uuid4())}

        monkeypatch.setattr(memory, "submit_async_refresh_mental_model", record_submit)
        original_scope_state = memory._compute_mental_model_scope_state
        scope_checks = 0

        async def count_scope_checks(*args, **kwargs):
            nonlocal scope_checks
            scope_checks += 1
            return await original_scope_state(*args, **kwargs)

        monkeypatch.setattr(memory, "_compute_mental_model_scope_state", count_scope_checks)
        maintenance = MaintenanceLoop(memory)
        await maintenance._run_scheduled_mm_refresh()
        checked_fire = next(iter(maintenance._scheduled_mm_checked_fires.values()))
        orphaned_key = ("deleted-schema", "deleted-bank", "deleted-model")
        maintenance._scheduled_mm_checked_fires[orphaned_key] = checked_fire
        await maintenance._run_scheduled_mm_refresh()
        assert orphaned_key not in maintenance._scheduled_mm_checked_fires
        assert mm["id"] not in submitted
        assert scope_checks == 1

        trigger_without_cron = {key: value for key, value in base_trigger.items() if key != "refresh_cron"}
        await memory.update_mental_model(
            bank_id,
            mm["id"],
            trigger=trigger_without_cron,
            request_context=request_context,
        )
        maintenance._scheduled_mm_checked_fires[orphaned_key] = checked_fire
        await maintenance._run_scheduled_mm_refresh()
        assert maintenance._scheduled_mm_checked_fires == {}

        await memory.update_mental_model(
            bank_id,
            mm["id"],
            trigger=base_trigger,
            request_context=request_context,
        )
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "UPDATE mental_models SET last_refreshed_at = NOW() WHERE bank_id = $1 AND id = $2",
                bank_id,
                mm["id"],
            )
        maintenance._scheduled_mm_checked_fires[orphaned_key] = checked_fire
        await maintenance._run_scheduled_mm_refresh()
        assert maintenance._scheduled_mm_checked_fires == {}
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "UPDATE mental_models SET last_refreshed_at = $3 WHERE bank_id = $1 AND id = $2",
                bank_id,
                mm["id"],
                after,
            )

        await maintenance._run_scheduled_mm_refresh()
        valid_checkpoints = dict(maintenance._scheduled_mm_checked_fires)
        assert len(valid_checkpoints) == 1
        maintenance._scheduled_mm_checked_fires[orphaned_key] = checked_fire

        async def fail_tenant_discovery():
            raise RuntimeError("simulated tenant discovery failure")

        with monkeypatch.context() as tenant_failure_patch:
            tenant_failure_patch.setattr(memory._tenant_extension, "list_tenants", fail_tenant_discovery)
            await maintenance._run_scheduled_mm_refresh()
        assert maintenance._scheduled_mm_checked_fires == valid_checkpoints

        # Trigger fields are null here, so Reflect resolves these values through
        # env/tenant/bank config. Changing the effective value must invalidate the
        # token even though the stored trigger and visible facts are unchanged.
        effective_config = await memory._config_resolver.get_bank_config(bank_id, request_context)
        effective_changes = {
            "recall_include_chunks": not effective_config["recall_include_chunks"],
            "recall_max_tokens": effective_config["recall_max_tokens"] + 1,
            "recall_chunks_max_tokens": effective_config["recall_chunks_max_tokens"] + 1,
            "reflect_source_facts_max_tokens": effective_config["reflect_source_facts_max_tokens"] + 1,
        }
        for config_key, changed_value in effective_changes.items():
            await memory._config_resolver.update_bank_config(
                bank_id,
                {config_key: changed_value},
                request_context,
            )
            async with memory._pool.acquire() as conn:
                assert (
                    await memory.compute_mental_model_is_stale(
                        conn,
                        bank_id,
                        mm_row,
                        request_context,
                    )
                    is True
                )
            await memory._config_resolver.update_bank_config(
                bank_id,
                {config_key: None},
                request_context,
            )
            async with memory._pool.acquire() as conn:
                assert (
                    await memory.compute_mental_model_is_stale(
                        conn,
                        bank_id,
                        mm_row,
                        request_context,
                    )
                    is False
                )

        async with memory._pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_units SET text = 'Changed' WHERE bank_id = $1 AND id = $2",
                bank_id,
                stale_fact_id,
            )
            assert await memory.compute_mental_model_is_stale(conn, bank_id, mm_row) is True
            await conn.execute(
                "DELETE FROM memory_units WHERE bank_id = $1 AND id = $2",
                bank_id,
                stale_fact_id,
            )
            assert await memory.compute_mental_model_is_stale(conn, bank_id, mm_row) is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_no_new_facts_preserves_inflight_memory_transaction(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """The refresh watermark must not pass facts excluded from its recall snapshot."""
        bank_id = f"test-delta-cutoff-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="User Preferences",
            source_query="What are the user's durable collaboration preferences?",
            content="# Preferences\n\nThe user prefers concise answers.\n",
            trigger={"mode": "delta", "refresh_cron": "* * * * *"},
            request_context=request_context,
        )

        assert memory._pool is not None
        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE mental_models
                SET last_refreshed_at = NOW() - INTERVAL '1 day',
                    last_refreshed_source_query = source_query
                WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                mm["id"],
            )

        async with memory._pool.acquire() as conn:
            stale_last_refreshed_at = await conn.fetchval(
                "SELECT last_refreshed_at FROM mental_models WHERE bank_id = $1 AND id = $2",
                bank_id,
                mm["id"],
            )

        # Start a transaction before refresh captures its state, but do not commit
        # it until after reflect has returned. Its timestamp is old enough to fall
        # behind a wall-clock cutoff even though recall cannot see the row.
        late_conn = await memory._pool.acquire()
        late_tx = late_conn.transaction()
        await late_tx.start()
        late_fact_id = uuid.uuid4()
        await late_conn.execute(
            """
            INSERT INTO memory_units
                (id, bank_id, text, fact_type, tags, created_at, updated_at)
            VALUES
                ($1, $2, 'The user now prefers detailed answers.', 'world',
                 ARRAY[]::varchar[], NOW(), NOW())
            """,
            late_fact_id,
            bank_id,
        )

        reflect_calls = patch_reflect(memory, text="No relevant preference changes.", facts=[])
        delta_llm_calls = patch_llm_call(memory, returns="should-not-be-called")
        original_update = memory.update_mental_model
        late_tx_committed = False

        async def commit_late_fact_then_update(*args, **kwargs):
            nonlocal late_tx_committed
            # refresh_mental_model has already consumed the reflect result when it
            # reaches update_mental_model. Commit the previously invisible fact in
            # that exact race window.
            await late_tx.commit()
            late_tx_committed = True
            return await original_update(*args, **kwargs)

        monkeypatch.setattr(memory, "update_mental_model", commit_late_fact_then_update)

        try:
            refreshed = await memory.refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mm["id"],
                request_context=request_context,
            )
        finally:
            if not late_tx_committed:
                await late_tx.rollback()
            await memory._pool.release(late_conn)

        assert refreshed is not None
        assert len(reflect_calls) == 1
        assert len(delta_llm_calls) == 0

        async with memory._pool.acquire() as conn:
            mm_row = await conn.fetchrow(
                """
                SELECT id, name, tags, trigger, source_query, reflect_response,
                       last_refreshed_at, max_tokens
                FROM mental_models WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                mm["id"],
            )
            late_updated_at = await conn.fetchval(
                "SELECT updated_at FROM memory_units WHERE bank_id = $1 AND id = $2",
                bank_id,
                late_fact_id,
            )
            assert mm_row is not None
            assert mm_row["last_refreshed_at"] == stale_last_refreshed_at
            assert late_updated_at > mm_row["last_refreshed_at"]
            stored_response = mm_row["reflect_response"]
            if isinstance(stored_response, str):
                stored_response = json.loads(stored_response)
            stored_scope_state = stored_response["delta_scope_state"]
            current_scope_state = await memory._compute_mental_model_scope_state(conn, bank_id, mm_row)
            assert stored_scope_state["count"] == 0
            assert current_scope_state is not None
            assert current_scope_state.count == 1
            assert current_scope_state.as_dict() != stored_scope_state
            assert await memory.compute_mental_model_is_stale(conn, bank_id, mm_row) is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_contentful_delta_preserves_inflight_memory_transaction(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """A content-changing delta must not advance past evidence it could not see."""
        bank_id = f"test-delta-content-race-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="User Preferences",
            source_query="What are the user's durable collaboration preferences?",
            content="# Preferences\n\nThe user prefers concise answers.\n",
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        assert memory._pool is not None
        async with memory._pool.acquire() as conn:
            stale_last_refreshed_at = await conn.fetchval(
                """
                UPDATE mental_models
                SET last_refreshed_at = NOW() - INTERVAL '1 day',
                    last_refreshed_source_query = source_query
                WHERE bank_id = $1 AND id = $2
                RETURNING last_refreshed_at
                """,
                bank_id,
                mm["id"],
            )

        late_conn = await memory._pool.acquire()
        late_tx = late_conn.transaction()
        await late_tx.start()
        late_fact_id = uuid.uuid4()
        await late_conn.execute(
            """
            INSERT INTO memory_units
                (id, bank_id, text, fact_type, tags, created_at, updated_at)
            VALUES
                ($1, $2, 'The user now prefers detailed answers.', 'world',
                 ARRAY[]::varchar[], NOW(), NOW())
            """,
            late_fact_id,
            bank_id,
        )

        patch_reflect(
            memory,
            text="The user prefers concise answers with examples.",
            facts=[
                {
                    "id": "visible-supporting-fact",
                    "text": "The user prefers examples.",
                    "type": "world",
                    "context": None,
                }
            ],
        )
        patch_llm_call(
            memory,
            returns=[
                {
                    "op": "append_block",
                    "section_id": "preferences",
                    "block": {"type": "paragraph", "text": "The user prefers examples."},
                }
            ],
        )
        original_update = memory.update_mental_model
        late_tx_committed = False

        async def commit_late_fact_then_update(*args, **kwargs):
            nonlocal late_tx_committed
            await late_tx.commit()
            late_tx_committed = True
            return await original_update(*args, **kwargs)

        monkeypatch.setattr(memory, "update_mental_model", commit_late_fact_then_update)
        try:
            refreshed = await memory.refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mm["id"],
                request_context=request_context,
            )
        finally:
            if not late_tx_committed:
                await late_tx.rollback()
            await memory._pool.release(late_conn)

        assert refreshed is not None
        assert "The user prefers examples." in refreshed["content"]
        async with memory._pool.acquire() as conn:
            mm_row = await conn.fetchrow(
                """
                SELECT id, name, tags, trigger, source_query, reflect_response,
                       last_refreshed_at, max_tokens
                FROM mental_models WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                mm["id"],
            )
            late_updated_at = await conn.fetchval(
                "SELECT updated_at FROM memory_units WHERE bank_id = $1 AND id = $2",
                bank_id,
                late_fact_id,
            )
            assert mm_row is not None
            assert mm_row["last_refreshed_at"] == stale_last_refreshed_at
            assert late_updated_at > mm_row["last_refreshed_at"]
            stored_response = mm_row["reflect_response"]
            if isinstance(stored_response, str):
                stored_response = json.loads(stored_response)
            stored_scope_state = stored_response["delta_scope_state"]
            current_scope_state = await memory._compute_mental_model_scope_state(conn, bank_id, mm_row)
            assert stored_scope_state["count"] == 0
            assert current_scope_state is not None
            assert current_scope_state.count == 1
            assert current_scope_state.as_dict() != stored_scope_state
            assert await memory.compute_mental_model_is_stale(conn, bank_id, mm_row) is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_contentful_delta_reverse_completion_remains_stale(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        monkeypatch,
    ):
        """An older contentful worker finishing last must leave a retry signal."""
        from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

        bank_id = f"test-delta-reverse-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="User Preferences",
            source_query="What are the user's durable collaboration preferences?",
            content="# Preferences\n\nThe user prefers concise answers.\n",
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        assert memory._pool is not None
        async with memory._pool.acquire() as conn:
            stale_last_refreshed_at = await conn.fetchval(
                """
                UPDATE mental_models
                SET last_refreshed_at = NOW() - INTERVAL '1 day',
                    last_refreshed_source_query = source_query
                WHERE bank_id = $1 AND id = $2
                RETURNING last_refreshed_at
                """,
                bank_id,
                mm["id"],
            )

        first_reflect_started = asyncio.Event()
        release_first_reflect = asyncio.Event()
        reflect_count = 0

        async def ordered_reflect(**kwargs):
            nonlocal reflect_count
            reflect_count += 1
            if reflect_count == 1:
                first_reflect_started.set()
                await release_first_reflect.wait()
                return _canned_reflect_result(
                    "Older worker candidate.",
                    [
                        {
                            "id": "older-support",
                            "text": "Older supporting fact.",
                            "type": "world",
                            "context": None,
                        }
                    ],
                )
            return _canned_reflect_result(
                "Newer worker candidate.",
                [
                    {
                        "id": "newer-support",
                        "text": "Newer supporting fact.",
                        "type": "world",
                        "context": None,
                    }
                ],
            )

        async def content_specific_delta(*, messages, **kwargs):
            prompt = str(messages)
            text = "Newer update." if "Newer supporting fact" in prompt else "Older update."
            return DeltaOperationList.model_validate(
                {
                    "operations": [
                        {
                            "op": "append_block",
                            "section_id": "preferences",
                            "block": {"type": "paragraph", "text": text},
                        }
                    ]
                }
            )

        monkeypatch.setattr(memory, "reflect_async", ordered_reflect)
        monkeypatch.setattr(memory._reflect_llm_config, "call", content_specific_delta)

        original_update = memory.update_mental_model
        newer_update_done = asyncio.Event()

        async def reverse_updates(*args, **kwargs):
            content = kwargs.get("content") or ""
            if "Older update." in content:
                await newer_update_done.wait()
                return await original_update(*args, **kwargs)
            result = await original_update(*args, **kwargs)
            newer_update_done.set()
            return result

        monkeypatch.setattr(memory, "update_mental_model", reverse_updates)
        older_task = asyncio.create_task(
            memory.refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mm["id"],
                request_context=request_context,
            )
        )
        await first_reflect_started.wait()

        async with memory._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at)
                VALUES ($1, $2, 'A fact visible only to the newer worker.', 'world',
                        ARRAY[]::varchar[], NOW(), NOW())
                """,
                uuid.uuid4(),
                bank_id,
            )

        newer_result = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )
        assert newer_result is not None
        assert "Newer update." in newer_result["content"]
        release_first_reflect.set()
        older_result = await older_task
        assert older_result is not None
        assert "Older update." in older_result["content"]

        async with memory._pool.acquire() as conn:
            mm_row = await conn.fetchrow(
                """
                SELECT id, name, tags, trigger, source_query, reflect_response,
                       last_refreshed_at, max_tokens
                FROM mental_models WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                mm["id"],
            )
            assert mm_row is not None
            assert mm_row["last_refreshed_at"] == stale_last_refreshed_at
            stored_response = mm_row["reflect_response"]
            if isinstance(stored_response, str):
                stored_response = json.loads(stored_response)
            assert stored_response["delta_scope_state"]["count"] == 0
            current_scope_state = await memory._compute_mental_model_scope_state(conn, bank_id, mm_row)
            assert current_scope_state is not None
            assert current_scope_state.count == 1
            assert await memory.compute_mental_model_is_stale(conn, bank_id, mm_row) is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_mode_applies_ops_when_query_stable(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """When content exists and source_query is stable, the delta LLM produces ops
        that are applied against the parsed structured doc. The unchanged section
        renders byte-identical, the new fact lands in a new block.
        """
        bank_id = f"test-delta-apply-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nAlice is the lead.\n\n## Members\n\n- Alice — lead\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # First refresh: empty op list → structured doc unchanged → markdown is the
        # render of the parsed existing content. This also seeds the tracking column.
        patch_reflect(memory, text="ignored — full mode candidate")
        patch_llm_call(memory, returns=[])  # zero ops
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        # Second refresh: a new fact arrives; LLM returns one append_block op.
        candidate = "# Team\n\nAlice is the lead. Bob joined as junior engineer."
        patch_reflect(
            memory,
            text=candidate,
            facts=[
                {
                    "id": "obs-bob",
                    "text": "Bob joined the team as junior engineer",
                    "type": "observation",
                    "context": None,
                }
            ],
        )
        ops = [
            {
                "op": "append_block",
                "section_id": "members",
                "block": {
                    "type": "bullet_list",
                    "items": ["Bob — junior engineer"],
                },
            }
        ]
        llm_calls = patch_llm_call(memory, returns=ops)

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(llm_calls) == 1, "Structured-delta LLM call must fire exactly once"
        system_msg = llm_calls[0]["messages"][0]["content"]
        user_msg = llm_calls[0]["messages"][1]["content"]
        # Prompt must include the structured doc + supporting facts + the system prompt.
        assert "integrating" in system_msg.lower()
        assert "operations" in system_msg.lower()
        assert "obs-bob" in user_msg
        assert "Bob joined" in user_msg
        # The structured JSON of the current doc must include the section id "members".
        assert '"members"' in user_msg

        # New content includes the new bullet.
        assert "Bob — junior engineer" in refreshed["content"]
        # Unchanged section ("Alice is the lead.") still present.
        assert "Alice is the lead." in refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is True
        applied = rr.get("delta_operations_applied") or []
        assert len(applied) == 1
        assert applied[0]["op"] == "append_block"
        assert applied[0]["section_id"] == "members"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_prompt_sends_only_new_facts_not_accumulated_history(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """Regression: the delta prompt carries only THIS refresh's facts.

        ``based_on`` accumulates across refreshes for grounding/audit, but the
        structured-delta LLM call must receive only the facts produced by the
        current reflect. Re-sending every historical fact each refresh grows the
        prompt without bound and trips provider input limits (e.g. Z.ai 1261).
        The accumulated set is still persisted in ``reflect_response.based_on``.
        """
        bank_id = f"test-delta-newfacts-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nAlice is the lead.\n\n## Members\n\n- Alice — lead\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # First refresh seeds prior based_on with an OLD fact (zero ops applied).
        patch_reflect(
            memory,
            text="ignored — delta keeps existing",
            facts=[
                {
                    "id": "obs-old-alice",
                    "text": "Alice has been the team lead since 2019",
                    "type": "observation",
                    "context": None,
                }
            ],
        )
        patch_llm_call(memory, returns=[])
        first = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        first_based_on = (first.get("reflect_response") or {}).get("based_on") or {}
        assert "obs-old-alice" in {f.get("id") for f in first_based_on.get("observation", [])}

        # Second refresh brings only a NEW fact.
        patch_reflect(
            memory,
            text="# Team\n\nAlice is the lead. Bob joined.",
            facts=[
                {
                    "id": "obs-new-bob",
                    "text": "Bob joined the team as junior engineer",
                    "type": "observation",
                    "context": None,
                }
            ],
        )
        ops = [
            {
                "op": "append_block",
                "section_id": "members",
                "block": {"type": "bullet_list", "items": ["Bob — junior engineer"]},
            }
        ]
        llm_calls = patch_llm_call(memory, returns=ops)

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(llm_calls) == 1
        user_msg = llm_calls[0]["messages"][1]["content"]
        # The NEW fact is sent to the delta call...
        assert "obs-new-bob" in user_msg
        assert "Bob joined the team" in user_msg
        # ...but the accumulated OLD fact must NOT be re-sent (the regression).
        assert "obs-old-alice" not in user_msg
        assert "Alice has been the team lead since 2019" not in user_msg

        # based_on still ACCUMULATES both facts for grounding/audit.
        based_on = (refreshed.get("reflect_response") or {}).get("based_on") or {}
        obs_ids = {f.get("id") for f in based_on.get("observation", [])}
        assert obs_ids == {"obs-new-bob", "obs-old-alice"}

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_zero_ops_keeps_existing_content_byte_identical(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """Zero operations from the LLM must mean zero changes in the rendered output.

        This is the structural guarantee: any sections/blocks not mentioned by an
        op come through byte-identical. A no-op refresh therefore re-renders the
        same structured doc — which (after the first refresh has parsed and
        re-rendered it) is byte-stable.
        """
        bank_id = f"test-delta-noop-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nAlice is the lead.\n\n## Members\n\n- Alice\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        # First refresh: parses + renders existing into structured form. The output
        # may not match `existing` byte-for-byte (whitespace normalised by renderer).
        patch_reflect(memory, text="ignored — full mode candidate")
        patch_llm_call(memory, returns=[])
        first = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        normalised = first["content"]

        # Second refresh: zero ops again → same bytes as first refresh.
        # Must include at least one fact so the no-new-facts short-circuit doesn't fire.
        patch_reflect(
            memory,
            text="something completely different from existing",
            facts=[{"id": "obs-1", "text": "irrelevant", "type": "observation", "context": None}],
        )
        patch_llm_call(memory, returns=[])
        second = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert second["content"] == normalised
        rr = second.get("reflect_response") or {}
        assert rr.get("delta_applied") is True  # delta path ran; produced no changes
        assert rr.get("delta_operations_applied") == []

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_llm_failure_falls_back_to_candidate(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        monkeypatch,
    ):
        """When the structured-delta LLM call raises, refresh falls back to the
        candidate markdown so the user still sees a fresh synthesis instead of
        an opaque failure.
        """
        bank_id = f"test-delta-llm-fail-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nExisting.\n",
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        # Seed tracking column with a successful zero-op refresh.
        patch_reflect(memory, text="ignored")

        async def ok_call(*, messages, **kwargs):
            from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

            return DeltaOperationList()

        monkeypatch.setattr(memory._reflect_llm_config, "call", ok_call)
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        # Now the second refresh: LLM raises. Refresh must not crash; it should
        # store the candidate markdown.
        candidate = "# Team\n\nFallback candidate from reflect_async.\n"
        patch_reflect(
            memory,
            text=candidate,
            facts=[{"id": "obs-new", "text": "some new fact", "type": "observation", "context": None}],
        )

        async def boom(*, messages, **kwargs):
            raise RuntimeError("simulated provider 500")

        monkeypatch.setattr(memory._reflect_llm_config, "call", boom)
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert "Fallback candidate" in refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is False

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_empty_reflect_answer_preserves_existing_content(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """Regression: when the reflect agent returns an empty answer (small models
        sometimes hit this after exhausting tool-call retries), the refresh must
        NOT overwrite the existing content with an empty string.

        Previously this destroyed the working document on every transient upstream
        failure, and the next refresh saw current_content == "" and skipped the
        delta path entirely — a snowball that emptied valuable mental models.

        The scenario covered here is the realistic failure path: the structured
        delta call also fails (because the empty supporting facts produce empty
        / invalid JSON) so the fallback path kicks in. Without the guard, the
        fallback would write "" to the DB; with it, the existing content stays.
        """
        bank_id = f"test-empty-reflect-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nAlice is the lead.\n\n## Members\n\n- Alice\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # Reflect returns "" — this is the upstream failure mode.
        # Must include at least one fact so the no-new-facts short-circuit doesn't fire.
        patch_reflect(
            memory,
            text="",
            facts=[{"id": "obs-new", "text": "some fact", "type": "observation", "context": None}],
        )

        # Delta call also fails (mirrors the real groq behaviour where empty
        # supporting facts often produce empty / invalid JSON). Refresh then
        # falls back to the empty candidate, which the guard rejects.
        async def boom(*, messages, **kwargs):
            raise RuntimeError("simulated empty/invalid JSON from provider")

        monkeypatch.setattr(memory._reflect_llm_config, "call", boom)

        from hindsight_api.engine.memory_engine import MentalModelRefreshError

        # Empty reflect answer must now RAISE — the previous silent-preserve
        # behavior masked upstream LLM failures from workers and tests. The
        # exception is the signal; existing content + reflect_response audit
        # still get persisted before the raise so the failure is recoverable.
        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        # Existing content was preserved in the DB, and the reflect_response
        # audit trail records the skip reason — fetch directly to verify.
        preserved = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert preserved is not None
        assert preserved["content"] == existing, (
            "Empty reflect answer overwrote existing content — preserve guard regressed"
        )
        rr = preserved.get("reflect_response") or {}
        assert rr.get("refresh_skipped") == "empty_candidate"

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Real-Gemini evaluation tests
# ---------------------------------------------------------------------------

_GEMINI_API_KEY = os.getenv("HINDSIGHT_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_RUN_LLM_EVAL = os.getenv("HINDSIGHT_RUN_GEMINI_EVALS") == "1" and (bool(_GEMINI_API_KEY) or bool(_OPENAI_API_KEY))


pytestmark_gemini = pytest.mark.skipif(
    not _RUN_LLM_EVAL,
    reason=(
        "Real-LLM delta evals are gated. Set HINDSIGHT_RUN_GEMINI_EVALS=1 and provide "
        "GEMINI_API_KEY (preferred) or OPENAI_API_KEY to run."
    ),
)


@pytest.fixture
async def gemini_memory(memory_no_llm_verify: MemoryEngine):
    """MemoryEngine wired to a real LLM for reflect + structured delta.

    Prefers Gemini (the original target) but falls back to OpenAI when the
    Gemini key is unavailable — the structured-delta architecture works
    against either, and waiting on a single provider's key would block
    iteration. The chosen model is logged so test failures are unambiguous
    about which provider produced them.
    """
    if _GEMINI_API_KEY:
        provider = "gemini"
        model = os.getenv("HINDSIGHT_GEMINI_EVAL_MODEL", "gemini-2.0-flash")
        cfg = LLMConfig(provider=provider, api_key=_GEMINI_API_KEY, base_url="", model=model)
    else:
        provider = "openai"
        model = os.getenv("HINDSIGHT_OPENAI_EVAL_MODEL", "gpt-4o-mini")
        cfg = LLMConfig(provider=provider, api_key=_OPENAI_API_KEY or "", base_url="", model=model)
    print(f"\n[delta-eval] using provider={provider} model={model}")
    memory_no_llm_verify._reflect_llm_config = cfg
    memory_no_llm_verify._llm_config = cfg
    memory_no_llm_verify._retain_llm_config = cfg
    memory_no_llm_verify._consolidation_llm_config = cfg
    yield memory_no_llm_verify


_NEWS_FEED_SKILL_MARKDOWN = """## Purpose

Generate a concise, top-N personalized AI/ML news brief in response to user-triggered requests such as "ai news", "top 5 this week", or "what matters for builders today".

## Scope

- **In scope**: collecting, filtering, and summarizing AI/ML articles from user-preferred RSS feeds, applying user preferences stored in the AI News Feed Preferences mental model, and delivering the brief to the user.
- **Out of scope**: non-AI news, detailed article content, legal or privacy reviews beyond user preferences, and posting the brief to external platforms without explicit user approval.

## Rules

- **Always**:
  1. Use the AI News Feed Preferences mental model to retrieve user preferences; do not embed preferences in the skill file.
  2. Do not post the brief to any platform unless the user explicitly approves.
  3. Do not persist preferences locally; rely solely on the mental model.
  4. Refresh the feed after consolidation if the trigger-refresh-after-consolidation flag is true.
- **Prefer**:
  1. Provide a concise summary (about 2-3 sentences per article) for the top-N articles.
  2. Default to the top-5 articles unless the user specifies otherwise.
  3. Order articles chronologically or by relevance as per user preference.
  4. Highlight any user-specified topics or tags if present.

## Procedure

1. **Trigger detection** — identify a request containing keywords like "ai news", "top N", or "what matters".
2. **Preference retrieval** — call memory recall for the AI News Feed Preferences mental model to obtain RSS feed URLs and any filtering criteria.
3. **Feed consolidation** — fetch all feeds, de-duplicate entries, and apply any user-specified filters.
4. **Article selection** — choose the top-N articles based on date or user preference; if trigger-refresh-after-consolidation is true, re-fetch feeds before selection.
5. **Summarization** — generate a brief summary for each article, keeping it short and to the point.
6. **Approval check** — if the brief is to be posted externally, verify explicit user approval; otherwise, deliver it directly to the user.
7. **Memory retention** — store any new learnings or preferences observed during the task using memory retain.

## Inputs and Context

- **Source feeds**: user-specified RSS URLs stored in the mental model (e.g., https://aiagentmemory.org/index.xml).
- **Time window**: the latest update from each feed; typically the last 7 days for weekly briefs.
- **User preferences**: stored in the AI News Feed Preferences mental model; may include topics, tags, or language.

## Output Shape

- **Structure**: list of articles with title, publication date, source, and a 2-sentence summary.
- **Format**: plain text or markdown (as requested by the user).
- **Length**: concise — approximately 2-3 sentences per article; total brief about 200-300 words for top-5.
- **Voice/Tone**: neutral, informative, and concise; use bullet points for clarity.

## Stop Conditions

- If the mental model cannot be retrieved, refuse or request clarification.
- If the user has not provided any RSS feed URLs, ask for a preferred source.
- If the brief is requested for posting and explicit approval is missing, refuse.
- If the user explicitly requests to remove a skill or stop the briefing, comply immediately.

## Open Questions

- Desired brief length or word count?
- Preferred summary style (bullet vs paragraph).
- Whether the user wants to include non-AI but AI-related topics.
- Frequency or schedule for automated briefs (if any).
- Specific user-defined tags or topics to highlight.
"""


@pytestmark_gemini
@pytest.mark.hs_llm_core
class TestDeltaRefreshGeminiEval:
    """Real-LLM evals for the structured-delta refresh path.

    The structural guarantee these tests verify: sections and blocks not
    targeted by an LLM-emitted operation are byte-identical between the
    pre-refresh and post-refresh markdown render. This is what the
    structured-ops architecture buys us — the LLM cannot drift on text it
    never re-emits.

    Real Gemini is used (not a mock) because the failure mode we're guarding
    against is precisely "the LLM doesn't reliably do what the prompt says,
    even at temperature 0". Mocked output would prove the wiring works but
    not that the contract holds against an actual model.
    """

    async def _seed(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        bank_id: str,
        existing_markdown: str,
        memories: list[str],
    ) -> dict[str, Any]:
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Skill Doc",
            source_query="Document the news-feed skill: purpose, rules, procedure, stop conditions.",
            content=existing_markdown,
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": m} for m in memories],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()
        # First refresh: parses existing into structured form. With well-aligned
        # memories the LLM should emit zero ops, so the structured doc is just
        # the parsed existing content. The rendered markdown is canonicalised.
        first = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        return {"mm": mm, "first": first}

    async def test_no_change_when_observations_agree_with_existing(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """When observations only restate the existing doc, a second delta
        refresh produces output byte-identical to the first refresh's output.

        The first refresh canonicalises whitespace via the parser+renderer; we
        compare the *second* refresh against the *first* (not against the raw
        seed markdown), which is the actual repeat-refresh behaviour users
        will see in production.
        """
        bank_id = f"eval-delta-noop-{uuid.uuid4().hex[:8]}"
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=_NEWS_FEED_SKILL_MARKDOWN,
            memories=[
                "The news-feed skill produces a concise top-N AI/ML news brief.",
                "Default brief size is top 5 unless the user specifies otherwise.",
                "Source feed: https://aiagentmemory.org/index.xml.",
                "The skill must not post externally without explicit approval.",
            ],
        )
        first_content = seeded["first"]["content"]

        second = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        second_content = second["content"]

        # Byte-identical render across refreshes when no new fact has arrived.
        assert second_content == first_content, (
            "Repeat delta refresh changed bytes when no new facts arrived.\n"
            f"--- diff sample (first 300 chars different) ---\n"
            f"first:  {first_content[:300]!r}\n"
            f"second: {second_content[:300]!r}"
        )
        rr = second.get("reflect_response") or {}
        # The LLM may emit zero ops (best case) or non-effective ops (still no
        # change to render); both are acceptable so long as the bytes match.
        assert rr.get("delta_applied") is True

        await gemini_memory.delete_bank(bank_id, request_context=request_context)

    async def test_new_observation_is_merged_surgically(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """A new fact arrives; only the section relevant to it should change.

        Asserts the architectural guarantee at the section level: every
        section that the LLM did NOT name in an operation must render exactly
        the same bytes after the refresh as before. The new fact itself must
        appear somewhere in the output.
        """
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            render_section,
        )

        bank_id = f"eval-delta-add-{uuid.uuid4().hex[:8]}"
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=_NEWS_FEED_SKILL_MARKDOWN,
            memories=[
                "The news-feed skill produces a concise top-N AI/ML news brief.",
                "Default brief size is top 5.",
                "Source feed: https://aiagentmemory.org/index.xml.",
            ],
        )
        first_content = seeded["first"]["content"]
        first_struct = StructuredDocument.model_validate(
            seeded["first"]["reflect_response"]["delta_operations_applied"]
            and seeded["first"].get("structured_content")
            or {"version": 1, "sections": []}
        )
        # The first refresh's structured snapshot is what the second refresh
        # will operate on. Re-fetch via get_mental_model would also work.
        # For preservation comparison we re-parse first_content.
        from hindsight_api.engine.reflect.structured_doc import parse_markdown

        before = parse_markdown(first_content)

        # Introduce a brand-new fact that fits into "Inputs and Context" or
        # similar — but the model may pick any reasonable section.
        await gemini_memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": (
                        "The default time window for the news brief is the last 7 days, "
                        "matching the weekly cadence preferred by the user."
                    )
                },
            ],
            request_context=request_context,
        )
        await gemini_memory.wait_for_background_tasks()

        refreshed = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        content = refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        applied_ops = rr.get("delta_operations_applied") or []
        touched_section_ids = {op.get("section_id") for op in applied_ops if op.get("section_id")}

        # The fact must show up.
        assert "7 days" in content or "seven days" in content.lower(), (
            f"New fact about 7-day window missing from delta output: {content!r}"
        )

        # Every untouched section must render byte-identical to its pre-refresh form.
        after = parse_markdown(content)
        before_by_id = {s.id: s for s in before.sections}
        for section in after.sections:
            if section.id in touched_section_ids:
                continue
            orig = before_by_id.get(section.id)
            if orig is None:
                continue  # newly added section, no preservation contract
            assert render_section(orig) == render_section(section), (
                f"Untouched section {section.id!r} drifted between refreshes — the "
                f"structured-ops architecture's preservation guarantee was violated.\n"
                f"BEFORE:\n{render_section(orig)!r}\n"
                f"AFTER:\n{render_section(section)!r}"
            )

        assert rr.get("delta_applied") is True

        await gemini_memory.delete_bank(bank_id, request_context=request_context)

    async def test_no_change_repeated_three_times_stays_byte_stable(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """Three consecutive no-change refreshes must produce three identical
        markdown outputs. This is the regression test for the original
        complaint where prose-merge delta drifted content across versions even
        when no observation changed.
        """
        bank_id = f"eval-delta-stable-{uuid.uuid4().hex[:8]}"
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=_NEWS_FEED_SKILL_MARKDOWN,
            memories=[
                "The news-feed skill produces a top-N AI brief on demand.",
                "It must not post without explicit user approval.",
            ],
        )
        c1 = seeded["first"]["content"]
        r2 = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        r3 = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        assert r2["content"] == c1, "second refresh drifted vs first"
        assert r3["content"] == c1, "third refresh drifted vs first"

        await gemini_memory.delete_bank(bank_id, request_context=request_context)

    async def test_source_query_change_forces_full_rewrite(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """Changing source_query must bypass delta and produce a full regeneration."""
        bank_id = f"eval-delta-query-change-{uuid.uuid4().hex[:8]}"
        await gemini_memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await gemini_memory.create_mental_model(
            bank_id=bank_id,
            name="Subject",
            source_query="Summarize the team and how it operates.",
            content="# Team Overview\n\nAlice leads the team.\n",
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        await gemini_memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice leads the team."},
                {"content": "The product is a memory system for AI agents."},
                {"content": "Customers include small SaaS startups and enterprise pilots."},
            ],
            request_context=request_context,
        )
        await gemini_memory.wait_for_background_tasks()

        # First refresh seeds tracking column under the team query.
        await gemini_memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # Change the topic entirely.
        await gemini_memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            source_query="Summarize our customers and what we sell them.",
            request_context=request_context,
        )

        refreshed = await gemini_memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        content = refreshed["content"].lower()
        # Content should now be about customers/product, not (only) about Alice leading the team.
        assert "customer" in content or "product" in content, (
            f"Full rewrite should cover the new topic, got: {refreshed['content']!r}"
        )
        # delta_applied should be absent/False because we took the full path.
        assert (refreshed.get("reflect_response") or {}).get("delta_applied") is not True

        await gemini_memory.delete_bank(bank_id, request_context=request_context)
