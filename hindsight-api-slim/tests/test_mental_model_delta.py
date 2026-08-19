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

import os
import uuid
from typing import Any

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.response_models import ReflectResult, TokenUsage
from hindsight_api.engine.retain import embedding_utils

#: Trigger for the tests below that pin the AGENTIC delta path — the reflect loop
#: plus the structured-delta call they patch. The deterministic fast path is on by
#: default and would answer these refreshes itself off the (empty) delta window,
#: never reaching the loop, so they opt out explicitly. The fast path's own tiers
#: are covered by TestDeltaFastPath.
_AGENTIC_DELTA = {"mode": "delta", "delta_fast_path": False}


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

    @pytest.mark.memory_backend_incompatible
    async def test_delta_no_new_facts_advances_watermark_to_newest_processed(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """A successful no-op refresh advances ``last_memory_seen_at`` to the newest
        in-scope memory it actually saw — not ``now()`` — and records that it ran by
        stamping ``last_refreshed_at``.

        The scheduled-refresh gate keys off the watermark. If a no-op refresh left it
        unchanged, one unrelated memory would make every maintenance tick submit another
        LLM refresh forever. Anchoring it to the newest processed memory stops that storm
        without jumping ahead of the real data, so a row that commits later stays newer
        than the watermark (see ``test_delta_refresh_watermark_survives_straddling_commit``).
        """
        bank_id = f"test-delta-watermark-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Preferences\n\nThe user prefers concise answers.\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="User Preferences",
            source_query="What are the user's durable collaboration preferences?",
            content=existing,
            trigger={**_AGENTIC_DELTA, "refresh_cron": "* * * * *"},
            request_context=request_context,
        )

        # Established model whose cron is overdue, plus a topic-irrelevant but in-scope
        # fact committed a couple of minutes ago. The coarse staleness query sees the
        # row while the reflect agent correctly returns no supporting facts.
        assert memory._pool is not None
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
            fact_updated_at = await conn.fetchval(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at)
                VALUES ($1, $2, 'The build server uses Linux.', 'world', ARRAY[]::varchar[],
                        NOW() - INTERVAL '2 minutes', NOW() - INTERVAL '2 minutes')
                RETURNING updated_at
                """,
                uuid.uuid4(),
                bank_id,
            )
            stale_row = await conn.fetchrow(
                "SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
                "FROM mental_models WHERE bank_id = $1 AND id = $2",
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
                "SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
                "FROM mental_models WHERE bank_id = $1 AND id = $2",
                bank_id,
                mm["id"],
            )
            assert mm_row is not None
            after = mm_row["last_memory_seen_at"]
            refreshed_at = mm_row["last_refreshed_at"]
            is_stale = await memory.compute_mental_model_is_stale(conn, bank_id, mm_row)
            history_count = await conn.fetchval(
                "SELECT COUNT(*) FROM mental_model_history WHERE bank_id = $1 AND mental_model_id = $2",
                bank_id,
                mm["id"],
            )
        # Watermark advanced to the newest in-scope memory actually seen — exactly its
        # updated_at, not now() — so the settled window no longer re-triggers.
        assert after == fact_updated_at
        assert after > before
        # The refresh ran, so the wall clock says so even though nothing was written.
        assert refreshed_at > before
        assert is_stale is False
        assert history_count == 0

        submitted: list[str] = []

        async def record_submit(
            *,
            bank_id: str,
            mental_model_id: str,
            request_context: RequestContext,
            skip_if_in_flight: bool = False,
        ) -> dict[str, str]:
            submitted.append(mental_model_id)
            return {"operation_id": str(uuid.uuid4())}

        monkeypatch.setattr(memory, "submit_async_refresh_mental_model", record_submit)
        await MaintenanceLoop(memory)._run_scheduled_mm_refresh()
        assert mm["id"] not in submitted

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.memory_backend_incompatible
    async def test_delta_refresh_watermark_survives_straddling_commit(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """A memory whose transaction starts before the refresh snapshot but commits
        after it must remain visible to a later refresh.

        ``memory_units.updated_at`` is the writing transaction's start time, but the row
        only becomes visible at COMMIT. A refresh that persisted its exact snapshot
        cutoff (or ``now()``) would leave such a straddling row below the watermark — its
        start time predates the cutoff — even though reflect never saw it, dropping it
        forever. Anchoring the watermark to ``max(updated_at)`` of the rows the refresh
        *actually saw* excludes the still-uncommitted straddler, so it stays strictly
        newer than the watermark and is picked up next time.
        """
        bank_id = f"test-delta-straddle-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="User Preferences",
            source_query="What are the user's durable collaboration preferences?",
            content="# Preferences\n\nThe user prefers concise answers.\n",
            trigger={**_AGENTIC_DELTA, "refresh_cron": "* * * * *"},
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
            # A committed baseline in-scope fact. This is the newest row the refresh can
            # see, so it becomes the max(seen) watermark.
            baseline_updated_at = await conn.fetchval(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at)
                VALUES ($1, $2, 'The user is on the platform team.', 'world', ARRAY[]::varchar[],
                        NOW() - INTERVAL '2 minutes', NOW() - INTERVAL '2 minutes')
                RETURNING updated_at
                """,
                uuid.uuid4(),
                bank_id,
            )

        reflect_calls = patch_reflect(memory, text="No relevant preference changes.", facts=[])
        delta_llm_calls = patch_llm_call(memory, returns="should-not-be-called")
        original_update = memory.update_mental_model

        # Open a transaction and insert a NEWER relevant memory, but hold the commit so
        # it is invisible at the refresh snapshot. Its updated_at (transaction-start) is
        # still before the cutoff, so an exact-cutoff/now() watermark would drop it.
        straddle_conn = await memory._pool.acquire()
        straddle_tx = straddle_conn.transaction()
        await straddle_tx.start()
        straddle_fact_id = uuid.uuid4()
        await straddle_conn.execute(
            """
            INSERT INTO memory_units
                (id, bank_id, text, fact_type, tags, created_at, updated_at)
            VALUES
                ($1, $2, 'The user now prefers detailed answers.', 'world',
                 ARRAY[]::varchar[], NOW(), NOW())
            """,
            straddle_fact_id,
            bank_id,
        )

        straddle_committed = False

        async def commit_straddle_then_update(*args, **kwargs):
            nonlocal straddle_committed
            # refresh has already captured its snapshot and finished reflect. Commit the
            # previously-invisible row in this exact window, after the snapshot.
            await straddle_tx.commit()
            straddle_committed = True
            return await original_update(*args, **kwargs)

        monkeypatch.setattr(memory, "update_mental_model", commit_straddle_then_update)

        try:
            refreshed = await memory.refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mm["id"],
                request_context=request_context,
            )
        finally:
            if not straddle_committed:
                await straddle_tx.rollback()
            await memory._pool.release(straddle_conn)

        assert refreshed is not None
        assert len(reflect_calls) == 1
        assert len(delta_llm_calls) == 0
        cutoff = reflect_calls[0].get("created_before")
        assert cutoff is not None

        async with memory._pool.acquire() as conn:
            mm_row = await conn.fetchrow(
                "SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
                "FROM mental_models WHERE bank_id = $1 AND id = $2",
                bank_id,
                mm["id"],
            )
            straddle_updated_at = await conn.fetchval(
                "SELECT updated_at FROM memory_units WHERE bank_id = $1 AND id = $2",
                bank_id,
                straddle_fact_id,
            )
            assert mm_row is not None
            after = mm_row["last_memory_seen_at"]
            # Watermark advanced only to the committed baseline the refresh actually saw.
            assert after == baseline_updated_at
            # The straddler was stamped before the cutoff (an exact-cutoff/now() watermark
            # would drop it), yet it is newer than max(seen), so the model reads stale.
            assert straddle_updated_at < cutoff
            assert after < straddle_updated_at
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
            trigger=_AGENTIC_DELTA,
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

    async def test_delta_call_is_traced_and_uses_decoupled_completion_cap(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        monkeypatch,
    ):
        """The structured-delta call is attributed to the refresh trace and its
        transport cap is the decoupled config, not the document budget (#3421).

        Two regressions in one assertion set:

        - Tracing: the delta call used to run on the raw ``_reflect_llm_config``
          outside any trace context, so its LLM calls were never written to the
          trace table — the blind spot that made delta parse failures impossible
          to diagnose. It must now run inside a ``mental_model_delta_ops`` trace
          bound to the bank + mental model.
        - Completion cap: passing the document-sized ``delta_max_tokens`` as the
          provider's ``max_completion_tokens`` truncated the ops JSON on thinking
          models (reasoning tokens eat the budget), which at temperature 0 fails
          the parse deterministically forever. The transport cap must be the
          decoupled ``reflect_max_completion_tokens`` (uncapped by default),
          exactly as reflect's synthesis (#3365/#3389).
        """
        from hindsight_api.config import get_config
        from hindsight_api.engine.llm_trace import current_trace_context
        from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

        bank_id = f"test-delta-trace-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nAlice is the lead.\n\n## Members\n\n- Alice — lead\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger=_AGENTIC_DELTA,
            request_context=request_context,
        )

        # Seed the tracking column (zero ops → no change).
        patch_reflect(memory, text="ignored — full mode candidate")

        captured: dict[str, Any] = {}

        async def capturing_call(*, messages, **kwargs):
            ctx = current_trace_context()
            captured["max_completion_tokens"] = kwargs.get("max_completion_tokens")
            captured["scope"] = kwargs.get("scope")
            captured["trace_operation"] = ctx.operation if ctx else None
            captured["trace_bank_id"] = ctx.bank_id if ctx else None
            captured["trace_metadata"] = dict(ctx.metadata) if ctx else None
            return DeltaOperationList()

        # First (seeding) refresh — value captured here is overwritten by the second.
        monkeypatch.setattr(memory._reflect_llm_config, "call", capturing_call)
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        # Second refresh with a genuine new fact so the delta call actually fires.
        patch_reflect(
            memory,
            text="Alice is the lead. Bob joined.",
            facts=[{"id": "obs-bob", "text": "Bob joined the team", "type": "observation", "context": None}],
        )
        captured.clear()
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        assert captured["scope"] == "mental_model_delta_ops"
        # Transport cap is the decoupled config (None by default), NOT delta_max_tokens
        # (which would be max(2048, 2048*1.5) == 3072 for the default document budget).
        assert captured["max_completion_tokens"] == get_config().reflect_max_completion_tokens
        assert captured["max_completion_tokens"] != 3072
        # The call ran inside a trace bound to this refresh.
        assert captured["trace_operation"] == "mental_model_delta_ops"
        assert captured["trace_bank_id"] == bank_id
        assert captured["trace_metadata"] == {"mental_model_id": str(mm["id"])}

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
            trigger=_AGENTIC_DELTA,
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
            trigger=_AGENTIC_DELTA,
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

    async def test_delta_llm_failure_preserves_document_and_raises(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        monkeypatch,
    ):
        """#3112: a failed structured-delta call must never overwrite the document.

        The reflect candidate was synthesised under ``created_after`` — only the
        memories newer than the last refresh — so writing it as the whole document
        deletes everything grounded in older ones. This used to be logged as
        "falling back to full synthesis", which it never was. The refresh now
        preserves the document and fails, the same way an empty candidate does.
        """
        bank_id = f"test-delta-llm-fail-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nExisting.\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger=_AGENTIC_DELTA,
            request_context=request_context,
        )
        # Seed tracking column + structured baseline with a successful zero-op refresh.
        patch_reflect(memory, text="ignored")

        async def ok_call(*, messages, **kwargs):
            from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

            return DeltaOperationList()

        monkeypatch.setattr(memory._reflect_llm_config, "call", ok_call)
        seeded = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        seeded_content = seeded["content"]
        seeded_refreshed_at = seeded["last_refreshed_at"]
        seeded_memory_seen_at = seeded["last_memory_seen_at"]

        # Second refresh: the delta LLM call raises. The candidate is deliberately
        # a plausible-looking document — the danger is that it *is* non-empty, so
        # the empty-content guard would let it through.
        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate covering only the new fact.\n",
            facts=[{"id": "obs-new", "text": "some new fact", "type": "observation", "context": None}],
        )

        async def boom(*, messages, **kwargs):
            raise RuntimeError("simulated provider 500")

        monkeypatch.setattr(memory._reflect_llm_config, "call", boom)

        from hindsight_api.engine.memory_engine import MentalModelRefreshError

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        preserved = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert preserved is not None
        assert preserved["content"] == seeded_content, (
            "Delta failure overwrote the document with the narrow-window candidate (#3112)"
        )
        # Neither timestamp moves: the new fact has to stay inside the window the retry
        # reads, or it is lost for good, and no refresh finished to record.
        assert preserved["last_memory_seen_at"] == seeded_memory_seen_at
        assert preserved["last_refreshed_at"] == seeded_refreshed_at
        rr = preserved.get("reflect_response") or {}
        assert rr.get("refresh_skipped") == "delta_ops_failed"
        assert rr.get("delta_applied") is False

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_all_ops_skipped_preserves_document_and_raises(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """Ops that are all rejected leave the document unchanged — that is a failure.

        Persisting it would look like a clean refresh while advancing the watermark
        past facts that never reached the document, putting them outside every
        future delta window. Distinct from the model emitting *zero* ops, which is a
        legitimate "nothing to add" and is covered by the byte-identical test above.
        """
        bank_id = f"test-delta-all-skipped-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nAlice is the lead.\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger=_AGENTIC_DELTA,
            request_context=request_context,
        )

        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate.\n",
            facts=[{"id": "obs-new", "text": "Bob joined", "type": "observation", "context": None}],
        )
        # Every op targets a section that does not exist, so apply_operations
        # rejects all of them.
        patch_llm_call(
            memory,
            returns=[
                {
                    "op": "append_block",
                    "section_id": "does-not-exist",
                    "block": {"type": "paragraph", "text": "Bob joined the team."},
                },
                {
                    "op": "append_block",
                    "section_id": "also-missing",
                    "block": {"type": "paragraph", "text": "Bob sits with Alice."},
                },
            ],
        )

        from hindsight_api.engine.memory_engine import (
            MentalModelRefreshError,
            _is_non_retryable_task_error,
        )

        with pytest.raises(MentalModelRefreshError) as exc_info:
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )
        assert exc_info.value.retryable is False
        assert _is_non_retryable_task_error(exc_info.value) is True

        preserved = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert preserved is not None
        assert preserved["content"] == existing
        rr = preserved.get("reflect_response") or {}
        assert rr.get("refresh_skipped") == "delta_ops_all_skipped"
        # The rejected ops are persisted so the reason each was dropped is
        # recoverable without re-running the refresh.
        assert len(rr.get("delta_operations_skipped") or []) == 2
        assert all("unknown section_id" in op.get("reason", "") for op in rr["delta_operations_skipped"])

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_partial_skip_applies_the_rest_and_records_it(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """One bad op must not sink the whole refresh — but it must be visible.

        Most of the new facts still reach the document, so the refresh proceeds;
        the rejected op is recorded on the model so a human can see that part of
        this run's evidence never landed.
        """
        bank_id = f"test-delta-partial-skip-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nAlice is the lead.\n",
            trigger=_AGENTIC_DELTA,
            request_context=request_context,
        )
        # First refresh establishes the structured doc, so section ids are known.
        # It needs a fact: with none, the no-new-facts short-circuit preserves the
        # content without ever writing structured_content.
        patch_reflect(
            memory,
            text="ignored",
            facts=[{"id": "obs-seed", "text": "seed", "type": "observation", "context": None}],
        )
        patch_llm_call(memory, returns=[])
        seeded = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        structured = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert structured is not None
        section_id = structured["structured_content"]["sections"][0]["id"]

        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate.\n",
            facts=[{"id": "obs-new", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_llm_call(
            memory,
            returns=[
                {
                    "op": "append_block",
                    "section_id": section_id,
                    "block": {"type": "paragraph", "text": "Bob joined the team."},
                },
                {
                    "op": "append_block",
                    "section_id": "does-not-exist",
                    "block": {"type": "paragraph", "text": "Dropped on the floor."},
                },
            ],
        )
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert "Bob joined the team." in refreshed["content"]
        assert "Alice is the lead." in refreshed["content"], "surviving op must not disturb existing content"
        assert refreshed["content"] != seeded["content"]
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is True
        assert len(rr.get("delta_operations_applied") or []) == 1
        assert len(rr.get("delta_operations_skipped") or []) == 1
        assert "refresh_skipped" not in rr

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_unusable_structured_content_rebuilds_baseline_from_markdown(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """A corrupt structured_content column must not disable delta forever.

        The markdown in ``content`` is the same document and parses leniently, so
        the baseline is re-derived from it and the refresh proceeds — repairing
        structured_content on the way. Failing instead would wedge the model:
        nothing else rewrites that column.
        """
        bank_id = f"test-delta-bad-struct-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nAlice is the lead.\n",
            trigger=_AGENTIC_DELTA,
            request_context=request_context,
        )
        # Valid JSON, wrong shape — what a schema change or a hand edit leaves behind.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            structured_content={"not_a_document": True},
            request_context=request_context,
        )

        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate.\n",
            facts=[{"id": "obs-new", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_llm_call(memory, returns=[])
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # Delta ran against the markdown-derived baseline: the existing content
        # survives (it was not replaced by the narrow candidate) and the
        # structured column is valid again.
        assert "Alice is the lead." in refreshed["content"]
        assert "Narrow candidate" not in refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is True
        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored is not None
        assert stored["structured_content"]["sections"]

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_unparseable_baseline_preserves_document_and_raises(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """With no readable baseline at all, delta has nothing to edit — so it fails.

        This is the second half of the #3112 guard: the candidate is just as narrow
        here as it is after an LLM failure, so it is refused for the same reason.
        """
        bank_id = f"test-delta-no-baseline-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = "# Team\n\nAlice is the lead.\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger=_AGENTIC_DELTA,
            request_context=request_context,
        )

        from hindsight_api.engine.reflect import structured_doc

        def unparseable(_markdown: str):
            raise ValueError("simulated unparseable markdown")

        monkeypatch.setattr(structured_doc, "parse_markdown", unparseable)

        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate.\n",
            facts=[{"id": "obs-new", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_llm_call(memory, returns=[])

        from hindsight_api.engine.memory_engine import MentalModelRefreshError

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        preserved = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert preserved is not None
        assert preserved["content"] == existing
        rr = preserved.get("reflect_response") or {}
        assert rr.get("refresh_skipped") == "structured_doc_unreadable"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_full_mode_candidate_is_still_written(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """The #3112 guard must not touch full mode.

        A full-mode candidate is synthesised over the whole history, so it IS the
        document — there is no narrowing to protect against, and refusing it would
        break the ordinary refresh path.
        """
        bank_id = f"test-full-mode-writes-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nAlice is the lead.\n",
            trigger={"mode": "full"},
            request_context=request_context,
        )

        calls = patch_reflect(
            memory,
            text="# Team\n\nFull rewrite over the whole history.\n",
            facts=[{"id": "obs-new", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_llm_call(memory, returns=[])
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert "created_after" not in calls[0], "full mode must not narrow the reflect window"
        assert "Full rewrite over the whole history." in refreshed["content"]

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
            trigger=_AGENTIC_DELTA,
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
# Deterministic delta fast path
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_window_facts(monkeypatch):
    """Patch the two typed retrieval calls the delta fast path makes itself.

    The fast path reads the delta window directly instead of letting the reflect
    agent's tools do it, so what that read returns is exactly what decides tier 0
    from tier 1. Patching the two wrappers makes that decision deterministic
    without seeding embeddings; ``test_tier1_over_real_retrieval`` covers the
    unpatched wiring end to end.

    These are the same module-level names ``reflect_async`` binds its tool
    callbacks to, which is harmless here: every test using this fixture also
    patches ``reflect_async`` itself.
    """
    from hindsight_api.engine import memory_engine as engine_module

    def _install(
        memory: MemoryEngine,
        *,
        memories: list[dict] | None = None,
        observations: list[dict] | None = None,
    ) -> list[dict]:
        calls: list[dict] = []

        async def fake_recall(_engine, bank_id, query, request_context, **kwargs):
            calls.append({"tool": "recall", "bank_id": bank_id, "query": query, **kwargs})
            return {"query": query, "memories": list(memories or []), "chunks": {}}

        async def fake_search_observations(_engine, bank_id, query, request_context, **kwargs):
            calls.append({"tool": "search_observations", "bank_id": bank_id, "query": query, **kwargs})
            return {
                "query": query,
                "count": len(observations or []),
                "observations": list(observations or []),
                "source_facts": {},
                "is_stale": False,
                "freshness": "up_to_date",
            }

        monkeypatch.setattr(engine_module, "tool_recall", fake_recall)
        monkeypatch.setattr(engine_module, "tool_search_observations", fake_search_observations)
        return calls

    return _install


@pytest.fixture
def patch_delta_llm_calls(monkeypatch):
    """Patch the structured-delta LLM call with one canned response per call.

    Differs from ``patch_llm_call`` in two ways the fast path needs: responses are
    consumed in order (so a test can make the fast path decline and then let the
    agentic path succeed), and ``return_usage=True`` is honoured, since tier 1
    asks for its own call's usage. Each recorded call also carries the trace
    attribution that was bound around it — that is what ends up in the
    ``llm_requests`` operation column.
    """

    def _install(memory: MemoryEngine, *, responses: list) -> list[dict]:
        calls: list[dict] = []
        queued = list(responses)
        assert queued, "at least one canned response is required"

        async def fake_call(*, messages, **kwargs):
            from hindsight_api.engine.llm_trace import current_trace_context

            trace_ctx = current_trace_context()
            calls.append(
                {
                    "messages": messages,
                    "operation": trace_ctx.operation if trace_ctx else None,
                    "trace_bank_id": trace_ctx.bank_id if trace_ctx else None,
                    **kwargs,
                }
            )
            response = queued.pop(0) if len(queued) > 1 else queued[0]
            if isinstance(response, Exception):
                raise response
            if kwargs.get("return_usage"):
                return response, TokenUsage(input_tokens=1200, output_tokens=90, total_tokens=1290)
            return response

        monkeypatch.setattr(memory._reflect_llm_config, "call", fake_call)
        return calls

    return _install


async def _age_watermark_and_seed_fact(
    memory: MemoryEngine,
    bank_id: str,
    mental_model_id: str,
    *,
    text: str = "The build server runs Linux.",
):
    """Age the model's watermark by a day and commit one in-scope fact.

    Gives the refresh a real delta window (so ``created_after`` is set and the
    persisted watermark is the fact's ``updated_at`` rather than the model's own
    prior watermark), and records ``last_refreshed_source_query`` so the mode
    decision stays in delta. Returns the fact's ``updated_at``.

    Both timestamps are aged: since #3538 the watermark lives in
    ``last_memory_seen_at`` and ``last_refreshed_at`` is the wall-clock time of
    the last refresh, and the window reads ``COALESCE(last_memory_seen_at,
    last_refreshed_at)`` — so ageing only one of them leaves the window's origin
    depending on which column the row happens to have stamped.
    """
    assert memory._pool is not None
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE mental_models
            SET last_refreshed_at = NOW() - INTERVAL '1 day',
                last_memory_seen_at = NOW() - INTERVAL '1 day',
                last_refreshed_source_query = source_query
            WHERE bank_id = $1 AND id = $2
            """,
            bank_id,
            mental_model_id,
        )
        return await conn.fetchval(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at)
            VALUES ($1, $2, $3, 'world', ARRAY[]::varchar[],
                    NOW() - INTERVAL '2 minutes', NOW() - INTERVAL '2 minutes')
            RETURNING updated_at
            """,
            uuid.uuid4(),
            bank_id,
            text,
        )


_APPEND_BOB_OP = {
    "op": "append_block",
    "section_id": "members",
    "block": {"type": "bullet_list", "items": ["Bob — junior engineer"]},
}
_NEW_OBSERVATION = {
    "id": "obs-bob",
    "text": "Bob joined the team as junior engineer",
    "fact_type": "observation",
    "context": None,
}


class TestDeltaFastPath:
    """Delta refreshes that never reach the agentic loop.

    Tier 0 reads the window and finds nothing new — no LLM call at all. Tier 1
    turns what it found into edit operations with exactly one. The whole point is
    a negative (the loop did not run), so these assert call counts on the mocks
    rather than only inspecting the document, which would pass just as happily if
    the loop had produced it.
    """

    SOURCE_QUERY = "Tell me about the team"
    BASELINE = "# Team\n\nAlice is the lead.\n\n## Members\n\n- Alice — lead\n"

    async def _delta_model(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        bank_id: str,
        *,
        trigger: dict | None = None,
        content: str | None = None,
    ) -> dict:
        await memory.get_bank_profile(bank_id, request_context=request_context)
        return await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query=self.SOURCE_QUERY,
            content=self.BASELINE if content is None else content,
            trigger={"mode": "delta"} if trigger is None else trigger,
            request_context=request_context,
        )

    # -- tier 1 ------------------------------------------------------------

    async def test_tier1_edits_the_document_in_one_call_without_reflect(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """The decisive assertion: pending facts, one LLM call, no reflect loop.

        Everything else here (ops applied, content updated, watermark advanced,
        history written) already held on the agentic path — the change is that
        reaching it costs one call instead of a multi-call loop.
        """
        bank_id = f"test-fastpath-tier1-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        fact_updated_at = await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])

        reflect_calls = patch_reflect(memory, text="MUST NOT BE USED")
        retrieval = patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        llm_calls = patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert reflect_calls == [], "the agentic reflect loop must not run on the fast path"
        assert len(llm_calls) == 1, "tier 1 is exactly one LLM call"
        assert {call["tool"] for call in retrieval} == {"recall", "search_observations"}

        assert "Bob — junior engineer" in refreshed["content"]
        assert "Alice is the lead." in refreshed["content"], "untouched sections come through unchanged"
        rr = refreshed["reflect_response"]
        assert rr["fast_path"] == "tier1"
        assert rr["fast_path_fallback_reason"] is None
        assert rr["delta_applied"] is True
        assert [op["op"] for op in rr["delta_operations_applied"]] == ["append_block"]

        # The single call carries the document and the window's facts — not a
        # reflect synthesis, which is the call being skipped.
        user_msg = llm_calls[0]["messages"][1]["content"]
        assert "obs-bob" in user_msg
        assert '"members"' in user_msg

        # get_mental_model renders timestamps as ISO strings. The watermark is
        # last_memory_seen_at since #3538 split it out of last_refreshed_at.
        assert refreshed["last_memory_seen_at"] == fact_updated_at.isoformat(), (
            "the watermark must advance on a tier-1 write"
        )
        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert len(history) == 1, "a tier-1 write is recorded in history like any other content write"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tier1_over_real_retrieval(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_delta_llm_calls,
    ):
        """The same path with nothing between the fast path and the database.

        The other tier-1 tests patch the two retrieval wrappers to keep the tier
        decision deterministic; this one retains a real memory and lets the fast
        path find it, so the scope, window and budget wiring is exercised rather
        than assumed.
        """
        bank_id = f"test-fastpath-real-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
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
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Bob joined the team as a junior engineer on the platform squad."}],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        reflect_calls = patch_reflect(memory, text="MUST NOT BE USED")
        llm_calls = patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert reflect_calls == [], "the agentic reflect loop must not run on the fast path"
        assert len(llm_calls) == 1
        assert refreshed["reflect_response"]["fast_path"] == "tier1"
        assert "Bob — junior engineer" in refreshed["content"]
        # The retained fact reached the prompt through the real retrieval path.
        assert "Bob joined the team" in llm_calls[0]["messages"][1]["content"]

        await memory.delete_bank(bank_id, request_context=request_context)

    # -- tier 0 ------------------------------------------------------------

    async def test_tier0_costs_no_llm_call_and_advances_the_watermark(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """An empty window is answered for free.

        Before the fast path this same outcome cost a full agentic loop first, run
        only to discover there was nothing to write.
        """
        bank_id = f"test-fastpath-tier0-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id, trigger={"mode": "delta", "keep_trace": True})
        fact_updated_at = await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])

        reflect_calls = patch_reflect(memory, text="MUST NOT BE USED")
        patch_window_facts(memory)  # nothing in the window
        llm_calls = patch_delta_llm_calls(memory, responses=["MUST NOT BE CALLED"])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert reflect_calls == [], "tier 0 must not run the reflect loop"
        assert llm_calls == [], "tier 0 must make no LLM call at all"

        assert refreshed["content"] == self.BASELINE, "content is preserved byte for byte"
        rr = refreshed["reflect_response"]
        assert rr["fast_path"] == "tier0"
        assert rr["delta_applied"] is False
        assert rr["delta_skipped_reason"] == "no_new_facts"

        async with memory._pool.acquire() as conn:
            newest_in_scope = await conn.fetchval(
                "SELECT MAX(updated_at) FROM memory_units WHERE bank_id = $1", bank_id
            )
        assert newest_in_scope == fact_updated_at
        assert refreshed["last_memory_seen_at"] == newest_in_scope.isoformat()

        history = await memory.get_mental_model_history(bank_id, mm["id"], request_context=request_context)
        assert history == [], "preserving content is not a new version"

        await memory.delete_bank(bank_id, request_context=request_context)

    # -- usage, trace and attribution --------------------------------------

    async def test_trace_and_usage_are_coherent_on_both_tiers(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """keep_trace on a fast-path refresh records what it actually did.

        Tier 0 books no LLM call and no tokens; tier 1 books exactly its own call.
        Both record the retrieval they performed, which is the line that answers
        "why did my refresh not pick up my memory" — the reason the trace exists.
        """
        bank_id = f"test-fastpath-trace-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id, trigger={"mode": "delta", "keep_trace": True})
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])
        patch_reflect(memory, text="MUST NOT BE USED")

        patch_window_facts(memory)
        patch_delta_llm_calls(memory, responses=["MUST NOT BE CALLED"])
        tier0 = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        trace = tier0["reflect_response"]["trace"]
        assert trace["fast_path"] == "tier0"
        assert trace["effective_mode"] == "delta"
        assert trace["outcome"] == "content_preserved_no_new_facts"
        assert trace["llm_calls"] == []
        assert trace["usage"]["total_tokens"] == 0
        assert [tc["tool"] for tc in trace["tool_calls"]] == ["recall", "search_observations"]
        assert all(tc["result_count"] == 0 for tc in trace["tool_calls"])
        assert all(tc["updated_at"] is not None for tc in trace["tool_calls"]), (
            "both fast-path fetches are window-bounded, so the trace must show the bound"
        )

        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])
        tier1 = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        trace = tier1["reflect_response"]["trace"]
        assert trace["fast_path"] == "tier1"
        assert trace["outcome"] == "content_written"
        assert [lc["scope"] for lc in trace["llm_calls"]] == ["mental_model_delta_ops"]
        assert trace["usage"]["total_tokens"] == 1290, "tier 1 books exactly one call's usage"
        assert trace["delta_operations"]["applied"], "the operations it applied are on the trace"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_ops_call_is_attributed_to_the_refresh_operation(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """The delta call must not log with a blank operation.

        It used to: reflect's trace context is already reset by the time the
        agentic path makes it, and a bare provider call binds none of its own, so
        every structured-delta request landed in llm_requests unattributed and
        uncountable. Both routes now bind a label of their own — the fast path
        books its single call under the refresh operation itself, and the agentic
        route under the ``mental_model_delta_ops`` label #3424 gave it.
        """
        bank_id = f"test-fastpath-label-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])
        patch_reflect(memory, text="MUST NOT BE USED")
        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        llm_calls = patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])

        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
        assert llm_calls[0]["operation"] == "refresh_mental_model"
        assert llm_calls[0]["trace_bank_id"] == bank_id
        assert llm_calls[0]["scope"] == "mental_model_delta_ops"

        # Same guarantee on the agentic route, which makes the same call.
        agentic_bank = f"test-agentic-label-{uuid.uuid4().hex[:8]}"
        agentic_mm = await self._delta_model(memory, request_context, agentic_bank, trigger=_AGENTIC_DELTA)
        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate.\n",
            facts=[{"id": "obs-new", "text": "Bob joined", "type": "observation", "context": None}],
        )
        agentic_calls = patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])
        await memory.refresh_mental_model(
            bank_id=agentic_bank, mental_model_id=agentic_mm["id"], request_context=request_context
        )
        # #3424 binds the agentic delta-ops call under its own label so the
        # row is countable after reflect's trace context has already reset.
        assert agentic_calls[0]["operation"] == "mental_model_delta_ops"
        assert agentic_calls[0]["scope"] == "mental_model_delta_ops"

        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.delete_bank(agentic_bank, request_context=request_context)

    # -- handing back to the agentic loop ----------------------------------

    async def test_needs_full_context_hands_back_to_the_reflect_loop(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """The escape hatch: the model says the window alone is not enough.

        The fast path trades the loop's retrieval for one call, so the model has
        to be able to say it needed that retrieval — otherwise the trade would be
        paid for in silently worse edits.
        """
        bank_id = f"test-fastpath-escape-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])

        reflect_calls = patch_reflect(
            memory,
            text="# Team\n\nSynthesis from the full loop.\n",
            facts=[{"id": "obs-bob", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        llm_calls = patch_delta_llm_calls(
            memory,
            responses=['{"operations": [], "needs_full_context": true}', {"operations": [_APPEND_BOB_OP]}],
        )

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(reflect_calls) == 1, "declining must reach the agentic loop"
        assert len(llm_calls) == 2, "the fast path's call, then the loop's own delta call"
        rr = refreshed["reflect_response"]
        assert rr["fast_path"] is None
        assert rr["fast_path_fallback_reason"] == "needs_full_context"
        assert rr["delta_applied"] is True
        assert "Bob — junior engineer" in refreshed["content"]

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_all_ops_invalid_hands_back_without_regenerating(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """Unparseable operations are the loop's problem, not a reason to rewrite.

        The refresh stays in delta: a full regenerate would read the unbounded
        window and replace the document, which is a much larger action than the
        one that just failed.
        """
        bank_id = f"test-fastpath-invalid-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])

        reflect_calls = patch_reflect(
            memory,
            text="# Team\n\nSynthesis from the full loop.\n",
            facts=[{"id": "obs-bob", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        patch_delta_llm_calls(
            memory,
            responses=[
                {"operations": [{"op": "not_an_operation", "section_id": "members"}]},
                {"operations": [_APPEND_BOB_OP]},
            ],
        )

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        rr = refreshed["reflect_response"]
        assert rr["fast_path"] is None
        assert rr["fast_path_fallback_reason"] == "delta_ops_invalid"
        assert rr["delta_applied"] is True
        assert reflect_calls[0].get("created_after") is not None, (
            "the hand-off must stay a delta refresh, not become a full regenerate"
        )

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_all_ops_skipped_hands_back_without_regenerating(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """Operations that all bounce leave the document untouched — the loop retries.

        Distinct from the invalid case above: these parsed fine and were rejected
        when applied, which is the signature of a model editing against section
        ids it could not see.
        """
        bank_id = f"test-fastpath-skipped-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])

        reflect_calls = patch_reflect(
            memory,
            text="# Team\n\nSynthesis from the full loop.\n",
            facts=[{"id": "obs-bob", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        missing_section = {
            "op": "append_block",
            "section_id": "does-not-exist",
            "block": {"type": "paragraph", "text": "Bob joined the team."},
        }
        patch_delta_llm_calls(memory, responses=[{"operations": [missing_section]}, {"operations": [_APPEND_BOB_OP]}])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        rr = refreshed["reflect_response"]
        assert rr["fast_path"] is None
        assert rr["fast_path_fallback_reason"] == "delta_ops_all_skipped"
        assert reflect_calls[0].get("created_after") is not None
        assert "Bob — junior engineer" in refreshed["content"]

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_hand_back_into_a_failing_loop_preserves_the_watermark(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """Declining must not cost the facts it declined on.

        Once the fast path hands back, the refresh is the agentic one in every
        respect — including that a failure preserves both the document and
        ``last_refreshed_at``, so the retry reads the same window rather than
        skipping past facts that never landed.
        """
        bank_id = f"test-fastpath-preserve-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])
        before = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)

        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate covering only the new fact.\n",
            facts=[{"id": "obs-bob", "text": "Bob joined", "type": "observation", "context": None}],
        )
        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        patch_delta_llm_calls(
            memory,
            responses=['{"operations": [], "needs_full_context": true}', RuntimeError("simulated provider 500")],
        )

        from hindsight_api.engine.memory_engine import MentalModelRefreshError

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        preserved = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert preserved["content"] == before["content"]
        assert preserved["last_refreshed_at"] == before["last_refreshed_at"]
        rr = preserved["reflect_response"]
        assert rr["refresh_skipped"] == "delta_ops_failed"
        assert rr["fast_path_fallback_reason"] == "needs_full_context", "the ledger must still explain why the loop ran"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_unreadable_baseline_hands_back_and_the_reason_survives_the_trace(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
        monkeypatch,
    ):
        """The one hand-back that happens before any LLM call, end to end.

        Also the regression test for the reason vocabulary: ``keep_trace`` builds
        ``MentalModelRefreshTrace`` — a Pydantic model whose
        ``fast_path_fallback_reason`` is a Literal — for every refresh, failing
        ones included, and it does so before the outcome is dispatched. A reason
        the engine can emit but the Literal does not list therefore turns a
        legible refresh failure into a ValidationError from the trace builder,
        with the real cause nowhere in it.
        """
        bank_id = f"test-fastpath-nobaseline-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id, trigger={"mode": "delta", "keep_trace": True})
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])

        from hindsight_api.engine.reflect import structured_doc

        def unparseable(_markdown: str):
            raise ValueError("simulated unparseable markdown")

        monkeypatch.setattr(structured_doc, "parse_markdown", unparseable)

        patch_reflect(
            memory,
            text="# Team\n\nNarrow candidate covering only the new fact.\n",
            facts=[{"id": "obs-bob", "text": "Bob joined", "type": "observation", "context": None}],
        )
        delta_calls = patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        llm_calls = patch_delta_llm_calls(memory, responses=["{}"])

        from hindsight_api.engine.memory_engine import MentalModelRefreshError

        with pytest.raises(MentalModelRefreshError):
            await memory.refresh_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )

        # Declined before spending anything: no window read, no delta call.
        assert delta_calls == []
        assert llm_calls == []

        preserved = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        rr = preserved["reflect_response"]
        assert rr["fast_path"] is None
        assert rr["fast_path_fallback_reason"] == "no_delta_baseline"
        assert rr["trace"]["fast_path_fallback_reason"] == "no_delta_baseline"
        # The mode fallback is the agentic path's own verdict on the same
        # baseline, recorded separately because the two answer different
        # questions: which route ran, and which mode it ran in.
        assert rr["trace"]["mode_fallback_reason"] == "structured_doc_unreadable"

        await memory.delete_bank(bank_id, request_context=request_context)

    # -- when the fast path is not consulted at all ------------------------

    async def test_full_mode_never_reaches_the_fast_path(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """Full mode regenerates the whole document, which edit ops cannot express."""
        bank_id = f"test-fastpath-fullmode-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id, trigger={"mode": "full"})
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])

        reflect_calls = patch_reflect(memory, text="# Team\n\nRegenerated from scratch.")
        retrieval = patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        llm_calls = patch_delta_llm_calls(memory, responses=["MUST NOT BE CALLED"])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(reflect_calls) == 1
        assert retrieval == [], "full mode must not run the fast path's window read"
        assert llm_calls == []
        assert refreshed["content"] == "# Team\n\nRegenerated from scratch."
        assert refreshed["reflect_response"]["fast_path"] is None
        assert refreshed["reflect_response"]["fast_path_fallback_reason"] is None

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_mode_fallbacks_are_decided_before_the_fast_path(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """No baseline and a changed topic still resolve to full mode first.

        Both mean there is nothing to edit surgically, so the fast path is never
        consulted — it is strictly a route within delta, not a new mode decision.
        """
        no_baseline_bank = f"test-fastpath-nobase-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, no_baseline_bank, content="")
        patch_reflect(memory, text="# Team\n\nFull fresh synthesis.")
        retrieval = patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        patch_delta_llm_calls(memory, responses=["MUST NOT BE CALLED"])
        refreshed = await memory.refresh_mental_model(
            bank_id=no_baseline_bank, mental_model_id=mm["id"], request_context=request_context
        )
        assert retrieval == []
        assert refreshed["reflect_response"]["fast_path"] is None

        changed_bank = f"test-fastpath-querychg-{uuid.uuid4().hex[:8]}"
        changed_mm = await self._delta_model(memory, request_context, changed_bank)
        await memory.update_mental_model(
            changed_bank,
            changed_mm["id"],
            last_refreshed_source_query="A completely different question",
            request_context=request_context,
        )
        patch_reflect(memory, text="# Team\n\nBrand new topic.")
        retrieval = patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        refreshed = await memory.refresh_mental_model(
            bank_id=changed_bank, mental_model_id=changed_mm["id"], request_context=request_context
        )
        assert retrieval == []
        assert refreshed["reflect_response"]["fast_path"] is None

        await memory.delete_bank(no_baseline_bank, request_context=request_context)
        await memory.delete_bank(changed_bank, request_context=request_context)

    # -- kill switches -----------------------------------------------------

    async def test_bank_config_can_switch_the_fast_path_off(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """The knob is hierarchical, so one bank can opt out without a redeploy."""
        bank_id = f"test-fastpath-bankoff-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])
        await memory.update_bank_config(
            bank_id, {"mental_model_delta_fast_path": False}, request_context=request_context
        )

        reflect_calls = patch_reflect(
            memory,
            text="# Team\n\nSynthesis from the full loop.\n",
            facts=[{"id": "obs-bob", "text": "Bob joined", "type": "observation", "context": None}],
        )
        retrieval = patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(reflect_calls) == 1
        assert retrieval == []
        rr = refreshed["reflect_response"]
        assert rr["fast_path"] is None
        assert rr["fast_path_fallback_reason"] is None, "never consulted is not the same as declined"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_trigger_overrides_the_resolved_default_in_both_directions(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """Per-model beats per-bank, off and on."""
        off_bank = f"test-fastpath-trigoff-{uuid.uuid4().hex[:8]}"
        off_mm = await self._delta_model(memory, request_context, off_bank, trigger=_AGENTIC_DELTA)
        await _age_watermark_and_seed_fact(memory, off_bank, off_mm["id"])
        reflect_calls = patch_reflect(
            memory,
            text="# Team\n\nSynthesis from the full loop.\n",
            facts=[{"id": "obs-bob", "text": "Bob joined", "type": "observation", "context": None}],
        )
        retrieval = patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])
        refreshed = await memory.refresh_mental_model(
            bank_id=off_bank, mental_model_id=off_mm["id"], request_context=request_context
        )
        assert len(reflect_calls) == 1, "trigger false must win over the default-on knob"
        assert retrieval == []
        assert refreshed["reflect_response"]["fast_path"] is None

        on_bank = f"test-fastpath-trigon-{uuid.uuid4().hex[:8]}"
        on_mm = await self._delta_model(
            memory, request_context, on_bank, trigger={"mode": "delta", "delta_fast_path": True}
        )
        await _age_watermark_and_seed_fact(memory, on_bank, on_mm["id"])
        await memory.update_bank_config(
            on_bank, {"mental_model_delta_fast_path": False}, request_context=request_context
        )
        reflect_calls = patch_reflect(memory, text="MUST NOT BE USED")
        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        llm_calls = patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])
        refreshed = await memory.refresh_mental_model(
            bank_id=on_bank, mental_model_id=on_mm["id"], request_context=request_context
        )
        assert reflect_calls == [], "trigger true must win over a bank that switched it off"
        assert len(llm_calls) == 1
        assert refreshed["reflect_response"]["fast_path"] == "tier1"

        await memory.delete_bank(off_bank, request_context=request_context)
        await memory.delete_bank(on_bank, request_context=request_context)

    def test_env_var_controls_the_default(self, monkeypatch):
        """The server-level default reads from the environment.

        Asserted against ``HindsightConfig.from_env`` rather than a live refresh:
        the resolver snapshots the global config when the engine is built, so a
        mid-test setenv would prove nothing about the engine already running.
        """
        from hindsight_api.config import ENV_MENTAL_MODEL_DELTA_FAST_PATH, HindsightConfig

        monkeypatch.delenv(ENV_MENTAL_MODEL_DELTA_FAST_PATH, raising=False)
        assert HindsightConfig.from_env().mental_model_delta_fast_path is True

        monkeypatch.setenv(ENV_MENTAL_MODEL_DELTA_FAST_PATH, "false")
        assert HindsightConfig.from_env().mental_model_delta_fast_path is False

        monkeypatch.setenv(ENV_MENTAL_MODEL_DELTA_FAST_PATH, "true")
        assert HindsightConfig.from_env().mental_model_delta_fast_path is True

    def test_trigger_accepts_true_false_and_none(self):
        """The per-model override is tri-state, and validation is otherwise untouched."""
        from hindsight_api.api.http import MentalModelTrigger

        assert MentalModelTrigger().delta_fast_path is None
        assert MentalModelTrigger(delta_fast_path=True).delta_fast_path is True
        assert MentalModelTrigger(delta_fast_path=False).delta_fast_path is False

        with pytest.raises(ValueError):
            MentalModelTrigger(refresh_after_consolidation=True, refresh_cron="0 3 * * *")

    # -- dry run -----------------------------------------------------------

    async def test_dry_run_previews_both_tiers_and_persists_nothing(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_window_facts,
        patch_delta_llm_calls,
    ):
        """A preview that skipped the fast path would stop predicting the refresh.

        Also pins what ``candidate_content`` means here: the fast path has no
        synthesis step, so it reports the document it would write — the current
        content on tier 0, the post-operation document on tier 1.
        """
        bank_id = f"test-fastpath-dryrun-{uuid.uuid4().hex[:8]}"
        mm = await self._delta_model(memory, request_context, bank_id)
        await _age_watermark_and_seed_fact(memory, bank_id, mm["id"])
        before = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        reflect_calls = patch_reflect(memory, text="MUST NOT BE USED")

        patch_window_facts(memory, observations=[_NEW_OBSERVATION])
        patch_delta_llm_calls(memory, responses=[{"operations": [_APPEND_BOB_OP]}])
        tier1 = await memory.dry_run_refresh_mental_model(bank_id, mm["id"], request_context=request_context)
        assert tier1.fast_path == "tier1"
        assert tier1.fast_path_fallback_reason is None
        assert tier1.effective_mode == "delta"
        assert tier1.outcome == "content_written"
        assert tier1.would_persist is True
        assert "Bob — junior engineer" in tier1.preview_content
        assert tier1.candidate_content == tier1.preview_content
        assert tier1.diff, "a preview that changes the document must show a diff"

        patch_window_facts(memory)
        patch_delta_llm_calls(memory, responses=["MUST NOT BE CALLED"])
        tier0 = await memory.dry_run_refresh_mental_model(bank_id, mm["id"], request_context=request_context)
        assert tier0.fast_path == "tier0"
        assert tier0.outcome == "content_preserved_no_new_facts"
        assert tier0.would_persist is False
        assert tier0.candidate_content == tier0.current_content
        assert tier0.diff == ""

        assert reflect_calls == [], "neither preview may run the agentic loop"
        after = await memory.get_mental_model(bank_id, mm["id"], request_context=request_context)
        assert after["content"] == before["content"]
        assert after["last_refreshed_at"] == before["last_refreshed_at"]

        await memory.delete_bank(bank_id, request_context=request_context)

    # -- prompt -------------------------------------------------------------

    def test_fast_path_prompt_extends_the_shared_one_without_changing_it(self):
        """The agentic route's prompt must stay byte-identical.

        It shares the structured-delta system prompt but does not read
        ``needs_full_context``, so a model answering "I cannot do this properly"
        there would be ignored and its empty op list written anyway. The escape
        hatch is therefore an addendum, not an edit.
        """
        from hindsight_api.engine.reflect.prompts import (
            STRUCTURED_DELTA_FAST_PATH_SYSTEM_PROMPT,
            STRUCTURED_DELTA_SYSTEM_PROMPT,
        )

        assert STRUCTURED_DELTA_FAST_PATH_SYSTEM_PROMPT.startswith(STRUCTURED_DELTA_SYSTEM_PROMPT)
        assert "needs_full_context" not in STRUCTURED_DELTA_SYSTEM_PROMPT
        assert "needs_full_context" in STRUCTURED_DELTA_FAST_PATH_SYSTEM_PROMPT


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
