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
import re
import uuid
from typing import Any

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.response_models import ReflectResult
from hindsight_api.engine.retain import embedding_utils
from tests.conftest import stub_refresh_has_sources


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

    def _install(memory: MemoryEngine, *, text: str, facts: list[dict] | None = None, document: Any = None):
        calls: list[dict] = []

        async def fake_reflect_async(**kwargs):
            calls.append(kwargs)
            result = _canned_reflect_result(text, facts)
            # ``document`` mirrors what the real agent returns in document mode.
            result.document = document
            return result

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)
        stub_refresh_has_sources(monkeypatch, memory)
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


async def _seed_fact_row(memory: MemoryEngine, bank_id: str, text: str) -> str:
    """Insert one real memory and return its id.

    Canned ``based_on`` entries have to name rows that actually exist: a refresh now
    checks its stored grounding against the live memories and treats a document whose
    citations resolve to nothing as a restored/copied bank rather than as evidence
    about this one (see ``reflect.retractions``). Synthetic ids would trip that.
    """
    from types import SimpleNamespace

    from hindsight_api.engine.memories import get_memories

    store = get_memories()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        unit_ids = await store.insert_facts(
            conn=conn,
            ops=memory._backend.ops,
            bank_id=bank_id,
            facts=[
                SimpleNamespace(
                    fact_text=text,
                    embedding=memory.embeddings.encode([text])[0],
                    fact_type="observation",
                    tags=[],
                    context=None,
                    document_id=None,
                    chunk_id=None,
                    metadata=None,
                    observation_scopes=None,
                    entities=[],
                    causal_relations=[],
                    occurred_start=None,
                    occurred_end=None,
                    mentioned_at=None,
                )
            ],
            document_id=None,
        )
    return unit_ids[0]


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
        assert refreshed["content"] == "# Team\n\nRegenerated from scratch.\n"
        assert len(llm_calls) == 0, "Delta merge LLM call must not happen in full mode"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_creating_a_model_with_markdown_stores_its_structure(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
    ):
        """A model authored as markdown is on the structured schema immediately.

        Leaving ``structured_content`` NULL until the first delta refresh would
        mean that refresh silently reshapes a document nobody asked it to touch —
        and there would be a window where the two columns describe different
        documents, which is what #3361 was.
        """
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            render_document,
        )

        bank_id = f"test-mm-create-structure-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        authored = "## Ops\n\n| Name | Role |\n|---|---|\n| Alice | Lead\n\n- top\n  - nested\n"
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=authored,
            request_context=request_context,
        )

        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored is not None
        assert stored["structured_content"] is not None, "a created model must carry its structure"
        doc = StructuredDocument.model_validate(stored["structured_content"])
        assert stored["content"] == render_document(doc)
        # The authored markdown survives the split: v1 flattened all three of these.
        lines = stored["content"].splitlines()
        assert "| Alice | Lead" in lines
        assert "  - nested" in lines
        assert [s.id for s in doc.sections] == ["ops"]

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_updating_content_alone_still_stores_a_matching_structure(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
    ):
        """``content`` and ``structured_content`` can never be written out of step.

        A caller handing over markdown alone must not leave the previous
        document's structure behind it — a delta refresh would then edit a
        document that is no longer what the column says.
        """
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            render_document,
        )

        bank_id = f"test-mm-update-structure-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="## Old\n\nOriginal.\n",
            request_context=request_context,
        )

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            content="## New\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n",
            request_context=request_context,
        )

        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored is not None
        doc = StructuredDocument.model_validate(stored["structured_content"])
        assert stored["content"] == render_document(doc)
        assert [s.id for s in doc.sections] == ["new"], "the stale structure must not survive"
        assert "| 1 | 2 |" in stored["content"].splitlines()

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_full_refresh_stores_the_document_the_agent_emitted(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
    ):
        """The generation path never reads markdown back.

        In document mode the agent states the document's structure, so the
        refresh stores that structure verbatim and renders the markdown from it.
        Splitting only happens when a run produced plain text instead.
        """
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            document_from_sections,
            render_document,
        )

        bank_id = f"test-doc-answer-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Old\n\nOriginal.\n",
            trigger={"mode": "full"},
            request_context=request_context,
        )

        emitted = document_from_sections(
            {
                "sections": [
                    {
                        "heading": "Operations",
                        "level": 2,
                        "blocks": ["| Op | Budget |\n| --- | --- |\n| retain | 12ms |", "- top\n  - nested"],
                    }
                ]
            }
        )
        # ``text`` is deliberately something else: if the refresh were still
        # deriving the document from markdown, it would store this instead.
        patch_reflect(memory, text="IGNORED MARKDOWN", document=emitted)

        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored is not None
        assert StructuredDocument.model_validate(stored["structured_content"]) == emitted
        assert stored["content"] == render_document(emitted)
        assert "IGNORED MARKDOWN" not in stored["content"]
        assert "| retain | 12ms |" in stored["content"].splitlines()

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_refresh_asks_the_agent_for_a_document(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
    ):
        """The refresh must request document mode; markdown mode would reintroduce the parse."""
        bank_id = f"test-doc-flag-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Old\n\nOriginal.\n",
            trigger={"mode": "full"},
            request_context=request_context,
        )
        calls = patch_reflect(memory, text="# Team\n\nSomething.\n")

        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        assert calls[0]["answer_as_document"] is True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_plain_text_answer_still_produces_a_document(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
    ):
        """A run that yields no document (provider dropped the tool call, iteration
        limit) still gets a structure — split from its text, losslessly."""
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            render_document,
        )

        bank_id = f"test-doc-fallback-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Old\n\nOriginal.\n",
            trigger={"mode": "full"},
            request_context=request_context,
        )
        patch_reflect(memory, text="## Ops\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n", document=None)

        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
        stored = await memory.get_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert stored is not None
        doc = StructuredDocument.model_validate(stored["structured_content"])
        assert stored["content"] == render_document(doc)
        assert "| 1 | 2 |" in stored["content"].splitlines()

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_stored_content_is_the_render_of_the_stored_structure(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """``content`` is a derived view of ``structured_content``, on both legs.

        Under the v1 schema the full leg stored the LLM candidate verbatim while
        deriving the structure from it with a lossy parser, so the two columns
        disagreed by construction and the next delta refresh silently published
        the degraded one (#3361). They must now agree after every write.
        """
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            render_document,
        )

        bank_id = f"test-delta-render-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nOriginal content.\n",
            trigger={"mode": "full"},
            request_context=request_context,
        )

        # A candidate whose markdown v1's parser could not model: a table with a
        # row missing its outer pipe, a nested list, and a hard line break.
        candidate = (
            "# Team\n\n| Name | Role |\n|---|---|\n| Alice | Lead\n\n- top\n  - nested\n\nline one  \nline two\n"
        )
        patch_reflect(
            memory,
            text=candidate,
            facts=[{"id": "obs-1", "text": "Alice leads the team", "type": "observation", "context": None}],
        )
        llm_calls = patch_llm_call(memory, returns=[{"op": "append_block", "section_id": "team", "text": "New note."}])

        async def _assert_invariant(where: str) -> dict:
            stored = await memory.get_mental_model(
                bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
            )
            assert stored is not None
            doc = StructuredDocument.model_validate(stored["structured_content"])
            assert stored["content"] == render_document(doc), f"{where}: content is not the render"
            # ...and the constructs v1 flattened are still on their own lines.
            lines = stored["content"].splitlines()
            assert "| Alice | Lead" in lines, where
            assert "  - nested" in lines, where
            assert "line one  " in lines, where
            return stored

        # Full leg: the candidate is what gets written.
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
        await _assert_invariant("full refresh")
        assert len(llm_calls) == 0, "full mode must not call the delta LLM"

        # Delta leg: the same invariant must hold after an operation is applied
        # on top of that structure, and the fragile blocks it never named must
        # come through untouched.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        patch_reflect(
            memory,
            text="# Team\n\nBob joined.\n",
            facts=[{"id": "obs-2", "text": "Bob joined the team", "type": "observation", "context": None}],
        )
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)
        stored = await _assert_invariant("delta refresh")
        assert len(llm_calls) == 1
        assert "New note." in stored["content"]

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

        assert refreshed["content"] == "# Team\n\nFull fresh synthesis.\n"
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

        assert refreshed["content"] == "# Backend\n\nFull fresh synthesis.\n"
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

        assert refreshed["content"] == "# Customers\n\nBrand new topic.\n"
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
            trigger={"mode": "delta", "refresh_cron": "* * * * *"},
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
            automatic: bool = False,
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
                "text": "- Bob — junior engineer",
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

    async def test_delta_call_sends_the_operation_schema(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """The delta call asks for structured output, like retain's extraction (#3901).

        It used to be a bare text call: the operation schema is a discriminated
        union, which no provider would accept as ``oneOf`` + ``discriminator``, so
        the prompt was the only thing describing the payload and a model that
        spelled a block differently cost a whole refresh. The union is now sent as
        ``anyOf``, so the schema travels with the request.

        ``skip_validation`` must stay on: the raw JSON goes to
        ``parse_delta_operation_list``, which drops a single malformed op instead
        of failing the batch the way ``model_validate`` would.
        """
        from hindsight_api.config import get_config
        from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

        bank_id = f"test-delta-schema-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nAlice is the lead.\n",
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # First refresh runs in full mode and seeds the tracking column; only the
        # second one takes the delta path whose call shape this test is about.
        patch_reflect(memory, text="ignored — full mode candidate")
        patch_llm_call(memory, returns=[])
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        patch_reflect(
            memory,
            text="# Team\n\nAlice is the lead. Bob joined.",
            facts=[{"id": "obs-bob", "text": "Bob joined the team", "type": "observation", "context": None}],
        )
        llm_calls = patch_llm_call(memory, returns=[])
        await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context)

        assert llm_calls, "the structured-delta call must fire"
        call = llm_calls[-1]
        assert call["response_format"] is DeltaOperationList
        assert call["skip_validation"] is True
        assert call["strict_schema"] == get_config().llm_strict_schema_reflect

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
            trigger={"mode": "delta"},
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
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        old_id = await _seed_fact_row(memory, bank_id, "Alice has been the team lead since 2019")
        new_id = await _seed_fact_row(memory, bank_id, "Bob joined the team as junior engineer")

        # First refresh seeds prior based_on with an OLD fact (zero ops applied).
        patch_reflect(
            memory,
            text="ignored — delta keeps existing",
            facts=[
                {
                    "id": old_id,
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
        assert old_id in {f.get("id") for f in first_based_on.get("observation", [])}

        # Second refresh brings only a NEW fact.
        patch_reflect(
            memory,
            text="# Team\n\nAlice is the lead. Bob joined.",
            facts=[
                {
                    "id": new_id,
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
                "text": "- Bob — junior engineer",
            }
        ]
        llm_calls = patch_llm_call(memory, returns=ops)

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(llm_calls) == 1
        user_msg = llm_calls[0]["messages"][1]["content"]
        # The NEW fact is sent to the delta call...
        assert new_id in user_msg
        assert "Bob joined the team" in user_msg
        # ...but the accumulated OLD fact must NOT be re-sent (the regression).
        assert old_id not in user_msg
        assert "Alice has been the team lead since 2019" not in user_msg

        # based_on still ACCUMULATES both facts for grounding/audit.
        based_on = (refreshed.get("reflect_response") or {}).get("based_on") or {}
        obs_ids = {f.get("id") for f in based_on.get("observation", [])}
        assert obs_ids == {new_id, old_id}

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
            trigger={"mode": "delta"},
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
            trigger={"mode": "delta"},
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
                    "text": "Bob joined the team.",
                },
                {
                    "op": "append_block",
                    "section_id": "also-missing",
                    "text": "Bob sits with Alice.",
                },
            ],
        )

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
            trigger={"mode": "delta"},
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
                    "text": "Bob joined the team.",
                },
                {
                    "op": "append_block",
                    "section_id": "does-not-exist",
                    "text": "Dropped on the floor.",
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
            trigger={"mode": "delta"},
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
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        from hindsight_api.engine.reflect import structured_doc

        def unreadable(_stored, _markdown: str):
            raise ValueError("simulated unreadable structured document")

        monkeypatch.setattr(structured_doc, "structured_document_from_stored", unreadable)

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
        # gemini-2.0-flash was retired by the provider (404 NOT_FOUND); this is the
        # model the repo's own .env and the other Gemini tests use.
        model = os.getenv("HINDSIGHT_GEMINI_EVAL_MODEL", "gemini-3.1-flash-lite")
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
        # Three outcomes all satisfy "nothing drifted": the delta ran and emitted
        # zero ops, it emitted non-effective ops, or the window held no new facts
        # at all and the content was preserved without an LLM call. Which one you
        # get depends on whether consolidation had produced a new observation by
        # the time the second refresh ran, so pinning one of them makes the test
        # flake on timing rather than on behaviour.
        assert rr.get("delta_applied") is True or rr.get("delta_skipped_reason") == "no_new_facts", (
            f"expected a clean no-change refresh, got {rr.get('delta_applied')=} "
            f"{rr.get('delta_skipped_reason')=} {rr.get('refresh_skipped')=}"
        )

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
            render_section,
            split_markdown,
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
        # The stored structure is what the second refresh operates on, and the
        # stored markdown is its render — so splitting the markdown back gives
        # the same sections, which is what the preservation check compares.
        before = split_markdown(first_content)

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
        after = split_markdown(content)
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

    async def test_document_survives_many_delta_rounds_intact(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """Repeated real-LLM delta refreshes must not erode the document.

        This is the failure mode #3361 reported: no single refresh looks wrong,
        but the markdown degrades a little on each one until a table is one line
        and the damage is a fixed point. A single-round test cannot see it, so
        this one runs several rounds against a real model, feeding a genuinely
        new fact each time, and checks the invariants after every round:

        1. ``content`` is exactly the render of the stored structure.
        2. Fragile constructs the model never named — a table, a nested list, a
           fenced code block, a hard line break — survive byte-for-byte.
        3. Sections no applied operation named are byte-identical to the round
           before.
        4. No line ever welds a table separator to other cells (the detector
           the issue used against production pages).
        5. The document keeps growing knowledge rather than collapsing.
        """
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            render_document,
            render_section,
            split_markdown,
        )

        # A separator cell that is not the whole line == a table welded onto one
        # physical line. Verbatim from the #3361 report.
        collapsed_table_rx = re.compile(r"\|\s*:?-{2,}:?\s*\|")

        canaries = {
            "table row": "| `retain` | Store a memory | 12ms |",
            "table separator": "| --- | --- | --- |",
            "nested bullet": "  - Nested under retries",
            "deep bullet": "    - Deeper still, three levels",
            "code fence line": '    return {"ok": True}',
            "hard line break": "Latency budget is 200ms  ",
            "blockquote": "> Never block the request path on consolidation.",
            "ordered from five": "5. Fifth step, numbering starts at five on purpose",
        }

        existing_markdown = (
            "## Purpose\n\n"
            "Document the API surface, its performance budget, and the delivery rules.\n\n"
            "## Operations\n\n"
            "| Operation | Description | Budget |\n"
            "| --- | --- | --- |\n"
            "| `retain` | Store a memory | 12ms |\n"
            "| `recall` | Retrieve memories | 40ms |\n\n"
            "## Failure Handling\n\n"
            "- Retry transient errors\n"
            "  - Nested under retries\n"
            "    - Deeper still, three levels\n"
            "- Fail loudly on schema errors\n\n"
            "## Example\n\n"
            "```python\n"
            "def handler(request):\n"
            "\n"
            '    return {"ok": True}\n'
            "```\n\n"
            "## Constraints\n\n"
            "Latency budget is 200ms  \n"
            "measured at the p95.\n\n"
            "> Never block the request path on consolidation.\n\n"
            "## Procedure\n\n"
            "5. Fifth step, numbering starts at five on purpose\n"
            "6. Sixth step\n"
        )

        rounds = [
            "The recall endpoint gained a rerank stage that adds about 15ms.",
            "Schema errors are now reported with the offending field name.",
            "A new operation, reflect, answers questions over stored memories.",
            "The p95 latency budget was raised from 200ms to 250ms.",
            "Consolidation runs on a background worker every five minutes.",
        ]

        bank_id = f"eval-delta-stability-{uuid.uuid4().hex[:8]}"
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=existing_markdown,
            memories=["The API exposes retain and recall operations."],
        )
        mental_model_id = seeded["mm"]["id"]
        previous_content = seeded["first"]["content"]

        def touched_sections(refresh: dict[str, Any]) -> set[str]:
            applied = (refresh.get("reflect_response") or {}).get("delta_operations_applied") or []
            ids = {op.get("section_id") for op in applied}
            ids |= {op.get("assigned_id") for op in applied}
            return {i for i in ids if i}

        def owning_section(markdown: str, line: str) -> str | None:
            return next((s.id for s in split_markdown(markdown).sections if line in render_section(s)), None)

        # The seeding refresh is a real delta refresh: the model may deliberately
        # rewrite sections there too, so the contract is the same as every later
        # round — a construct in a section no operation named must survive.
        first_touched = touched_sections(seeded["first"])
        for name, canary in canaries.items():
            if owning_section(existing_markdown, canary) in first_touched:
                continue
            assert canary in previous_content.splitlines(), (
                f"{name} was lost by the first refresh, which never named its section "
                f"(touched: {sorted(first_touched)}):\n{previous_content}"
            )
        # Constructs the seed refresh edited away are no longer part of the contract.
        canaries = {n: c for n, c in canaries.items() if c in previous_content.splitlines()}

        previous_sections = {s.id: render_section(s) for s in split_markdown(previous_content).sections}
        # Sections no operation has *ever* named must still be byte-identical at
        # the end of the run, not just between consecutive rounds.
        never_touched = dict(previous_sections)
        for section_id in first_touched:
            never_touched.pop(section_id, None)

        for round_index, new_fact in enumerate(rounds, start=1):
            await gemini_memory.retain_batch_async(
                bank_id=bank_id,
                contents=[{"content": new_fact}],
                request_context=request_context,
            )
            await gemini_memory.wait_for_background_tasks()

            refreshed = await gemini_memory.refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            content = refreshed["content"]
            where = f"round {round_index} (fact: {new_fact!r})"

            stored = await gemini_memory.get_mental_model(
                bank_id=bank_id, mental_model_id=mental_model_id, request_context=request_context
            )
            assert stored is not None
            structured = stored.get("structured_content")
            assert structured is not None, f"{where}: no structure was persisted"
            doc = StructuredDocument.model_validate(structured)
            assert content == render_document(doc), (
                f"{where}: stored markdown is not the render of the stored structure"
            )

            for line in content.splitlines():
                if collapsed_table_rx.search(line):
                    assert line.strip().startswith("|") and set(line.replace("|", "").strip()) <= set("-: "), (
                        f"{where}: a table was welded onto one line (#3361):\n{line}"
                    )

            rr = refreshed.get("reflect_response") or {}
            applied = rr.get("delta_operations_applied") or []
            touched = touched_sections(refreshed)
            for section_id in touched:
                never_touched.pop(section_id, None)
            print(
                f"[delta-stability] {where}: applied={len(applied)} "
                f"skipped={len(rr.get('delta_operations_skipped') or [])} "
                f"touched={sorted(i for i in touched if i)} bytes={len(content)}"
            )

            current = split_markdown(content)
            for section in current.sections:
                if section.id in touched:
                    continue
                before = previous_sections.get(section.id)
                if before is None:
                    continue  # a section added this round has no prior form
                assert render_section(section) == before, (
                    f"{where}: untouched section {section.id!r} drifted.\n"
                    f"BEFORE:\n{before}\n\nAFTER:\n{render_section(section)}"
                )

            previous_lines = previous_content.splitlines()
            for name, canary in canaries.items():
                if canary not in previous_lines:
                    continue  # an earlier round deliberately edited it away
                if owning_section(previous_content, canary) in touched:
                    continue  # the model deliberately edited that section
                assert canary in content.splitlines(), (
                    f"{where}: {name} disappeared from a section the model never touched.\n{content}"
                )

            previous_content = content
            previous_sections = {s.id: render_section(s) for s in current.sections}

        print(f"[delta-stability] final document after {len(rounds)} rounds:\n{previous_content}")

        # The end-to-end contract: a section no operation ever named survives all
        # five rounds byte-for-byte. Consecutive-round checks alone would miss a
        # slow erosion that moves a section a little at a time.
        assert never_touched, (
            "Every section was edited at least once, so this run proves nothing about "
            "preservation — the fixture's facts have drifted too close to the seed document."
        )
        final_by_id = {s.id: render_section(s) for s in split_markdown(previous_content).sections}
        for section_id, original in never_touched.items():
            assert final_by_id.get(section_id) == original, (
                f"Section {section_id!r} eroded across {len(rounds)} rounds without any "
                f"operation ever naming it.\nBEFORE:\n{original}\n\nAFTER:\n{final_by_id.get(section_id)}"
            )

        await gemini_memory.delete_bank(bank_id, request_context=request_context)

    async def test_an_over_budget_document_reclaims_space(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """A real model, told the document is over budget, makes room.

        The mechanical half (the budget reaches the prompt, the outcome is
        recorded) is covered without an LLM. What cannot be mocked is whether a
        model asked to reclaim space actually removes superseded content instead
        of appending anyway — which is the whole reason the budget is stated
        rather than enforced by truncation.
        """
        from hindsight_api.engine.reflect.tokenization import count_prompt_tokens

        bank_id = f"eval-budget-{uuid.uuid4().hex[:8]}"
        # A document padded with obsolete history: there is plenty here that a
        # model can drop without losing anything the topic needs.
        stale_sections = "\n\n".join(
            f"## Archived note {i}\n\nDuring the {2019 + i} season the team used the legacy pipeline, "
            f"which was retired years ago and no longer affects how anything works today."
            for i in range(12)
        )
        # Two things that must survive, because "get under budget" must not become
        # "delete the document": the current cadence, and a checklist that is
        # nowhere near stale. Space has to come from the archive.
        current = (
            "## Current process\n\nReleases are cut on Tuesdays.\n\n"
            "## Checklist\n\n- Migrations applied\n- Smoke tests green\n- On-call engineer signed off\n"
        )
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=f"{current}\n{stale_sections}\n",
            memories=["Releases are cut on Tuesdays and go to staging before production."],
        )
        mental_model_id = seeded["mm"]["id"]
        budget = 200
        await gemini_memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model_id,
            max_tokens=budget,
            request_context=request_context,
        )
        before = count_prompt_tokens(seeded["first"]["content"])
        assert before > budget, f"the fixture must start over budget, got {before} <= {budget}"

        await gemini_memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Releases moved from Tuesdays to Wednesdays."}],
            request_context=request_context,
        )
        await gemini_memory.wait_for_background_tasks()

        refreshed = await gemini_memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mental_model_id, request_context=request_context
        )
        after = count_prompt_tokens(refreshed["content"])
        rr = refreshed.get("reflect_response") or {}
        print(f"[budget] {before} -> {after} tokens against a {budget}-token budget")

        assert rr.get("document_budget") == budget
        assert rr.get("document_tokens") == after
        # The new fact still lands — reclaiming space must never cost the update.
        assert "wednesday" in refreshed["content"].lower()
        # And the document moved toward its budget rather than growing further.
        assert after < before, (
            f"an over-budget document grew from {before} to {after} tokens instead of "
            f"reclaiming space:\n{refreshed['content']}"
        )
        # Space came from the archive, not from the document. Getting under budget
        # by deleting current content would satisfy every check above and be worse
        # than going over it.
        content = refreshed["content"].lower()
        assert "checklist" in content, f"the current checklist was deleted to fit the budget:\n{refreshed['content']}"
        assert "smoke tests" in content, f"current content was dropped to fit the budget:\n{refreshed['content']}"
        archived_left = content.count("archived note")
        assert archived_left < 12, "nothing was reclaimed from the archived sections"

        await gemini_memory.delete_bank(bank_id, request_context=request_context)


class TestDocumentBudget:
    """``max_tokens`` on the delta leg.

    It was enforced in exactly one place — a rewrite of the *synthesis* answer —
    and in delta mode that answer is only context for the operations call. The
    document that actually gets stored was never measured against it, and a delta
    refresh only adds, so a long-lived page drifts past its configured size with
    nothing noticing. See ``test_mental_model_document_budget.py`` for the prompt
    that tells the model where it stands.
    """

    async def test_refresh_records_size_against_budget(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """Every delta refresh reports where the document stands."""
        bank_id = f"test-mm-budget-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content="## Ops\n\nOriginal.\n",
            max_tokens=256,
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        patch_reflect(
            memory,
            text="## Ops\n\nCandidate.\n",
            facts=[{"id": "o1", "text": "a new fact", "type": "observation", "context": None}],
        )
        patch_llm_call(memory, returns=[{"op": "append_block", "section_id": "ops", "text": "New note."}])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        rr = refreshed.get("reflect_response") or {}
        assert rr["document_budget"] == 256
        assert rr["document_tokens"] > 0

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_over_budget_document_is_reported_not_truncated(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """Going over is a warning, never a silent deletion of content."""
        bank_id = f"test-mm-over-budget-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        long_body = " ".join(f"sentence number {i} about the API." for i in range(200))
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="API Reference",
            source_query="Document the API",
            content=f"## Ops\n\n{long_body}\n",
            max_tokens=256,
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        patch_reflect(
            memory,
            text="## Ops\n\nCandidate.\n",
            facts=[{"id": "o1", "text": "a new fact", "type": "observation", "context": None}],
        )
        patch_llm_call(memory, returns=[{"op": "append_block", "section_id": "ops", "text": "New note."}])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert "sentence number 199" in refreshed["content"], "content must not be truncated"
        assert "New note." in refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        assert rr["document_tokens"] > rr["document_budget"]

        await memory.delete_bank(bank_id, request_context=request_context)
