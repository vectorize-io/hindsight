"""HTTP + engine integration tests for the knowledge base (folders + pages).

Pages are seeded directly via the engine (deterministic content, no LLM) so the
tree, markdown rendering, move/rename, and cascade-delete behaviour can be asserted
without consolidation.
"""

import asyncio
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

import asyncpg
import pytest
import pytest_asyncio

import hindsight_api.engine.memory_engine as memory_engine_module
from hindsight_api.engine.db import DatabaseConnection
from hindsight_api.engine.memory_engine import (
    MENTAL_MODEL_PENDING_CONTENT,
    MemoryEngine,
    _may_need_refresh,
    fq_table,
)
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.extensions import (
    BankReadContext,
    BankReadOperation,
    BankWriteContext,
    BankWriteOperation,
    OperationValidatorExtension,
    ValidationResult,
)


def _enc(bank_id: str) -> str:
    return urllib.parse.quote(bank_id, safe="")


# The statement structural knowledge-tree writers use to serialize on the bank row.
_BANK_LOCK_SQL = "FOR NO KEY UPDATE"


class _BankLockPauser:
    """Freeze the first structural tree writer for one bank while it holds the lock.

    Wraps the connection the engine acquires and pauses inside the *real* bank-lock
    query, after it has taken the row lock. A second writer then reaches the same
    query and cannot get past it until the first commits, so the interleaving under
    test is driven by events rather than by sleeps and is repeatable.

    Matching is scoped to this bank's id, not just the SQL text: other callers take
    the same lock on their own bank rows (async-operation submit dedupe, for one),
    and mistaking one of those for the writer under test would pause an unrelated
    transaction and desynchronize the test.
    """

    def __init__(self, bank_id: str) -> None:
        self.bank_id = bank_id
        self.first_holds_lock = asyncio.Event()
        self.second_reached_lock = asyncio.Event()
        self.release_first = asyncio.Event()
        self._locks_seen = 0

    def install(self, monkeypatch) -> None:
        original_acquire = memory_engine_module.acquire_with_retry
        pauser = self

        class _PausingConnection:
            def __init__(self, conn):
                self._conn = conn

            async def fetchrow(self, query, *args, **kwargs):
                if _BANK_LOCK_SQL in query and args and args[0] == pauser.bank_id:
                    pauser._locks_seen += 1
                    if pauser._locks_seen == 1:
                        row = await self._conn.fetchrow(query, *args, **kwargs)
                        pauser.first_holds_lock.set()
                        await pauser.release_first.wait()
                        return row
                    pauser.second_reached_lock.set()
                return await self._conn.fetchrow(query, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        @asynccontextmanager
        async def pausing_acquire(backend, *args, **kwargs):
            async with original_acquire(backend, *args, **kwargs) as conn:
                yield _PausingConnection(conn)

        monkeypatch.setattr(memory_engine_module, "acquire_with_retry", pausing_acquire)


async def _assert_blocked(task: asyncio.Task, label: str) -> None:
    """Assert the task is still waiting on the bank lock (it can only be released by a commit)."""
    done, _pending = await asyncio.wait({task}, timeout=0.5)
    assert not done, f"{label} must block on the bank lock until the first writer commits"


class _RecordingValidator(OperationValidatorExtension):
    """A validator that records every bank read/write and rejects one operation.

    A concrete subclass (not a MagicMock) so every inherited hook — including the
    async post-hooks the background mental-model refresh worker fires after a page
    create — is a real coroutine. Only the named operation is rejected; every other
    hook (including DELETE_BANK, used by the ``kb_bank`` fixture teardown) accepts.
    """

    def __init__(
        self,
        *,
        reject_read: BankReadOperation | None = None,
        reject_write: BankWriteOperation | None = None,
        reason: str = "operation is forbidden",
    ) -> None:
        super().__init__({})
        self._reject_read = reject_read
        self._reject_write = reject_write
        self._reason = reason
        self.read_ops: list[BankReadOperation] = []
        self.write_ops: list[BankWriteOperation] = []
        # A page read is a mental model read, so it runs the metering pair
        # too. Recorded here so tests can assert the meter fires rather than
        # only that access was checked.
        self.model_gets: list[str] = []
        self.model_reads: list[str] = []
        self.model_get_tokens: list[int] = []
        self.reject_model_get = False

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_mental_model_get(self, ctx) -> ValidationResult:
        self.model_gets.append(ctx.mental_model_id)
        if self.reject_model_get:
            return ValidationResult.reject("insufficient credits")
        return ValidationResult.accept()

    async def on_mental_model_get_complete(self, result) -> None:
        # Recorded separately from `model_gets`: the gate and the completion
        # hook do not always both fire. A single read runs both; a list reports
        # completion for each model it delivered without gating each one.
        self.model_reads.append(result.mental_model_id)
        self.model_get_tokens.append(result.output_tokens)

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        self.read_ops.append(ctx.operation)
        if ctx.operation is self._reject_read:
            return ValidationResult.reject(self._reason)
        return ValidationResult.accept()

    async def validate_bank_write(self, ctx: BankWriteContext) -> ValidationResult:
        self.write_ops.append(ctx.operation)
        if ctx.operation is self._reject_write:
            return ValidationResult.reject(self._reason)
        return ValidationResult.accept()


def _kb_validator(
    *,
    reject_read: BankReadOperation | None = None,
    reject_write: BankWriteOperation | None = None,
    reason: str = "operation is forbidden",
) -> _RecordingValidator:
    return _RecordingValidator(reject_read=reject_read, reject_write=reject_write, reason=reason)


def _read_ops(validator: _RecordingValidator) -> list[BankReadOperation]:
    return list(validator.read_ops)


def _write_ops(validator: _RecordingValidator) -> list[BankWriteOperation]:
    return list(validator.write_ops)


class _Seed:
    """Holds the ids created by the seed fixture for assertions."""

    def __init__(self, **ids):
        self.__dict__.update(ids)


@pytest_asyncio.fixture
async def kb_bank(memory: MemoryEngine, request_context):
    """A bank with folders, nested folders, and pages."""
    bank_id = f"test-kb-{uuid.uuid4().hex[:8]}"

    runbooks = await memory.create_knowledge_folder(bank_id, "Runbooks", request_context=request_context)
    policies = await memory.create_knowledge_folder(bank_id, "Policies", request_context=request_context)
    sub = await memory.create_knowledge_folder(
        bank_id, "Sub", parent_id=runbooks["id"], request_context=request_context
    )
    orders = await memory.create_knowledge_page(
        bank_id,
        "Orders",
        "What are the order facts?",
        "# Orders\n\nOne row per order.",
        parent_id=runbooks["id"],
        tags=["type:runbook", "sales", "revenue"],
        request_context=request_context,
    )
    billing = await memory.create_knowledge_page(
        bank_id,
        "Billing",
        "What is the billing policy?",
        "# Billing\n\nNet-30.",
        parent_id=policies["id"],
        tags=["type:policy", "revenue"],
        request_context=request_context,
    )
    loose = await memory.create_knowledge_page(
        bank_id,
        "Loose",
        "A root page.",
        "# Loose\n\nNo folder, no tags.",
        tags=[],
        request_context=request_context,
    )

    yield (
        bank_id,
        _Seed(
            runbooks=runbooks["id"],
            policies=policies["id"],
            sub=sub["id"],
            orders=orders["id"],
            billing=billing["id"],
            loose=loose["id"],
            orders_mm=orders["mental_model_id"],
        ),
    )

    await memory.delete_bank(bank_id, request_context=request_context)


class TestTree:
    async def test_nested_tree(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        assert resp.status_code == 200, resp.text
        roots = {r["name"]: r for r in resp.json()["roots"]}
        assert set(roots) == {"Runbooks", "Policies", "Loose"}

        runbooks = roots["Runbooks"]
        assert runbooks["kind"] == "folder"
        child_names = {c["name"] for c in runbooks["children"]}
        assert child_names == {"Sub", "Orders"}

        orders = next(c for c in runbooks["children"] if c["name"] == "Orders")
        assert orders["kind"] == "page"
        # Human-created pages are pinned (not curator-managed).
        assert orders["managed"] is False
        assert "sales" in orders["tags"]
        # The tree computes per-page sync status. These seeds are created with
        # content (refreshed at creation) and the bank has no memories, so nothing
        # is newer than the refresh → in sync.
        assert orders["is_stale"] is False
        assert roots["Loose"]["kind"] == "page"

    async def test_tree_stale_flag_is_page_only(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        roots = {r["name"]: r for r in resp.json()["roots"]}
        # Folders never carry a sync status (None → omitted from the response).
        assert roots["Runbooks"].get("is_stale") is None
        # Pages always do.
        assert isinstance(roots["Loose"]["is_stale"], bool)

    async def test_tree_exposes_a_page_refresh_policy(self, api_client, kb_bank):
        """A page's trigger is readable where the page is.

        It decides when a page rebuilds itself and what that costs, so a client that only speaks
        the knowledge base — the control plane's tree, the coding-agents plugin — could neither
        show it nor tell whether its own settings still applied. The alternative was walking to
        the mental-models API once per page.
        """
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        roots = {r["name"]: r for r in resp.json()["roots"]}
        # The EFFECTIVE policy, not the stored keys: it serializes as MentalModelTrigger, so a
        # field nobody set comes back at that model's default (keep_trace=False here). Asserting
        # the whole dict would pin every future field of that model into this test.
        trigger = roots["Loose"]["trigger"]
        assert trigger["mode"] == "delta"
        assert trigger["fact_types"] == ["observation"]
        assert trigger["exclude_mental_models"] is True
        assert trigger["refresh_after_consolidation"] is True
        # A folder has no backing mental model, so it has no refresh policy either —
        # and a null is dropped from the response entirely (ExcludeNoneRoute), the same
        # way is_stale is absent on folders rather than null.
        assert roots["Runbooks"].get("trigger") is None

    async def test_tree_reflects_a_changed_refresh_policy(self, api_client, memory, kb_bank, request_context):
        """Read-back closes the loop: a client can compare and skip a no-op write.

        The page was created auto-refreshing; after moving it onto a schedule the tree shows the
        schedule and auto-refresh off. (That the engine *stores* no ``refresh_after_consolidation``
        key at all is asserted in TestPageDefaults — through this model it serializes as False,
        which is the same policy stated a different way.)
        """
        bank_id, ids = kb_bank
        await memory.update_knowledge_node(
            bank_id, ids.loose, trigger={"refresh_cron": "0 3 * * *"}, request_context=request_context
        )
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        loose = next(r for r in resp.json()["roots"] if r["name"] == "Loose")
        assert loose["trigger"]["refresh_cron"] == "0 3 * * *"
        assert loose["trigger"]["refresh_after_consolidation"] is False
        assert loose["trigger"]["mode"] == "delta"  # untouched by the patch

    @staticmethod
    async def _insert_memory(memory: MemoryEngine, bank_id: str, tags: list[str]) -> None:
        """One memory straight into the table — no LLM, no consolidation.

        An observation, because that is what knowledge pages are built from:
        ``KNOWLEDGE_PAGE_DEFAULT_TRIGGER`` scopes them to ``fact_types:
        ["observation"]``, so a raw experience is out of scope for every page
        however its tags line up.
        """
        async with memory._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at) "
                "VALUES (gen_random_uuid(), $1, $2, 'observation', $3::varchar[], now())",
                bank_id,
                "An observation written after the pages were built.",
                tags,
            )

    @pytest.mark.memory_backend_incompatible
    async def test_tree_staleness_ignores_writes_outside_a_page_scope(self, api_client, memory, kb_bank):
        """A write nowhere near a page's tags does not flag that page.

        The tree used to answer from one bank-wide watermark, so *any* write
        flagged *every* page that had not read it — and since only an in-scope
        write can move a page's own watermark, a page whose scope stayed quiet
        stayed flagged forever while the refresh gate correctly refused to
        refresh it (#3291). Each page is now asked about its own scope.
        """
        bank_id, ids = kb_bank
        # Untagged: in scope for the untagged page (which defaults to tags_match
        # "any" and so matches everything), out of scope for every tagged page
        # (which default to all_strict).
        await self._insert_memory(memory, bank_id, [])

        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        assert resp.status_code == 200, resp.text
        roots = {r["name"]: r for r in resp.json()["roots"]}
        orders = next(c for c in roots["Runbooks"]["children"] if c["name"] == "Orders")
        billing = next(c for c in roots["Policies"]["children"] if c["name"] == "Billing")

        assert roots["Loose"]["is_stale"] is True, "an untagged page sees every memory in the bank"
        assert orders["is_stale"] is False, "the write carried none of Orders' tags"
        assert billing["is_stale"] is False, "the write carried none of Billing's tags"

    @pytest.mark.memory_backend_incompatible
    async def test_tree_flags_the_page_whose_own_scope_changed(self, api_client, memory, kb_bank):
        """Only the page the write actually belongs to is flagged."""
        bank_id, ids = kb_bank
        # Carries all of Orders' tags (all_strict is a superset test) and only one
        # of Billing's, so it is in scope for Orders and not for Billing.
        await self._insert_memory(memory, bank_id, ["type:runbook", "sales", "revenue"])

        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        roots = {r["name"]: r for r in resp.json()["roots"]}
        orders = next(c for c in roots["Runbooks"]["children"] if c["name"] == "Orders")
        billing = next(c for c in roots["Policies"]["children"] if c["name"] == "Billing")

        assert orders["is_stale"] is True
        assert billing["is_stale"] is False, "Billing's type:policy tag is not on the write"
        assert roots["Loose"]["is_stale"] is True

    @pytest.mark.memory_backend_incompatible
    async def test_tree_agrees_with_the_exact_per_model_check(self, api_client, memory, kb_bank, request_context):
        """The tree and the single-model read answer the same question.

        This is the property #3291 lost: the tree reported almost everything as
        needing a refresh while the per-model check reported nothing did.
        """
        bank_id, ids = kb_bank
        await self._insert_memory(memory, bank_id, ["type:runbook", "sales", "revenue"])

        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")

        def _pages(nodes):
            for node in nodes:
                if node["kind"] == "page":
                    yield node
                yield from _pages(node.get("children") or [])

        pages = list(_pages(resp.json()["roots"]))
        assert pages, "fixture seeds pages"
        for page in pages:
            model = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
            assert page["is_stale"] == model["is_stale"], page["name"]

    @pytest.mark.memory_backend_incompatible
    async def test_tree_asks_once_for_the_whole_tree(self, api_client, memory, kb_bank):
        """Per-page answers, but not a query per page — the tree view polls."""
        bank_id, ids = kb_bank
        await self._insert_memory(memory, bank_id, [])

        from hindsight_api.engine.memories import get_memories

        store = get_memories()
        single_calls = 0
        batch_calls = 0
        original_single = store.any_memory_updated_since
        original_batch = store.any_memory_updated_since_batch

        async def counting_single(*args, **kwargs):
            nonlocal single_calls
            single_calls += 1
            return await original_single(*args, **kwargs)

        async def counting_batch(*args, **kwargs):
            nonlocal batch_calls
            batch_calls += 1
            return await original_batch(*args, **kwargs)

        store.any_memory_updated_since = counting_single
        store.any_memory_updated_since_batch = counting_batch
        try:
            resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
            assert resp.status_code == 200, resp.text
        finally:
            store.any_memory_updated_since = original_single
            store.any_memory_updated_since_batch = original_batch

        assert batch_calls == 1, "one batched question for every page in the tree"
        assert single_calls == 0, "no per-page fallback for plain flat-tag scopes"


class TestWatermarkRule:
    """The pure rule behind every "may need refresh" badge."""

    def test_never_refreshed_always_needs_one(self):
        assert _may_need_refresh(None, datetime.now(timezone.utc)) is True
        assert _may_need_refresh(None, None) is True

    def test_empty_bank_is_never_stale(self):
        assert _may_need_refresh(datetime.now(timezone.utc), None) is False

    def test_refresh_at_or_after_the_watermark_is_current(self):
        refreshed = datetime.now(timezone.utc)
        assert _may_need_refresh(refreshed, refreshed) is False
        assert _may_need_refresh(refreshed, refreshed - timedelta(seconds=1)) is False

    def test_a_write_after_the_refresh_may_need_one(self):
        refreshed = datetime.now(timezone.utc)
        assert _may_need_refresh(refreshed, refreshed + timedelta(microseconds=1)) is True


class TestSearch:
    """Doc-level hybrid search (BM25 + vector, RRF-fused). The BM25 arm runs on a
    generated tsvector over page name + content, so ranking is deterministic even
    though the seeds carry embeddings too."""

    async def test_ranks_relevant_page_first(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # "Billing" name + "Net-30" body → the BM25 arm lifts Billing to the top
        # of the fusion even though every page shares vocabulary.
        resp = await api_client.get(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/search",
            params={"q": "billing net-30", "limit": 5},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        names = [r["name"] for r in body["results"]]
        assert names, "expected at least one hit"
        assert names[0] == "Billing"
        assert body["total"] == len(body["results"])
        # Scores are strictly descending.
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)
        top = body["results"][0]
        assert top["id"] == ids.billing
        assert top["mental_model_id"]
        assert "Net-30" in top["snippet"] or "Billing" in top["snippet"]

    async def test_excludes_folders_and_respects_limit(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/search",
            params={"q": "orders billing loose net-30", "limit": 10},
        )
        assert resp.status_code == 200, resp.text
        result_ids = {r["id"] for r in resp.json()["results"]}
        assert not (result_ids & {ids.runbooks, ids.policies, ids.sub}), "folders must never appear"
        assert ids.billing in result_ids

        capped = await api_client.get(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/search",
            params={"q": "order", "limit": 1},
        )
        assert len(capped.json()["results"]) <= 1

    @pytest.mark.memory_backend_incompatible
    async def test_natural_language_question_still_matches_bm25(
        self, memory: MemoryEngine, kb_bank, request_context, monkeypatch
    ):
        """A multi-word question must not have to appear in a page word for word.

        The BM25 arm ORs its tokens, so "billing" alone carries the match; the
        conjunctive ``websearch_to_tsquery`` this replaced required *every* term and
        returned nothing for ordinary questions. The embedding is suppressed so the
        BM25 arm answers alone — otherwise the vector arm would hide the defect.
        """
        bank_id, ids = kb_bank

        async def no_embedding(embeddings, texts, *, input_type=None):
            return [None]

        monkeypatch.setattr(embedding_utils, "generate_embeddings_batch", no_embedding)

        results = await memory.search_knowledge_pages(
            bank_id,
            "what are the billing terms for late payments?",
            request_context=request_context,
        )

        assert results, "the BM25 arm must still generate candidates for a question"
        assert results[0]["id"] == ids.billing

    @pytest.mark.memory_backend_incompatible
    async def test_query_without_word_characters_returns_nothing(
        self, memory: MemoryEngine, kb_bank, request_context, monkeypatch
    ):
        """No tokens means no BM25 arm; with no embedding either there is nothing to
        rank on, and the empty token list must not reach ``to_tsquery``."""
        bank_id, _ = kb_bank

        async def no_embedding(embeddings, texts, *, input_type=None):
            return [None]

        monkeypatch.setattr(embedding_utils, "generate_embeddings_batch", no_embedding)

        assert await memory.search_knowledge_pages(bank_id, "???", request_context=request_context) == []

    async def test_query_is_required(self, api_client, kb_bank):
        bank_id, _ = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/search")
        assert resp.status_code == 422


class TestPageDefaults:
    """A knowledge page is a living document by default: observation-only, delta,
    auto-refreshing, with a larger token budget than a plain mental model."""

    async def test_default_trigger_and_max_tokens(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kb-def-{uuid.uuid4().hex[:8]}"
        page = await memory.create_knowledge_page(bank_id, "P", "What is P?", "seed", request_context=request_context)
        mm = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
        assert mm["trigger"] == {
            "mode": "delta",
            "fact_types": ["observation"],
            "exclude_mental_models": True,
            "refresh_after_consolidation": True,
        }
        assert mm["max_tokens"] == 4096
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_client_trigger_and_max_tokens_override_defaults(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kb-ovr-{uuid.uuid4().hex[:8]}"
        page = await memory.create_knowledge_page(
            bank_id,
            "P",
            "What is P?",
            "seed",
            trigger={"mode": "full", "refresh_after_consolidation": False},
            max_tokens=1024,
            request_context=request_context,
        )
        mm = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
        assert mm["trigger"]["mode"] == "full"
        assert mm["trigger"].get("refresh_after_consolidation") is False
        assert mm["max_tokens"] == 1024
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_partial_trigger_merges_over_the_page_defaults(self, memory: MemoryEngine, request_context):
        """Overriding one field must not silently give up the rest of the page contract.

        A supplied trigger used to REPLACE the defaults outright, so a client that only
        wanted different fact types also lost ``mode: "delta"`` and
        ``exclude_mental_models`` — its page rebuilt itself from scratch on every refresh
        and reflected over its sibling pages while doing it (#3506).
        """
        bank_id = f"test-kb-merge-{uuid.uuid4().hex[:8]}"
        page = await memory.create_knowledge_page(
            bank_id,
            "P",
            "What is P?",
            "seed",
            trigger={"fact_types": ["world", "experience", "observation"]},
            request_context=request_context,
        )
        mm = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
        assert mm["trigger"] == {
            "mode": "delta",
            "fact_types": ["world", "experience", "observation"],
            "exclude_mental_models": True,
            "refresh_after_consolidation": True,
        }
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_cron_trigger_drops_the_default_auto_refresh(self, memory: MemoryEngine, request_context):
        """The merge must not synthesize a pair no request could have expressed.

        ``MentalModelTrigger`` rejects a body carrying both refresh triggers, so inheriting
        the default's ``refresh_after_consolidation`` alongside a client's ``refresh_cron``
        would store a combination the API itself would have refused.
        """
        bank_id = f"test-kb-cron-{uuid.uuid4().hex[:8]}"
        page = await memory.create_knowledge_page(
            bank_id,
            "P",
            "What is P?",
            "seed",
            trigger={"refresh_cron": "0 3 * * *"},
            request_context=request_context,
        )
        mm = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
        assert mm["trigger"]["refresh_cron"] == "0 3 * * *"
        assert "refresh_after_consolidation" not in mm["trigger"]
        assert mm["trigger"]["mode"] == "delta"  # still a knowledge page
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_update_patches_the_trigger_instead_of_replacing_it(self, memory: MemoryEngine, request_context):
        """Changing when a page refreshes must not reset how it refreshes.

        ``update_mental_model`` overwrites the whole trigger column, so forwarding a
        partial one straight through would strip every field the client didn't mention
        — the create-path defect (#3506) one endpoint over.
        """
        bank_id = f"test-kb-upd-{uuid.uuid4().hex[:8]}"
        page = await memory.create_knowledge_page(bank_id, "P", "What is P?", "seed", request_context=request_context)
        await memory.update_knowledge_node(
            bank_id,
            page["id"],
            trigger={"refresh_cron": "0 3 * * *"},
            request_context=request_context,
        )
        mm = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
        assert mm["trigger"]["refresh_cron"] == "0 3 * * *"
        assert mm["trigger"]["mode"] == "delta"
        assert mm["trigger"]["fact_types"] == ["observation"]
        assert mm["trigger"]["exclude_mental_models"] is True
        # Moving onto a schedule clears the auto-refresh it was created with, in the
        # direction the create path never had to handle.
        assert "refresh_after_consolidation" not in mm["trigger"]

        # ...and back again: the stated auto-refresh clears the stored cron.
        await memory.update_knowledge_node(
            bank_id,
            page["id"],
            trigger={"refresh_after_consolidation": True},
            request_context=request_context,
        )
        mm = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
        assert mm["trigger"]["refresh_after_consolidation"] is True
        assert "refresh_cron" not in mm["trigger"]
        assert mm["trigger"]["mode"] == "delta"
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_update_without_a_trigger_leaves_it_alone(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kb-keep-{uuid.uuid4().hex[:8]}"
        page = await memory.create_knowledge_page(
            bank_id,
            "P",
            "What is P?",
            "seed",
            trigger={"refresh_cron": "0 3 * * *"},
            request_context=request_context,
        )
        await memory.update_knowledge_node(bank_id, page["id"], max_tokens=2048, request_context=request_context)
        mm = await memory.get_mental_model(bank_id, page["mental_model_id"], request_context=request_context)
        assert mm["max_tokens"] == 2048
        assert mm["trigger"]["refresh_cron"] == "0 3 * * *"
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_update_endpoint_accepts_and_forwards_a_partial_trigger(
        self, api_client, kb_bank, memory, monkeypatch
    ):
        """The PATCH body carries `trigger` at all, and only the fields the client set.

        Both halves are load-bearing: the field was missing from ``UpdateNodeRequest``
        entirely, so a page's refresh policy could not be changed through the
        knowledge-base API — and a full dump would carry this model's defaults into
        every update.
        """
        bank_id, ids = kb_bank
        captured: dict[str, Any] = {}

        async def fake_update(**kwargs):
            captured.update(kwargs)
            return {"id": ids.orders, "kind": "page", "name": "Orders", "mental_model_id": ids.orders_mm}

        monkeypatch.setattr(memory, "update_knowledge_node", fake_update)
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"trigger": {"refresh_cron": "0 4 * * *"}},
        )
        assert resp.status_code == 200, resp.text
        assert captured["trigger"] == {"refresh_cron": "0 4 * * *"}

    async def test_create_endpoint_forwards_only_the_fields_the_client_set(
        self, api_client, kb_bank, memory, monkeypatch
    ):
        """The merge is only meaningful if the HTTP layer stops filling in model defaults.

        ``model_dump()`` on the request model yields every field — mode="full",
        exclude_mental_models=False — which would override the page defaults on every
        create that carries a trigger at all.
        """
        bank_id, _ = kb_bank
        captured: dict[str, Any] = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return {"id": "kp-fake", "mental_model_id": "mm-fake"}

        async def fake_submit(**kwargs):
            return {"operation_id": "op-fake"}

        monkeypatch.setattr(memory, "create_knowledge_page", fake_create)
        monkeypatch.setattr(memory, "submit_async_refresh_mental_model", fake_submit)
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages",
            json={"name": "P", "source_query": "what is P?", "trigger": {"refresh_cron": "0 3 * * *"}},
        )
        assert resp.status_code == 201, resp.text
        assert captured["trigger"] == {"refresh_cron": "0 3 * * *"}


class TestGetPage:
    async def test_okf_document(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")
        assert resp.status_code == 200, resp.text
        page = resp.json()
        assert page["type"] == "runbook"
        assert page["body"].startswith("# Orders")
        assert page["markdown"].startswith("---\n")
        assert 'type: "runbook"' in page["markdown"]

    async def test_missing_page_404(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/nope")
        assert resp.status_code == 404


class TestCreate:
    async def test_create_folder(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders",
            json={"name": "Guides", "parent_id": None},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["kind"] == "folder"
        assert resp.json()["name"] == "Guides"

    async def test_create_folder_bad_parent(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # parent that is a page, not a folder → 400
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders",
            json={"name": "Nope", "parent_id": ids.orders},
        )
        assert resp.status_code == 400

    async def test_create_page_missing_parent_rolls_back_mental_model(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kb-create-{uuid.uuid4().hex[:8]}"
        await memory.create_knowledge_folder(bank_id, "Root", request_context=request_context)
        before = await memory.list_mental_models(bank_id, request_context=request_context)

        with pytest.raises(ValueError, match="not found"):
            await memory.create_knowledge_page(
                bank_id,
                "Orphan",
                "What is orphaned?",
                "seed",
                parent_id="missing-parent",
                request_context=request_context,
            )

        after = await memory.list_mental_models(bank_id, request_context=request_context)
        assert {mm["id"] for mm in after.items} == {mm["id"] for mm in before.items}
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_create_page_under_page_rolls_back_mental_model(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kb-create-{uuid.uuid4().hex[:8]}"
        parent = await memory.create_knowledge_page(
            bank_id, "Parent page", "What is the parent?", "seed", request_context=request_context
        )
        before = await memory.list_mental_models(bank_id, request_context=request_context)

        with pytest.raises(ValueError, match="is not a folder"):
            await memory.create_knowledge_page(
                bank_id,
                "Orphan",
                "What is orphaned?",
                "seed",
                parent_id=parent["id"],
                request_context=request_context,
            )

        after = await memory.list_mental_models(bank_id, request_context=request_context)
        assert {mm["id"] for mm in after.items} == {mm["id"] for mm in before.items}
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_duplicate_page_rolls_back_mental_model(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kb-create-{uuid.uuid4().hex[:8]}"
        parent = await memory.create_knowledge_folder(bank_id, "Root", request_context=request_context)
        await memory.create_knowledge_page(
            bank_id,
            "Existing",
            "What exists?",
            "seed",
            parent_id=parent["id"],
            request_context=request_context,
        )
        rolled_back_mm_id = f"mm-{uuid.uuid4().hex}"

        duplicate = await memory.create_knowledge_page(
            bank_id,
            "Existing",
            "What is duplicated?",
            "seed",
            parent_id=parent["id"],
            mental_model_id=rolled_back_mm_id,
            request_context=request_context,
        )

        assert duplicate is None
        assert await memory.get_mental_model(bank_id, rolled_back_mm_id, request_context=request_context) is None
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_duplicate_mental_model_id_is_not_reported_as_duplicate_page(
        self, memory: MemoryEngine, request_context
    ):
        bank_id = f"test-kb-create-{uuid.uuid4().hex[:8]}"
        existing = await memory.create_mental_model(
            bank_id, "Existing MM", "What exists?", "seed", request_context=request_context
        )

        with pytest.raises(asyncpg.UniqueViolationError):
            await memory.create_knowledge_page(
                bank_id,
                "New page",
                "What is new?",
                "seed",
                mental_model_id=existing["id"],
                request_context=request_context,
            )

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_non_unique_failure_after_mental_model_insert_rolls_back(
        self, memory: MemoryEngine, request_context, monkeypatch
    ):
        bank_id = f"test-kb-create-{uuid.uuid4().hex[:8]}"
        mental_model_id = f"mm-{uuid.uuid4().hex}"
        insert_mental_model = memory._insert_pinned_mental_model

        async def insert_then_fail(conn: DatabaseConnection, **kwargs: Any) -> NoReturn:
            await insert_mental_model(conn, **kwargs)
            raise RuntimeError("page write failed")

        monkeypatch.setattr(memory, "_insert_pinned_mental_model", insert_then_fail)

        with pytest.raises(RuntimeError, match="page write failed"):
            await memory.create_knowledge_page(
                bank_id,
                "Rolled back",
                "What is rolled back?",
                "seed",
                mental_model_id=mental_model_id,
                request_context=request_context,
            )

        assert await memory.get_mental_model(bank_id, mental_model_id, request_context=request_context) is None
        await memory.delete_bank(bank_id, request_context=request_context)


class TestExport:
    async def test_export_bundle_nested_index(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/export")
        assert resp.status_code == 200, resp.text
        files = {f["path"]: f["content"] for f in resp.json()["files"]}
        assert "index.md" in files
        assert f"{ids.orders}.md" in files
        # index reflects the folder hierarchy
        assert "**Runbooks/**" in files["index.md"]
        assert "One row per order." in files[f"{ids.orders}.md"]


class TestMoveRenameDelete:
    async def test_rename(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.policies}",
            json={"name": "Compliance"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "Compliance"

    async def test_rename_page_syncs_backing_model_and_search(self, api_client, kb_bank, memory, request_context):
        """Renaming a page must also rename its backing mental model so the page's
        searchable document (name + content) reflects the new name — #3307. Before
        the fix the visible name changed but the mental model kept the old name,
        leaving stale lexical/vector projections."""
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"name": "Purchase Receipts"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "Purchase Receipts"

        # The backing mental model's name is updated in the same transaction.
        mm = await memory.get_mental_model(bank_id, ids.orders_mm, request_context=request_context)
        assert mm["name"] == "Purchase Receipts"

        # The new name is now searchable (the BM25 arm indexes page name + content).
        hit = await api_client.get(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/search",
            params={"q": "purchase receipts", "limit": 5},
        )
        assert hit.status_code == 200, hit.text
        assert any(r["id"] == ids.orders for r in hit.json()["results"])

    async def test_rename_page_reembeds_backing_model(
        self, api_client, kb_bank, memory: MemoryEngine, request_context, monkeypatch
    ):
        """Regression for #3926: the rename reached the backing model's name and its
        lexical projection, but not its vector, so semantic recall kept matching the
        page on its old title."""
        bank_id, ids = kb_bank

        # No engine read exposes a model's embedding, and the stale column is the
        # defect under test, so it has to be read directly.
        async def stored_embedding() -> str:
            async with memory._pool.acquire() as conn:
                return await conn.fetchval(
                    f"SELECT embedding::text FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    ids.orders_mm,
                )

        before = await stored_embedding()

        embedded: list[str] = []
        original_generate = embedding_utils.generate_embeddings_batch

        async def recording_generate(backend, texts, *args, **kwargs):
            embedded.extend(texts)
            return await original_generate(backend, texts, *args, **kwargs)

        monkeypatch.setattr(embedding_utils, "generate_embeddings_batch", recording_generate)

        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"name": "Order Operations"},
        )
        assert resp.status_code == 200, resp.text

        mm = await memory.get_mental_model(bank_id, ids.orders_mm, request_context=request_context)
        assert embedded == [f"Order Operations {mm['content']}"]
        assert await stored_embedding() != before

    async def test_update_page_options(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={
                "source_query": "summarize every order fact and its revenue",
                "tags": ["type:runbook", "sales", "priority"],
                "max_tokens": 2048,
            },
        )
        assert resp.status_code == 200, resp.text
        node = resp.json()
        assert node["kind"] == "page"
        assert set(node["tags"]) == {"type:runbook", "sales", "priority"}
        # source_query persists — it surfaces as the `description` on the page.
        page = (await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")).json()
        assert page["description"] == "summarize every order fact and its revenue"

    async def test_update_requires_a_field(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={},
        )
        assert resp.status_code == 400

    async def test_move_into_folder(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # move the Loose root page under Policies
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.loose}",
            json={"parent_id": ids.policies},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["parent_id"] == ids.policies

    async def test_move_cycle_rejected(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # moving Runbooks under its own descendant Sub must fail
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.runbooks}",
            json={"parent_id": ids.sub},
        )
        assert resp.status_code == 400

    async def test_concurrent_opposite_moves_are_serialized(self, memory: MemoryEngine, request_context, monkeypatch):
        """The second opposite move must observe the first committed parent link.

        Without the bank lock both moves read the same pre-commit tree, both pass
        the Python cycle guard, and both commit — leaving ``A.parent = B`` and
        ``B.parent = A``.
        """
        bank_id = f"test-kb-move-race-{uuid.uuid4().hex[:8]}"
        folder_a = await memory.create_knowledge_folder(bank_id, "A", request_context=request_context)
        folder_b = await memory.create_knowledge_folder(bank_id, "B", request_context=request_context)
        pauser = _BankLockPauser(bank_id)
        pauser.install(monkeypatch)
        try:
            first = asyncio.create_task(
                memory.update_knowledge_node(
                    bank_id, folder_a["id"], parent_id=folder_b["id"], request_context=request_context
                )
            )
            await asyncio.wait_for(pauser.first_holds_lock.wait(), timeout=5)

            second = asyncio.create_task(
                memory.update_knowledge_node(
                    bank_id, folder_b["id"], parent_id=folder_a["id"], request_context=request_context
                )
            )
            await asyncio.wait_for(pauser.second_reached_lock.wait(), timeout=5)
            await _assert_blocked(second, "the second move")

            pauser.release_first.set()
            first_result = await first
            assert first_result["parent_id"] == folder_b["id"]
            with pytest.raises(ValueError, match="own subtree"):
                await second
        finally:
            pauser.release_first.set()
            await memory.delete_bank(bank_id, request_context=request_context)

    async def test_move_serializes_behind_a_concurrent_delete(self, memory: MemoryEngine, request_context, monkeypatch):
        """A move into a folder another request is deleting waits for that delete.

        Delete is a structural writer too, so it takes the same lock. The move
        then validates its destination against the committed tree and refuses
        cleanly, instead of validating a stale parent and hitting a raw foreign
        key violation on the UPDATE.
        """
        bank_id = f"test-kb-move-delete-race-{uuid.uuid4().hex[:8]}"
        folder_a = await memory.create_knowledge_folder(bank_id, "A", request_context=request_context)
        folder_b = await memory.create_knowledge_folder(bank_id, "B", request_context=request_context)
        pauser = _BankLockPauser(bank_id)
        pauser.install(monkeypatch)
        try:
            deleting = asyncio.create_task(
                memory.delete_knowledge_node(bank_id, folder_b["id"], request_context=request_context)
            )
            await asyncio.wait_for(pauser.first_holds_lock.wait(), timeout=5)

            moving = asyncio.create_task(
                memory.update_knowledge_node(
                    bank_id, folder_a["id"], parent_id=folder_b["id"], request_context=request_context
                )
            )
            await asyncio.wait_for(pauser.second_reached_lock.wait(), timeout=5)
            await _assert_blocked(moving, "the move")

            pauser.release_first.set()
            assert await deleting is True
            with pytest.raises(ValueError, match="not found"):
                await moving
        finally:
            pauser.release_first.set()
            await memory.delete_bank(bank_id, request_context=request_context)

    async def test_create_serializes_behind_a_concurrent_delete(
        self, memory: MemoryEngine, request_context, monkeypatch
    ):
        """Creating into a folder another request is deleting waits for that delete.

        Create is the third structural writer, so it takes the same lock and
        validates its parent against the committed tree — a clean rejection
        rather than an insert against a parent that is already gone.
        """
        bank_id = f"test-kb-create-delete-race-{uuid.uuid4().hex[:8]}"
        folder = await memory.create_knowledge_folder(bank_id, "Doomed", request_context=request_context)
        pauser = _BankLockPauser(bank_id)
        pauser.install(monkeypatch)
        try:
            deleting = asyncio.create_task(
                memory.delete_knowledge_node(bank_id, folder["id"], request_context=request_context)
            )
            await asyncio.wait_for(pauser.first_holds_lock.wait(), timeout=5)

            creating = asyncio.create_task(
                memory.create_knowledge_folder(
                    bank_id, "Child", parent_id=folder["id"], request_context=request_context
                )
            )
            await asyncio.wait_for(pauser.second_reached_lock.wait(), timeout=5)
            await _assert_blocked(creating, "the create")

            pauser.release_first.set()
            assert await deleting is True
            with pytest.raises(ValueError, match="not found"):
                await creating
        finally:
            pauser.release_first.set()
            await memory.delete_bank(bank_id, request_context=request_context)

    async def test_tree_walks_terminate_on_a_pre_existing_cycle(self, memory: MemoryEngine, request_context):
        """A tree corrupted before the bank lock existed must not hang a request.

        The lock stops this process from committing a parent loop, but a bank
        corrupted by the old race — or restored from an export of one — still
        carries it, and nothing in the schema forbids it. Both Python tree walks
        are bounded, so such a bank fails or deletes instead of spinning forever
        inside an open transaction.
        """
        bank_id = f"test-kb-cycle-{uuid.uuid4().hex[:8]}"
        folder_a = await memory.create_knowledge_folder(bank_id, "A", request_context=request_context)
        folder_b = await memory.create_knowledge_folder(bank_id, "B", request_context=request_context)
        folder_c = await memory.create_knowledge_folder(bank_id, "C", request_context=request_context)
        try:
            # Raw SQL because the engine API can no longer produce this state —
            # forging the loop is the whole point. A -> B -> A.
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE knowledge_pages SET parent_id = $2 WHERE bank_id = $1 AND id = $3",
                    bank_id,
                    folder_b["id"],
                    folder_a["id"],
                )
                await conn.execute(
                    "UPDATE knowledge_pages SET parent_id = $2 WHERE bank_id = $1 AND id = $3",
                    bank_id,
                    folder_a["id"],
                    folder_b["id"],
                )

            # The ancestor walk never reaches C, so only the cycle guard ends it.
            with pytest.raises(ValueError, match="cycle"):
                await asyncio.wait_for(
                    memory.update_knowledge_node(
                        bank_id, folder_c["id"], parent_id=folder_a["id"], request_context=request_context
                    ),
                    timeout=10,
                )

            # The subtree walk visits each node once and deletes what it reached.
            assert (
                await asyncio.wait_for(
                    memory.delete_knowledge_node(bank_id, folder_a["id"], request_context=request_context),
                    timeout=10,
                )
                is True
            )
            remaining = {n["id"] for n in await memory.list_knowledge_nodes(bank_id, request_context=request_context)}
            assert remaining == {folder_c["id"]}
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    async def test_patch_rolls_back_the_rename_when_the_move_fails(self, api_client, kb_bank, memory, request_context):
        """One PATCH is one transaction.

        A rename used to commit on its own connection before the move was even
        attempted, so a bad parent left the node renamed but not moved: state the
        client never asked for, and one its retry of the same PATCH could not undo.
        """
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"name": "Renamed Orders", "parent_id": "kf-does-not-exist"},
        )
        assert resp.status_code == 400, resp.text

        node = next(
            n
            for n in await memory.list_knowledge_nodes(bank_id, request_context=request_context)
            if n["id"] == ids.orders
        )
        assert node["name"] == "Orders"
        assert node["parent_id"] == ids.runbooks
        # The backing mental model is written in the same transaction, so it has to
        # roll back with the node rather than keep a name nothing points at.
        mm = await memory.get_mental_model(bank_id, ids.orders_mm, request_context=request_context)
        assert mm["name"] == "Orders"

    async def test_patch_rolls_back_the_rename_when_the_page_options_do_not_apply(
        self, api_client, kb_bank, memory, request_context
    ):
        """Same guarantee on the other ordering: page options are rejected last.

        A folder has no backing mental model, so page options cannot apply to it.
        That verdict used to arrive after the rename had already committed.
        """
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.policies}",
            json={"name": "Renamed Policies", "tags": ["type:policy"]},
        )
        assert resp.status_code == 404, resp.text

        node = next(
            n
            for n in await memory.list_knowledge_nodes(bank_id, request_context=request_context)
            if n["id"] == ids.policies
        )
        assert node["name"] == "Policies"

    async def test_patch_rolls_back_the_move_when_it_would_make_a_cycle(
        self, api_client, kb_bank, memory, request_context
    ):
        """The cycle guard fires mid-transaction; the rename beside it must not survive."""
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.runbooks}",
            json={"name": "Renamed Runbooks", "parent_id": ids.sub},
        )
        assert resp.status_code == 400, resp.text
        assert "subtree" in resp.text

        node = next(
            n
            for n in await memory.list_knowledge_nodes(bank_id, request_context=request_context)
            if n["id"] == ids.runbooks
        )
        assert node["name"] == "Runbooks"
        assert node["parent_id"] is None

    async def test_patch_applies_rename_move_and_page_options_together(
        self, api_client, kb_bank, memory, request_context
    ):
        """The whole patch commits as one, and each field still lands where it lives."""
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.loose}",
            json={
                "name": "Compliance Rules",
                "parent_id": ids.policies,
                "source_query": "what are the compliance rules?",
                "tags": ["type:policy", "compliance"],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200, resp.text
        node = resp.json()
        assert node["name"] == "Compliance Rules"
        assert node["parent_id"] == ids.policies
        assert set(node["tags"]) == {"type:policy", "compliance"}

        mm = await memory.get_mental_model(bank_id, node["mental_model_id"], request_context=request_context)
        assert mm["name"] == "Compliance Rules"
        assert mm["source_query"] == "what are the compliance rules?"
        assert set(mm["tags"]) == {"type:policy", "compliance"}
        assert mm["max_tokens"] == 1024

    async def test_patch_does_not_schedule_a_refresh_it_rolled_back(self, api_client, kb_bank, memory, monkeypatch):
        """A new source query rebuilds the page — but only if the patch committed.

        The refresh is submitted after the transaction, so a move that fails beside
        it leaves no job queued against a source query the page never took on.
        """
        bank_id, ids = kb_bank
        submitted: list[str] = []

        async def fake_submit(*, bank_id: str, mental_model_id: str, request_context):
            submitted.append(mental_model_id)

        monkeypatch.setattr(memory, "submit_async_refresh_mental_model", fake_submit)
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"source_query": "a question the page never adopts", "parent_id": "kf-does-not-exist"},
        )
        assert resp.status_code == 400, resp.text
        assert submitted == []

        # The committing case still queues exactly one rebuild.
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"source_query": "a question the page does adopt"},
        )
        assert resp.status_code == 200, resp.text
        assert submitted == [ids.orders_mm]

    async def test_delete_folder_cascades(self, api_client, kb_bank, memory, request_context):
        bank_id, ids = kb_bank
        # deleting Runbooks removes Sub + Orders (and Orders' mental model)
        resp = await api_client.delete(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.runbooks}")
        assert resp.status_code == 200, resp.text

        tree = (await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")).json()
        root_names = {r["name"] for r in tree["roots"]}
        assert "Runbooks" not in root_names
        # the backing mental model is gone too
        mm = await memory.get_mental_model(bank_id, ids.orders_mm, request_context=request_context)
        assert mm is None


class TestListReportsTheContentItDelivers:
    """A list that carries content is a bulk read of that content.

    A knowledge page is a mental_models row and is not excluded from
    list_mental_models, so a list that returns content hands back every
    page in the bank. Reporting the single-page read while leaving that
    unreported would watch the narrow door and leave the wide one open — a
    caller could enumerate a bank's synthesized knowledge by asking for it in
    one page instead of one at a time.
    """

    async def test_listing_with_content_reports_each_model(self, memory, kb_bank, request_context, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        page = await memory.list_mental_models(bank_id=bank_id, detail="content", request_context=request_context)

        with_content = [m for m in page.items if m.get("content")]
        assert with_content, "fixture should have at least one model with content"
        assert sorted(validator.model_reads) == sorted(str(m["id"]) for m in with_content)

    async def test_the_page_backing_model_is_among_them(self, memory, kb_bank, request_context, monkeypatch):
        # The specific hole: pages ride in on this list.
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        await memory.list_mental_models(bank_id=bank_id, detail="content", request_context=request_context)

        assert ids.orders_mm in validator.model_reads

    async def test_listing_metadata_reports_nothing(self, memory, kb_bank, request_context, monkeypatch):
        # No content delivered, nothing to report — and this is the detail
        # level the internal template-provisioning callers now ask for.
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        page = await memory.list_mental_models(bank_id=bank_id, detail="metadata", request_context=request_context)

        assert page.items, "metadata listing should still return the models"
        assert validator.model_reads == []

    async def test_reported_size_tracks_the_content_returned(self, memory, kb_bank, request_context, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        page = await memory.list_mental_models(bank_id=bank_id, detail="content", request_context=request_context)

        expected = sorted(len(m["content"]) // 4 for m in page.items if m.get("content"))
        assert sorted(validator.model_get_tokens) == expected

    async def test_config_detail_carries_the_definition_without_the_body(
        self, memory, kb_bank, request_context, monkeypatch
    ):
        # The level the bank-template export asks for: it copies how a model is
        # built and never reads what it says, so it must not pull — or be
        # reported for — content it discards.
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        page = await memory.list_mental_models(bank_id=bank_id, detail="config", request_context=request_context)

        assert page.items, "config listing should still return the models"
        assert all("content" not in m for m in page.items)
        # The fields a template manifest is built from survive the trim.
        assert all(m["source_query"] and "trigger" in m and "max_tokens" in m for m in page.items)
        assert validator.model_reads == []

    async def test_a_model_still_generating_is_not_reported(self, memory, kb_bank, request_context, monkeypatch):
        # The placeholder is not synthesized knowledge; nothing was delivered.
        bank_id, ids = kb_bank
        pending = await memory.create_mental_model(
            bank_id=bank_id,
            name="Pending",
            source_query="Not answered yet.",
            content=MENTAL_MODEL_PENDING_CONTENT,
            request_context=request_context,
        )
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        page = await memory.list_mental_models(bank_id=bank_id, detail="content", request_context=request_context)

        assert any(m["id"] == pending["id"] for m in page.items), "the pending model should still be listed"
        assert pending["id"] not in validator.model_reads


class TestExportReportsOnceNotPerPage:
    """An export is a single named operation, not N model reads.

    export_knowledge_base runs its per-page reads under
    _authorize_nested_operations, so the whole bundle costs exactly one
    EXPORT_KNOWLEDGE_BASE hook. Pinned here because it is the widest read of
    model content in the engine, and the suppression that keeps it to one hook
    is invisible at the call site.
    """

    async def test_export_reports_one_hook_for_the_whole_bundle(self, memory, kb_bank, request_context, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        export = await memory.export_knowledge_base(bank_id=bank_id, request_context=request_context)

        assert len(export.pages) >= 3, "fixture seeds three pages"
        assert _read_ops(validator) == [BankReadOperation.EXPORT_KNOWLEDGE_BASE]
        # The nested page and list reads inside it report nothing of their own.
        assert validator.model_gets == []
        assert validator.model_reads == []


class TestPageReadIsAModelRead:
    """Reading a page runs the same validator pair as reading its mental model.

    get_knowledge_page joins mental_models and returns mm.content as
    the page body, so a page read delivers exactly what get_mental_model
    delivers off the same row. The two endpoints are doors onto one object, and
    a deployment that meters model reads must see both — otherwise the price of
    the same content depends on which URL the caller picked.
    """

    async def test_reading_a_page_runs_the_model_get_validator(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")

        assert resp.status_code == 200, resp.text
        # Gated on the page's backing model, not on the page id.
        assert validator.model_gets == [ids.orders_mm]

    async def test_reading_a_page_reports_completion_with_the_content_size(
        self, api_client, kb_bank, memory, monkeypatch
    ):
        # The post-hook is what records usage; without it a deployment could
        # gate a read it then never bills for.
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")
        body = resp.json()

        # The response renames the model's content to body; the hook is
        # given the same string, so the recorded size must track it.
        assert len(validator.model_get_tokens) == 1
        assert validator.model_get_tokens[0] == len(body.get("body") or "") // 4
        assert validator.model_get_tokens[0] > 0

    async def test_a_refused_model_get_returns_no_page_content(self, api_client, kb_bank, memory, monkeypatch):
        # The gate has to run before the body is handed back, or it is
        # decoration: the caller would get the content and the refusal.
        bank_id, ids = kb_bank
        validator = _kb_validator()
        validator.reject_model_get = True
        monkeypatch.setattr(memory, "_operation_validator", validator)

        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")

        assert resp.status_code != 200
        assert "Orders" not in resp.text
        # Refused before delivery means nothing to record.
        assert validator.model_get_tokens == []

    async def test_bank_read_access_check_still_runs(self, api_client, kb_bank, memory, monkeypatch):
        # Metering is added alongside the access check, not in place of it.
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        validator.read_ops.clear()
        await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")

        assert _read_ops(validator) == [BankReadOperation.GET_KNOWLEDGE_PAGE]


class TestAuthorizationReadDenied:
    """A validator that denies a knowledge-base read blocks it with 403 and leaks
    nothing — knowledge pages render mental-model content, so this is the sharp
    edge of #3312 (read-your-neighbour's-synthesized-memories)."""

    async def test_tree_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_read=BankReadOperation.GET_KNOWLEDGE_BASE_TREE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        assert resp.status_code == 403, resp.text
        assert "Orders" not in resp.text
        assert _read_ops(validator) == [BankReadOperation.GET_KNOWLEDGE_BASE_TREE]

    async def test_get_page_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_read=BankReadOperation.GET_KNOWLEDGE_PAGE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")
        assert resp.status_code == 403, resp.text
        assert "One row per order." not in resp.text
        assert _read_ops(validator) == [BankReadOperation.GET_KNOWLEDGE_PAGE]

    async def test_search_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_read=BankReadOperation.SEARCH_KNOWLEDGE_BASE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.get(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/search",
            params={"q": "orders"},
        )
        assert resp.status_code == 403, resp.text
        assert "Orders" not in resp.text
        assert _read_ops(validator) == [BankReadOperation.SEARCH_KNOWLEDGE_BASE]

    async def test_export_denied_leaks_nothing_and_gates_once(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_read=BankReadOperation.EXPORT_KNOWLEDGE_BASE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/export")
        assert resp.status_code == 403, resp.text
        assert "One row per order." not in resp.text
        # A single export read gate — the per-page reads never run on a denied path.
        assert _read_ops(validator) == [BankReadOperation.EXPORT_KNOWLEDGE_BASE]


class TestAuthorizationWriteDenied:
    """A validator that denies a knowledge-base write blocks it with 403 and leaves
    the tree unchanged."""

    async def _tree_names(self, api_client, bank_id) -> set[str]:
        tree = (await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")).json()

        def walk(nodes):
            for n in nodes:
                yield n["name"]
                yield from walk(n.get("children", []))

        return set(walk(tree["roots"]))

    async def test_create_folder_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        before = await self._tree_names(api_client, bank_id)
        validator = _kb_validator(reject_write=BankWriteOperation.CREATE_KNOWLEDGE_FOLDER)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders",
            json={"name": "Guides", "parent_id": None},
        )
        assert resp.status_code == 403, resp.text
        assert _write_ops(validator) == [BankWriteOperation.CREATE_KNOWLEDGE_FOLDER]
        monkeypatch.setattr(memory, "_operation_validator", None)
        assert await self._tree_names(api_client, bank_id) == before

    async def test_create_page_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        before = await self._tree_names(api_client, bank_id)
        validator = _kb_validator(reject_write=BankWriteOperation.CREATE_KNOWLEDGE_PAGE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages",
            json={"name": "New page", "source_query": "what is new?"},
        )
        assert resp.status_code == 403, resp.text
        # Rejected before the backing mental model is created — a single write hook.
        assert _write_ops(validator) == [BankWriteOperation.CREATE_KNOWLEDGE_PAGE]
        monkeypatch.setattr(memory, "_operation_validator", None)
        assert await self._tree_names(api_client, bank_id) == before

    async def test_rename_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_write=BankWriteOperation.RENAME_KNOWLEDGE_NODE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.policies}",
            json={"name": "Compliance"},
        )
        assert resp.status_code == 403, resp.text
        assert _write_ops(validator) == [BankWriteOperation.RENAME_KNOWLEDGE_NODE]
        monkeypatch.setattr(memory, "_operation_validator", None)
        assert "Policies" in await self._tree_names(api_client, bank_id)

    async def test_move_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_write=BankWriteOperation.MOVE_KNOWLEDGE_NODE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.loose}",
            json={"parent_id": ids.policies},
        )
        assert resp.status_code == 403, resp.text
        assert _write_ops(validator) == [BankWriteOperation.MOVE_KNOWLEDGE_NODE]

    async def test_update_page_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_write=BankWriteOperation.UPDATE_KNOWLEDGE_PAGE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"source_query": "changed"},
        )
        assert resp.status_code == 403, resp.text
        # Rejected before touching the backing mental model — a single write hook.
        assert _write_ops(validator) == [BankWriteOperation.UPDATE_KNOWLEDGE_PAGE]

    async def test_delete_denied(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator(reject_write=BankWriteOperation.DELETE_KNOWLEDGE_NODE)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.delete(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.runbooks}")
        assert resp.status_code == 403, resp.text
        assert _write_ops(validator) == [BankWriteOperation.DELETE_KNOWLEDGE_NODE]
        monkeypatch.setattr(memory, "_operation_validator", None)
        assert "Runbooks" in await self._tree_names(api_client, bank_id)

    async def test_denied_create_leaves_no_bank_behind(self, api_client, memory, request_context, monkeypatch):
        """An unauthorized create must not lazily provision the target bank."""
        bank_id = f"kb-denied-create-{uuid.uuid4().hex[:8]}"
        validator = _kb_validator(reject_write=BankWriteOperation.CREATE_KNOWLEDGE_FOLDER)
        monkeypatch.setattr(memory, "_operation_validator", validator)
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders",
            json={"name": "Guides", "parent_id": None},
        )
        assert resp.status_code == 403, resp.text
        monkeypatch.setattr(memory, "_operation_validator", None)
        assert await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False) is None


class TestAuthorizationSuccessHookCounts:
    """Successful knowledge-base routes invoke exactly one validator hook — the
    knowledge-base operation — and never the nested mental-model hooks."""

    async def test_reads_gate_once(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        validator.read_ops.clear()
        await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        assert _read_ops(validator) == [BankReadOperation.GET_KNOWLEDGE_BASE_TREE]

        validator.read_ops.clear()
        await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")
        assert _read_ops(validator) == [BankReadOperation.GET_KNOWLEDGE_PAGE]

        validator.read_ops.clear()
        await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/search", params={"q": "orders"})
        assert _read_ops(validator) == [BankReadOperation.SEARCH_KNOWLEDGE_BASE]

    async def test_export_gates_once_and_suppresses_nested_reads(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)
        validator.read_ops.clear()
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/export")
        assert resp.status_code == 200, resp.text
        # Exactly one gate for the whole bundle; the per-page reads run under it.
        assert _read_ops(validator) == [BankReadOperation.EXPORT_KNOWLEDGE_BASE]

    async def test_create_page_gates_once_without_nested_mental_model_write(
        self, kb_bank, memory, request_context, monkeypatch
    ):
        # Tested at the engine level: the HTTP route additionally schedules an
        # async refresh whose background worker later writes the generated content
        # (a separate, legitimately metered UPDATE_MENTAL_MODEL). Here we assert the
        # synchronous create in isolation — the backing CREATE_MENTAL_MODEL is
        # suppressed, so the KB write is the only bank_write hook.
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)
        validator.write_ops.clear()
        node = await memory.create_knowledge_page(
            bank_id,
            "Fresh page",
            "what is fresh?",
            "content",
            request_context=request_context,
        )
        assert node is not None
        assert _write_ops(validator) == [BankWriteOperation.CREATE_KNOWLEDGE_PAGE]

    async def test_writes_gate_once(self, api_client, kb_bank, memory, monkeypatch):
        bank_id, ids = kb_bank
        validator = _kb_validator()
        monkeypatch.setattr(memory, "_operation_validator", validator)

        validator.write_ops.clear()
        await api_client.post(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders", json={"name": "Guides"})
        assert _write_ops(validator) == [BankWriteOperation.CREATE_KNOWLEDGE_FOLDER]

        validator.write_ops.clear()
        await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.policies}",
            json={"name": "Compliance"},
        )
        assert _write_ops(validator) == [BankWriteOperation.RENAME_KNOWLEDGE_NODE]

        validator.write_ops.clear()
        await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.orders}",
            json={"tags": ["type:runbook"]},
        )
        assert _write_ops(validator) == [BankWriteOperation.UPDATE_KNOWLEDGE_PAGE]

        validator.write_ops.clear()
        await api_client.delete(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.billing}")
        assert _write_ops(validator) == [BankWriteOperation.DELETE_KNOWLEDGE_NODE]


class TestAuthorizationDisabled:
    """Without an operation validator (OSS default), knowledge-base routes are
    unauthenticated-by-tenant and work exactly as before."""

    async def test_engine_calls_do_not_raise(self, memory, request_context):
        assert memory._operation_validator is None
        bank_id = f"kb-noauth-{uuid.uuid4().hex[:8]}"
        folder = await memory.create_knowledge_folder(bank_id, "Docs", request_context=request_context)
        nodes = await memory.list_knowledge_nodes(bank_id=bank_id, request_context=request_context)
        assert folder["id"] in {n["id"] for n in nodes}
        export = await memory.export_knowledge_base(bank_id=bank_id, request_context=request_context)
        assert any(n["id"] == folder["id"] for n in export.nodes)
        await memory.delete_bank(bank_id, request_context=request_context)
