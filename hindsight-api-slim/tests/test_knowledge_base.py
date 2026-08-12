"""HTTP + engine integration tests for the knowledge base (folders + pages).

Pages are seeded directly via the engine (deterministic content, no LLM) so the
tree, markdown rendering, move/rename, and cascade-delete behaviour can be asserted
without consolidation.
"""

import urllib.parse
import uuid
from datetime import datetime, timedelta, timezone

import pytest_asyncio

from hindsight_api.engine.memory_engine import MemoryEngine, _may_need_refresh
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

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

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

    async def test_tree_staleness_follows_the_bank_watermark(self, api_client, memory, kb_bank):
        """The tree answers from one bank-wide watermark, not a scan per page.

        The page-level answer is therefore conservative: an untagged memory flips
        even the tagged pages to "may need refresh", though a scoped check would
        call them current. That is the trade — the exact answer costs a full scan
        of the bank's memories per page, on a view that polls.
        """
        bank_id, ids = kb_bank
        scoped_checks = 0
        original = memory.compute_mental_model_is_stale

        async def counting_check(*args, **kwargs):
            nonlocal scoped_checks
            scoped_checks += 1
            return await original(*args, **kwargs)

        memory.compute_mental_model_is_stale = counting_check
        try:
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO memory_units (id, bank_id, text, fact_type, created_at) "
                    "VALUES (gen_random_uuid(), $1, $2, 'experience', now())",
                    bank_id,
                    "An untagged memory, outside every page's tags.",
                )
            # The watermark is served from the stats cache, so a write inside the
            # TTL only shows up once that entry expires or is dropped.
            await memory._bank_stats_cache.clear()

            resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
            assert resp.status_code == 200, resp.text
            roots = {r["name"]: r for r in resp.json()["roots"]}
            orders = next(c for c in roots["Runbooks"]["children"] if c["name"] == "Orders")
            assert roots["Loose"]["is_stale"] is True
            assert orders["is_stale"] is True, "tagged pages are flagged too — the watermark is bank-wide"
            assert scoped_checks == 0, "the tree must not run a scoped staleness query per page"
        finally:
            memory.compute_mental_model_is_stale = original
            await memory._bank_stats_cache.clear()


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
