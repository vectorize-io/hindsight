"""Tests for the knowledge-base folder curator.

The op-application logic is deterministic and tested directly with hand-built
plans (no LLM). The model's *decision* (which ops to emit) is covered by a
separate hs_llm_core judge test in test_knowledge_curator_llm.py.
"""

import uuid

import pytest_asyncio

from hindsight_api.engine import knowledge_curator as kc
from hindsight_api.engine.knowledge_curator import CuratorOp, CuratorPlan
from hindsight_api.engine.memory_engine import MemoryEngine


@pytest_asyncio.fixture
async def folder_bank(memory: MemoryEngine, request_context):
    """A bank with one folder containing a managed page and a pinned page."""
    bank_id = f"test-kc-{uuid.uuid4().hex[:8]}"
    folder = await memory.create_knowledge_folder(
        bank_id, "Incidents", mission="Track payment incidents", request_context=request_context
    )
    managed = await memory.create_knowledge_page(
        bank_id,
        "Old outage",
        "What happened in the old outage?",
        "# Old\n\nstale",
        parent_id=folder["id"],
        managed=True,
        request_context=request_context,
    )
    pinned = await memory.create_knowledge_page(
        bank_id,
        "Runbook",
        "How to recover?",
        "# Runbook\n\npinned",
        parent_id=folder["id"],
        managed=False,
        request_context=request_context,
    )
    yield memory, bank_id, folder["id"], managed["id"], pinned["id"]
    await memory.delete_bank(bank_id, request_context=request_context)


def _names(nodes, parent_id):
    return {n["name"] for n in nodes if n.get("parent_id") == parent_id and n["kind"] == "page"}


class TestApplyPlan:
    async def test_create_page(self, folder_bank, request_context):
        memory, bank_id, folder_id, _managed, _pinned = folder_bank
        plan = CuratorPlan(
            operations=[CuratorOp(action="create_page", name="Checkout outage", source_query="What happened?")]
        )
        result = await kc.apply_curator_plan(memory, bank_id, folder_id, plan, request_context=request_context)
        assert [a.action for a in result.applied] == ["create_page"]
        nodes = await memory.list_knowledge_nodes(bank_id, request_context=request_context)
        created = next(n for n in nodes if n["name"] == "Checkout outage")
        assert created["managed"] is True

    async def test_delete_only_managed(self, folder_bank, request_context):
        memory, bank_id, folder_id, managed_id, pinned_id = folder_bank
        plan = CuratorPlan(
            operations=[
                CuratorOp(action="delete_page", page_id=managed_id, reason="stale"),
                CuratorOp(action="delete_page", page_id=pinned_id, reason="should be skipped"),
            ]
        )
        result = await kc.apply_curator_plan(memory, bank_id, folder_id, plan, request_context=request_context)
        assert [a.action for a in result.applied] == ["delete_page"]
        assert [a.action for a in result.skipped] == ["delete_page"]
        nodes = await memory.list_knowledge_nodes(bank_id, request_context=request_context)
        names = _names(nodes, folder_id)
        assert "Old outage" not in names  # managed → deleted
        assert "Runbook" in names  # pinned → untouched

    async def test_merge_pages(self, folder_bank, request_context):
        memory, bank_id, folder_id, managed_id, _pinned = folder_bank
        # add a second managed page so there are two to merge
        second = await memory.create_knowledge_page(
            bank_id,
            "Old outage dup",
            "dup?",
            "# dup",
            parent_id=folder_id,
            managed=True,
            request_context=request_context,
        )
        plan = CuratorPlan(
            operations=[
                CuratorOp(
                    action="merge_pages",
                    page_ids=[managed_id, second["id"]],
                    name="Outage summary",
                    source_query="Summarize the outages.",
                )
            ]
        )
        result = await kc.apply_curator_plan(memory, bank_id, folder_id, plan, request_context=request_context)
        assert [a.action for a in result.applied] == ["merge_pages"]
        nodes = await memory.list_knowledge_nodes(bank_id, request_context=request_context)
        names = _names(nodes, folder_id)
        assert "Outage summary" in names
        assert "Old outage" not in names and "Old outage dup" not in names

    async def test_create_subfolder(self, folder_bank, request_context):
        memory, bank_id, folder_id, _managed, _pinned = folder_bank
        plan = CuratorPlan(
            operations=[CuratorOp(action="create_subfolder", name="Checkout", mission="Checkout incidents only")]
        )
        result = await kc.apply_curator_plan(memory, bank_id, folder_id, plan, request_context=request_context)
        assert [a.action for a in result.applied] == ["create_subfolder"]
        nodes = await memory.list_knowledge_nodes(bank_id, request_context=request_context)
        sub = next(n for n in nodes if n["name"] == "Checkout" and n["kind"] == "folder")
        assert sub["parent_id"] == folder_id
        assert sub["mission"] == "Checkout incidents only"
        assert sub["managed"] is True

    async def test_subfolder_depth_bound(self, folder_bank, request_context):
        memory, bank_id, folder_id, _managed, _pinned = folder_bank
        # Build a chain to MAX_DEPTH so a spawn at the bottom is rejected.
        parent = folder_id
        for i in range(kc.MAX_DEPTH):
            child = await memory.create_knowledge_folder(
                bank_id, f"L{i}", parent_id=parent, request_context=request_context
            )
            parent = child["id"]
        plan = CuratorPlan(operations=[CuratorOp(action="create_subfolder", name="TooDeep", mission="x")])
        result = await kc.apply_curator_plan(memory, bank_id, parent, plan, request_context=request_context)
        assert result.applied == []
        assert any("bounds" in (a.detail or "") for a in result.skipped)

    async def test_dry_run_applies_nothing(self, folder_bank, request_context):
        memory, bank_id, folder_id, managed_id, _pinned = folder_bank
        plan = CuratorPlan(operations=[CuratorOp(action="delete_page", page_id=managed_id)])
        result = await kc.apply_curator_plan(
            memory, bank_id, folder_id, plan, request_context=request_context, dry_run=True
        )
        assert [a.action for a in result.applied] == ["delete_page"]
        nodes = await memory.list_knowledge_nodes(bank_id, request_context=request_context)
        assert "Old outage" in _names(nodes, folder_id)  # not actually deleted


class TestCurateGating:
    async def test_no_mission_is_noop(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kc-nm-{uuid.uuid4().hex[:8]}"
        folder = await memory.create_knowledge_folder(bank_id, "NoMission", request_context=request_context)
        result = await memory.curate_folder(bank_id, folder["id"], request_context=request_context)
        assert result.applied == [] and result.skipped == []
        assert result.mission is None
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_set_folder_mission(self, memory: MemoryEngine, request_context):
        bank_id = f"test-kc-sm-{uuid.uuid4().hex[:8]}"
        folder = await memory.create_knowledge_folder(bank_id, "F", request_context=request_context)
        updated = await memory.set_folder_mission(
            bank_id, folder["id"], "Collect release notes", request_context=request_context
        )
        assert updated["mission"] == "Collect release notes"
        await memory.delete_bank(bank_id, request_context=request_context)


class TestDuplicatePageGuard:
    async def test_duplicate_page_name_blocked(self, memory: MemoryEngine, request_context):
        """Two pages with the same name in one folder: the second is rejected (None).

        This is the deterministic guard that stops concurrent curator runs from
        creating duplicate pages (e.g. two 'Anna' pages), independent of the LLM.
        """
        bank_id = f"test-kc-dup-{uuid.uuid4().hex[:8]}"
        folder = await memory.create_knowledge_folder(
            bank_id, "people", mission="track all people", request_context=request_context
        )
        first = await memory.create_knowledge_page(
            bank_id,
            "Anna",
            "Who is Anna?",
            "# Anna",
            parent_id=folder["id"],
            managed=True,
            request_context=request_context,
        )
        assert first is not None
        # Same name (any case) in the same folder → blocked.
        dup = await memory.create_knowledge_page(
            bank_id,
            "anna",
            "Who is anna?",
            "# anna",
            parent_id=folder["id"],
            managed=True,
            request_context=request_context,
        )
        assert dup is None, "duplicate page name should be rejected"

        nodes = await memory.list_knowledge_nodes(bank_id, request_context=request_context)
        annas = [n for n in nodes if n["kind"] == "page" and n["name"].lower() == "anna"]
        assert len(annas) == 1, f"expected 1 Anna page, got {len(annas)}"

        # A page with the same name in a DIFFERENT folder is allowed.
        other = await memory.create_knowledge_folder(bank_id, "other", request_context=request_context)
        ok = await memory.create_knowledge_page(
            bank_id, "Anna", "q", "# a", parent_id=other["id"], managed=True, request_context=request_context
        )
        assert ok is not None
        await memory.delete_bank(bank_id, request_context=request_context)
