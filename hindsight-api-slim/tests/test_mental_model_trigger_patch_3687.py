"""Updating one trigger setting must not reset the others — issue #3687.

`_merge_trigger` landed with #3506, but only the two knowledge-page routes were
wired to it. `PATCH /mental-models/{id}` kept dumping the whole request model and
the engine kept writing that dict wholesale, so a client that set one field also
stamped `MentalModelTrigger`'s own defaults over everything it left unset.

That is the route the control plane uses to edit a knowledge page's advanced
options, and the MCP `update_mental_model` tool drives it with a literal one-key
dict (`{"refresh_after_consolidation": ...}`). Either one silently turned a page
from an observation-only delta document into a from-scratch rebuild over every
fact type that also reflected on its sibling pages — the page still refreshed, so
nothing surfaced the change until the content came back different.

The assertions read the trigger back through the public API rather than the
`mental_models` row, since the stored trigger is exactly what the API reports.
"""

import uuid

import pytest

from hindsight_api import MemoryEngine, RequestContext

pytestmark = pytest.mark.asyncio

# What a knowledge page is created with, and therefore everything a one-field
# patch has to leave standing.
PAGE_TRIGGER = {
    "mode": "delta",
    "fact_types": ["observation"],
    "exclude_mental_models": True,
    "refresh_after_consolidation": True,
}


async def _page(memory: MemoryEngine, request_context: RequestContext) -> tuple[str, str]:
    bank_id = f"mmpatch-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    node = await memory.create_knowledge_page(
        bank_id=bank_id,
        name="Homelab Infrastructure",
        source_query="NAS, ThinkPad, docker containers, jellyfin",
        content="Generating content...",
        tags=["type:runbook", "homelab"],
        request_context=request_context,
    )
    assert node is not None
    return bank_id, node["mental_model_id"]


async def _trigger_of(memory: MemoryEngine, bank_id: str, mm_id: str, request_context: RequestContext) -> dict:
    model = await memory.get_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=request_context)
    assert model is not None
    return model["trigger"]


class TestMentalModelTriggerPatch:
    async def test_setting_tag_groups_keeps_the_page_defaults(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The #3687 case: widening a page's scope must not rebuild it from scratch."""
        bank_id, mm_id = await _page(memory, request_context)
        tag_groups = [{"or": [{"tags": ["homelab"]}, {"tags": ["infra"], "match": "all_strict"}]}]

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm_id,
            trigger={"tag_groups": tag_groups},
            request_context=request_context,
        )

        trigger = await _trigger_of(memory, bank_id, mm_id, request_context)
        assert trigger["tag_groups"] == tag_groups
        for key, value in PAGE_TRIGGER.items():
            assert trigger[key] == value, f"{key} was reset by an unrelated trigger patch"

    async def test_setting_tags_match_keeps_the_page_defaults(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id, mm_id = await _page(memory, request_context)

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm_id,
            trigger={"tags_match": "all"},
            request_context=request_context,
        )

        trigger = await _trigger_of(memory, bank_id, mm_id, request_context)
        assert trigger["tags_match"] == "all"
        for key, value in PAGE_TRIGGER.items():
            assert trigger[key] == value, f"{key} was reset by an unrelated trigger patch"

    async def test_mcp_style_single_key_patch_keeps_the_rest(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The exact shape the MCP `update_mental_model` tool sends."""
        bank_id, mm_id = await _page(memory, request_context)

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm_id,
            trigger={"refresh_after_consolidation": True},
            request_context=request_context,
        )

        trigger = await _trigger_of(memory, bank_id, mm_id, request_context)
        assert trigger["mode"] == "delta"
        assert trigger["fact_types"] == ["observation"]
        assert trigger["exclude_mental_models"] is True

    async def test_a_full_trigger_still_replaces(self, memory: MemoryEngine, request_context: RequestContext):
        """Merging must not make a complete trigger un-settable.

        Bank-template import sends a full dump on purpose — every key present, so
        the merge overwrites everything and the manifest stays declarative.
        """
        bank_id, mm_id = await _page(memory, request_context)

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm_id,
            trigger={"mode": "full", "fact_types": ["world", "experience"], "exclude_mental_models": False},
            request_context=request_context,
        )

        trigger = await _trigger_of(memory, bank_id, mm_id, request_context)
        assert trigger["mode"] == "full"
        assert trigger["fact_types"] == ["world", "experience"]
        assert trigger["exclude_mental_models"] is False

    async def test_moving_onto_a_cron_clears_the_consolidation_trigger(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The pair stays mutually exclusive through a patch, as it does on pages."""
        bank_id, mm_id = await _page(memory, request_context)

        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm_id,
            trigger={"refresh_cron": "0 3 * * *"},
            request_context=request_context,
        )

        trigger = await _trigger_of(memory, bank_id, mm_id, request_context)
        assert trigger["refresh_cron"] == "0 3 * * *"
        assert not trigger.get("refresh_after_consolidation")
        # The scope settings are unrelated to the schedule and must survive it.
        assert trigger["fact_types"] == ["observation"]
