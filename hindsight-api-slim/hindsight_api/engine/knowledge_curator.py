"""Folder curator: mission-driven page maintenance for the knowledge base.

After a consolidation, a folder that has a ``mission`` recalls the memories
relevant to that mission, looks at its current pages, and asks the LLM to emit a
small set of ops — create a page, merge pages, delete a page, or spawn a
sub-folder. Curators are **independent**: each folder pulls over the bank's
memories on its own (the tree is for organization, not routing), so a memory can
surface in more than one folder.

Safety: only ``managed`` pages (curator-created) are ever merged or deleted;
human-authored (pinned) pages are never touched. Sub-folder spawning is bounded
by depth and count. Page *content* is still synthesized by the normal mental-
model refresh — the curator only decides which pages should exist.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..models import RequestContext
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# Bounds (kept conservative; the curator should converge, not churn).
MAX_DEPTH = 3
MAX_SUBFOLDERS = 8
DEFAULT_MAX_OPS = 8
# How many new memories to feed the curator per run (newest first). The curator
# reads the delta — memories created since the folder's last curation — not a
# semantic recall, so this caps the LLM context, not relevance.
CURATOR_MAX_MEMORIES = 200
MAX_FOLDERS_PER_RUN = 20

# Trigger for curator-created pages: each page is a living document synthesized
# from the consolidated **observations** (not raw facts), refreshed incrementally
# (delta) after each consolidation, and excluding other mental models so a page
# never reflects on its siblings (which produced the "based_on: mental-models"
# / "no information" pollution we saw).
CURATOR_PAGE_TRIGGER = {
    "mode": "delta",
    "fact_types": ["observation"],
    "exclude_mental_models": True,
    "refresh_after_consolidation": True,
}

CURATOR_SYSTEM_PROMPT = """\
You curate ONE folder of a knowledge base. The folder has a MISSION that defines exactly
what belongs in it. You are given the folder's CURRENT PAGES (which already exist) and the
NEW MEMORIES added since the last curation. Output the minimal set of operations.

Work in two steps:
1. FILTER: keep only the new memories that clearly advance THIS folder's mission. Discard
   everything else — a memory about another topic is NOT this folder's job, even if it is
   interesting. If a memory does not fit the mission, ignore it completely.
2. For the kept memories, choose operations.

Operations (emit JSON {{"operations": [...]}}; at most {max_ops}):
- create_page: a mission-relevant topic that is NOT already one of the CURRENT PAGES.
  Provide "name" and "source_query" (the question that rebuilds the page).
- merge_pages: two CURRENT PAGES cover the same topic. Provide "page_ids" + the "name"
  and "source_query" of the single replacement.
- delete_page: a CURRENT PAGE no longer fits the mission. Provide "page_id" + "reason".
- create_subfolder: the topics split into a clear sub-theme. Provide "name" + "mission".

Hard rules:
- ON MISSION ONLY. Do NOT create a page for a memory that doesn't fit the mission. Creating
  an off-topic page is the worst mistake you can make.
- NO DUPLICATES. Before creating, check CURRENT PAGES — if the topic is already there,
  do NOT recreate it. Never emit two create_page ops for the same subject in one response.
- One page = one topic. Merge overlapping topics instead of adding near-duplicates.
- Be conservative: when unsure whether something fits or is already covered, do nothing.
- Ground every page in the given memories; never invent facts. Use the exact page ids shown.
- Return {{"operations": []}} when the folder already reflects the new memories.
"""


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO timestamp (or pass through a datetime); None on failure/empty."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class CuratorOp(BaseModel):
    """One curator operation. Flat schema (no discriminated union) for provider portability."""

    action: Literal["create_page", "merge_pages", "delete_page", "create_subfolder"]
    name: str | None = Field(default=None, description="Title for create_page / merge result / create_subfolder")
    source_query: str | None = Field(default=None, description="Question that rebuilds a page")
    mission: str | None = Field(default=None, description="Mission for a spawned sub-folder")
    page_ids: list[str] = Field(default_factory=list, description="Pages to merge")
    page_id: str | None = Field(default=None, description="Page to delete")
    reason: str | None = Field(default=None, description="Short justification")


class CuratorPlan(BaseModel):
    """The LLM's proposed set of operations for a folder."""

    operations: list[CuratorOp] = Field(default_factory=list)


class CuratorAction(BaseModel):
    """An applied or skipped op, for the result audit trail."""

    action: str
    target: str | None = None
    detail: str | None = None


class CuratorResult(BaseModel):
    """Outcome of curating one folder."""

    folder_id: str
    mission: str | None = None
    applied: list[CuratorAction] = Field(default_factory=list)
    skipped: list[CuratorAction] = Field(default_factory=list)


def parse_plan(raw: Any) -> CuratorPlan:
    """Coerce an LLM response (model / dict / JSON string) into a CuratorPlan."""
    if isinstance(raw, CuratorPlan):
        return raw
    if isinstance(raw, dict):
        try:
            return CuratorPlan.model_validate(raw)
        except Exception:
            return CuratorPlan()
    if isinstance(raw, str):
        try:
            return CuratorPlan.model_validate(json.loads(raw))
        except Exception:
            return CuratorPlan()
    return CuratorPlan()


def _depth(folder_id: str, by_id: dict[str, dict[str, Any]]) -> int:
    """Depth of a node from the root (root nodes are depth 0)."""
    depth = 0
    cursor = by_id.get(folder_id, {}).get("parent_id")
    while cursor is not None:
        depth += 1
        cursor = by_id.get(cursor, {}).get("parent_id")
    return depth


async def apply_curator_plan(
    engine: "MemoryEngine",
    bank_id: str,
    folder_id: str,
    plan: CuratorPlan,
    *,
    request_context: "RequestContext",
    max_ops: int = DEFAULT_MAX_OPS,
    dry_run: bool = False,
) -> CuratorResult:
    """Apply a curator plan to a folder, enforcing safety + bounds.

    Re-reads the folder's current state so the same logic serves both the live
    curator and deterministic tests (hand-built plans). Only ``managed`` pages are
    merged/deleted; sub-folder spawning is capped by depth and count.
    """
    nodes = await engine.list_knowledge_nodes(bank_id, request_context=request_context)
    by_id = {n["id"]: n for n in nodes}
    folder = by_id.get(folder_id)
    if folder is None or folder.get("kind") != "folder":
        raise ValueError(f"Folder '{folder_id}' not found")

    managed_page_ids = {
        n["id"] for n in nodes if n.get("parent_id") == folder_id and n["kind"] == "page" and n.get("managed")
    }
    child_folder_count = sum(1 for n in nodes if n.get("parent_id") == folder_id and n["kind"] == "folder")
    depth = _depth(folder_id, by_id)

    # Deterministic dedup guard: small models routinely emit near-duplicate
    # create_page ops (and re-create existing pages) despite the prompt, so we
    # drop any create whose title collides with an existing page in this folder
    # or one already created in this run. Titles are matched case-insensitively.
    existing_titles = {
        (n["name"] or "").strip().lower() for n in nodes if n.get("parent_id") == folder_id and n["kind"] == "page"
    }

    result = CuratorResult(folder_id=folder_id, mission=folder.get("mission"))
    spawned = 0

    async def _new_page(name: str, source_query: str) -> bool:
        """Create a managed page; return False if it already exists (dup name)."""
        node = await engine.create_knowledge_page(
            bank_id,
            name,
            source_query,
            "Generating content...",
            parent_id=folder_id,
            managed=True,
            trigger=dict(CURATOR_PAGE_TRIGGER),
            request_context=request_context,
        )
        if node is None:
            # A concurrent run already created this page (DB uniqueness guard).
            return False
        # Generate the content in the background, like the manual create path.
        # Best-effort: the page already exists; a refresh hiccup must not abort the op.
        try:
            await engine.submit_async_refresh_mental_model(
                bank_id, node["mental_model_id"], request_context=request_context
            )
        except Exception as e:
            logger.warning("Refresh scheduling failed for new page %s: %s", node["id"], e)
        return True

    for op in plan.operations:
        if len(result.applied) >= max_ops:
            result.skipped.append(CuratorAction(action=op.action, detail="max ops reached"))
            continue

        if op.action == "create_page":
            if not op.name or not op.source_query:
                result.skipped.append(CuratorAction(action="create_page", detail="missing name/source_query"))
                continue
            title = op.name.strip().lower()
            if title in existing_titles:
                result.skipped.append(CuratorAction(action="create_page", target=op.name, detail="duplicate title"))
                continue
            existing_titles.add(title)
            if not dry_run and not await _new_page(op.name, op.source_query):
                # DB uniqueness guard rejected it (a concurrent run won the race).
                result.skipped.append(CuratorAction(action="create_page", target=op.name, detail="already exists"))
                continue
            result.applied.append(CuratorAction(action="create_page", target=op.name, detail=op.source_query))

        elif op.action == "delete_page":
            if op.page_id not in managed_page_ids:
                result.skipped.append(
                    CuratorAction(action="delete_page", target=op.page_id, detail="not a managed page")
                )
                continue
            if not dry_run:
                await engine.delete_knowledge_node(bank_id, op.page_id, request_context=request_context)
            managed_page_ids.discard(op.page_id)
            result.applied.append(CuratorAction(action="delete_page", target=op.page_id, detail=op.reason))

        elif op.action == "merge_pages":
            mergeable = [pid for pid in op.page_ids if pid in managed_page_ids]
            if len(mergeable) < 2 or not op.name or not op.source_query:
                result.skipped.append(
                    CuratorAction(action="merge_pages", detail="need >=2 managed pages + name/source_query")
                )
                continue
            if not dry_run:
                if not await _new_page(op.name, op.source_query):
                    result.skipped.append(CuratorAction(action="merge_pages", target=op.name, detail="already exists"))
                    continue
                for pid in mergeable:
                    await engine.delete_knowledge_node(bank_id, pid, request_context=request_context)
            for pid in mergeable:
                managed_page_ids.discard(pid)
            result.applied.append(CuratorAction(action="merge_pages", target=op.name, detail=",".join(mergeable)))

        elif op.action == "create_subfolder":
            if not op.name or not op.mission:
                result.skipped.append(CuratorAction(action="create_subfolder", detail="missing name/mission"))
                continue
            if depth >= MAX_DEPTH or (child_folder_count + spawned) >= MAX_SUBFOLDERS:
                result.skipped.append(CuratorAction(action="create_subfolder", target=op.name, detail="bounds reached"))
                continue
            if not dry_run:
                await engine.create_knowledge_folder(
                    bank_id,
                    op.name,
                    parent_id=folder_id,
                    mission=op.mission,
                    managed=True,
                    request_context=request_context,
                )
            spawned += 1
            result.applied.append(CuratorAction(action="create_subfolder", target=op.name, detail=op.mission))

    return result


async def _ask_llm_for_plan(
    engine: "MemoryEngine",
    bank_id: str,
    mission: str,
    folder_name: str,
    current_pages: list[dict[str, Any]],
    memories: list[str],
    max_ops: int,
    request_context: "RequestContext",
) -> CuratorPlan:
    """Build the curator prompt and ask the bank's configured LLM for a plan."""
    resolved = await engine._config_resolver.resolve_full_config(bank_id, request_context)
    llm = engine._reflect_llm_config.with_config(resolved, bank_id=bank_id, operation="curate_folder")

    pages_block = (
        "\n".join(f"- id={p['id']} | {p['name']} | source_query={p.get('source_query')}" for p in current_pages)
        or "(none)"
    )
    memories_block = "\n".join(f"- {m}" for m in memories) or "(no new memories)"
    user = (
        f"FOLDER: {folder_name}\n"
        f"MISSION (only content matching this belongs here): {mission}\n\n"
        f"CURRENT PAGES — these already exist, do NOT recreate them:\n{pages_block}\n\n"
        f"NEW MEMORIES — filter to the ones matching the mission, then act:\n{memories_block}\n"
    )
    raw, _usage = await llm.call(
        messages=[
            {"role": "system", "content": CURATOR_SYSTEM_PROMPT.format(max_ops=max_ops)},
            {"role": "user", "content": user},
        ],
        response_format=CuratorPlan,
        scope="curate_folder",
        temperature=0.1,
        max_completion_tokens=2048,
        skip_validation=True,
        return_usage=True,
    )
    return parse_plan(raw)


async def curate_folder(
    engine: "MemoryEngine",
    bank_id: str,
    folder_id: str,
    *,
    request_context: "RequestContext",
    max_ops: int = DEFAULT_MAX_OPS,
    dry_run: bool = False,
) -> CuratorResult:
    """Run the curator for one folder: read new memories → LLM plan → apply.

    The curator looks at the memories created since this folder was last curated
    (the delta), like the mental-model refresh — NOT a semantic recall. If nothing
    new has arrived, it is a no-op.
    """
    nodes = await engine.list_knowledge_nodes(bank_id, request_context=request_context)
    folder = next((n for n in nodes if n["id"] == folder_id and n.get("kind") == "folder"), None)
    if folder is None:
        raise ValueError(f"Folder '{folder_id}' not found")
    mission = folder.get("mission")
    if not mission:
        # No mission → nothing to curate.
        return CuratorResult(folder_id=folder_id, mission=None)

    current_pages = [n for n in nodes if n.get("parent_id") == folder_id and n["kind"] == "page"]
    since = _parse_dt(folder.get("last_curated_at"))
    memories = await engine.list_new_memory_texts(
        bank_id, since=since, limit=CURATOR_MAX_MEMORIES, request_context=request_context
    )
    if not memories:
        # No new memories since the last run → nothing to do.
        return CuratorResult(folder_id=folder_id, mission=mission)

    plan = await _ask_llm_for_plan(
        engine, bank_id, mission, folder["name"], current_pages, memories, max_ops, request_context
    )
    result = await apply_curator_plan(
        engine, bank_id, folder_id, plan, request_context=request_context, max_ops=max_ops, dry_run=dry_run
    )
    if not dry_run:
        await engine.mark_folder_curated(bank_id, folder_id, request_context=request_context)
    return result
