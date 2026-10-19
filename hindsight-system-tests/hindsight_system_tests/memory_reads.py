"""The single-memory read must agree with the listing row for the same memory."""

from __future__ import annotations

from hindsight_client import Hindsight
from hindsight_client_api import MemoryUnitListItem

# Fields both reads return under the same name. The detail view names the fact
# type `type` and lists entities instead of joining them; those two are compared
# separately. Renaming them would break existing detail-view callers.
SHARED_FIELDS = (
    "id",
    "text",
    "context",
    "date",
    "document_id",
    "chunk_id",
    "mentioned_at",
    "occurred_start",
    "occurred_end",
    "tags",
    "metadata",
    "state",
    "invalidation_reason",
    "invalidated_at",
    "edited_at",
    "consolidated_at",
    "consolidation_failed_at",
)


async def assert_detail_matches_listing(client: Hindsight, bank_id: str, row: MemoryUnitListItem) -> None:
    """Fetch ``row`` alone and assert it says what its listing row said."""
    # by_alias: the client model renames the wire field `date` to `var_date`.
    listed = row.model_dump(by_alias=True)
    detail = await client.memory.get_memory(bank_id, row.id)

    assert {f: detail[f] for f in SHARED_FIELDS} == {f: listed[f] for f in SHARED_FIELDS}
    assert detail["type"] == row.fact_type
    assert ", ".join(detail["entities"]) == row.entities
