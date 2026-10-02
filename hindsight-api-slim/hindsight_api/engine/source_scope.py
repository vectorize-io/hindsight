"""Which source documents a tag-scoped reader may see (#5030).

A fact can reach a reader through its own tags — a label-derived tag such as
``kind:rule`` shares it — while the document it was extracted from carries only the
writer's tags. Source text (a chunk, or a document's full text) therefore follows its
DOCUMENT's tags, not the fact's: returning it because the fact matched hands the reader
the rest of a document their filter excludes.

Every read that returns source text for memories found by tag goes through
:func:`visible_document_ids`, so the rule lives in one place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .search.tags import (
    TagGroup,
    TagsMatch,
    filter_results_by_tag_groups,
    filter_results_by_tags,
)

# Re-exported under the scope-side name callers already use; the predicate itself
# lives in search/tags.py so "is this filter on" has one definition.
from .search.tags import tag_filter_active as tag_filter_is_active

if TYPE_CHECKING:
    from .db.base import DatabaseConnection


@dataclass(frozen=True)
class _Tagged:
    """A row reduced to what the Python tag matchers read: its ``tags``."""

    id: str
    tags: list[str]


def ids_passing(
    tags_by_id: Mapping[str, list[str] | None],
    *,
    tags: list[str] | None,
    tags_match: TagsMatch,
    tag_groups: list[TagGroup] | None,
) -> set[str]:
    """The ids in ``tags_by_id`` whose tags pass the filter, with the same semantics the
    fact filters use (tags and tag_groups are AND-ed, as in recall)."""
    rows = [_Tagged(i, list(t or [])) for i, t in tags_by_id.items()]
    rows = filter_results_by_tags(rows, tags, tags_match)
    rows = filter_results_by_tag_groups(rows, tag_groups)
    return {r.id for r in rows}


async def visible_document_ids(
    conn: DatabaseConnection | None,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_ids: Iterable[str | None],
    *,
    tags: list[str] | None,
    tags_match: TagsMatch,
    tag_groups: list[TagGroup] | None,
) -> set[str]:
    """The subset of ``document_ids`` the reader's tag filter allows.

    Fails closed: a document that cannot be found (deleted mid-read, or in another bank)
    is not visible, and neither is a ``None`` id (a chunk with no document). With no
    active filter every id is returned without a read. ``conn``
    may be ``None`` for a store-owned bank, whose store answers without SQL.
    """
    ids: set[str] = {d for d in document_ids if d}
    if not ids or not tag_filter_is_active(tags, tags_match, tag_groups):
        return ids

    from .memories import get_memories

    doc_tags = await get_memories().documents_tags(
        conn=conn, fq_table=fq_table, bank_id=bank_id, document_ids=sorted(ids)
    )
    return ids_passing(doc_tags, tags=tags, tags_match=tags_match, tag_groups=tag_groups)
