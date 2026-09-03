"""
Tags filtering utilities for retrieval.

Provides SQL building functions for filtering memories by tags.
Supports five matching modes via TagsMatch enum:
- "any": OR matching, includes untagged memories (default, backward compatible)
- "all": AND matching, includes untagged memories
- "any_strict": OR matching, excludes untagged memories
- "all_strict": AND matching, excludes untagged memories
- "exact": set-equality matching, excludes untagged memories

OR matching (any/any_strict): Memory matches if ANY of its tags overlap with request tags
AND matching (all/all_strict): Memory matches if ALL request tags are present in its tags
EXACT matching: Memory matches only if its tag set EQUALS the request tag set (order-
    independent). Used for observation "scope" filtering, where each observation lives
    under exactly one scope (its full tag set) and "scope [a]" must not match "[a, b]".
    An EMPTY request scope (no tags — ``[]`` or ``None``) is the global/untagged scope and
    matches only untagged memories — the scope that ``observation_scopes="shared"``
    consolidation writes to. This is the one mode where absent tags filter rather than
    meaning "no filter"; all other modes treat empty/absent tags as "no filtering". This
    mirrors the ``GET .../graph`` endpoint, where ``tags_match="exact"`` with no tags also
    selects the global scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

TagsMatch = Literal["any", "all", "any_strict", "all_strict", "exact"]

#: How a tag group leaf's tags are matched. Orthogonal to ``TagsMatch``: it says what the
#: tags *are*, not how they are combined, so it composes with every mode instead of
#: multiplying the enum. See ``tag_resolution`` for the rule.
TagResolution = Literal["exact", "fuzzy"]


@dataclass(frozen=True)
class TagMatchSemantics:
    """How one ``TagsMatch`` mode compares a tags column against a value."""

    operator: str
    """``&&`` (overlap) for any/any_strict, ``@>`` (contains) for all/all_strict."""

    include_untagged: bool
    """True for any/all — untagged rows are visible; False for the ``_strict`` modes."""


@dataclass(frozen=True)
class TagClause:
    """A tag filter rendered to SQL, with the bind bookkeeping it implies.

    The three parts are one unit: ``sql`` embeds ``$n`` placeholders that only
    line up if ``params`` is appended to the caller's list in this order *and*
    ``next_param_offset`` is threaded into whatever builds the next clause.
    Returned as a bare 3-tuple, that protocol was invisible — a caller that
    dropped the offset (several use ``_``) or bound the params out of order got
    no error, just a query silently reading the wrong placeholder.
    """

    sql: str
    """The clause, starting with ``AND`` for the top-level builders, or empty for no filter."""

    params: list = field(default_factory=list)
    """Values to bind, in placeholder order. Empty when the clause needs no binds."""

    next_param_offset: int = 1
    """First unused placeholder number. Unchanged when nothing was bound."""


def _parse_tags_match(match: TagsMatch) -> TagMatchSemantics:
    """Parse TagsMatch into the operator and untagged-visibility it implies."""
    if match == "any":
        return TagMatchSemantics("&&", include_untagged=True)
    elif match == "all":
        return TagMatchSemantics("@>", include_untagged=True)
    elif match == "any_strict":
        return TagMatchSemantics("&&", include_untagged=False)
    elif match == "all_strict":
        return TagMatchSemantics("@>", include_untagged=False)
    elif match == "exact":
        # Set equality is handled by the callers via `@> AND <@`; the operator
        # here is unused. Untagged rows never equal a non-empty scope.
        return TagMatchSemantics("@>", include_untagged=False)
    else:
        # Default to "any" behavior
        return TagMatchSemantics("&&", include_untagged=True)


def build_tags_where_clause(
    tags: list[str] | None,
    param_offset: int = 1,
    table_alias: str = "",
    match: TagsMatch = "any",
    value_expr: str | None = None,
) -> TagClause:
    """
    Build a SQL WHERE clause for filtering by tags.

    Supports four matching modes:
    - "any" (default): OR matching, includes untagged memories
    - "all": AND matching, includes untagged memories
    - "any_strict": OR matching, excludes untagged memories
    - "all_strict": AND matching, excludes untagged memories

    Args:
        tags: List of tags to filter by. If None or empty, returns empty clause (no filtering).
        param_offset: Starting parameter number for SQL placeholders (default 1).
        table_alias: Optional table alias prefix (e.g., "mu." for "memory_units mu").
        match: Matching mode. Defaults to "any".
        value_expr: SQL expression to match the tags column against, in place of a
            bind parameter — e.g. ``"s.tags"`` when the clause is applied per row of
            a joined scope set (the batched staleness check). When given, ``tags``
            only decides *which* clause is built, nothing is bound, and
            ``param_offset`` comes back unchanged. It lands in the SQL verbatim, so
            it must be a literal the caller controls, never user input.

    Returns:
        A TagClause carrying the SQL, its bind values, and the next free placeholder.

    Example:
        >>> built = build_tags_where_clause(['user_a'], 3, 'mu.', 'any_strict')
        >>> print(built.sql)  # "AND mu.tags IS NOT NULL AND mu.tags != '{}' AND mu.tags && $3"
    """
    column = f"{table_alias}tags" if table_alias else "tags"
    # Every branch below matches the column against one right-hand side. Naming it
    # once is what lets a bound parameter and a joined column share these match
    # semantics instead of growing a second implementation of them.
    value = value_expr if value_expr is not None else f"${param_offset}"
    # A caller supplying the expression binds nothing, so the parameter cursor must
    # not advance and no value is appended.
    next_offset = param_offset if value_expr is not None else param_offset + 1
    bound: list = [] if value_expr is not None else [tags]

    if match == "exact" and not tags:
        # Empty/absent scope = global/untagged: match only untagged rows. No bind param
        # needed (callers gate the param on truthy `tags`, so none is appended).
        return TagClause(f"AND ({column} IS NULL OR {column} = '{{}}')", [], param_offset)

    if not tags:
        return TagClause("", [], param_offset)

    if match == "exact":
        # Set equality (order-independent): superset AND subset. Untagged rows
        # (empty array) never satisfy `@>` of a non-empty scope, so they're excluded.
        clause = f"AND ({column} @> {value} AND {column} <@ {value})"
        return TagClause(clause, bound, next_offset)

    semantics = _parse_tags_match(match)

    if semantics.include_untagged:
        # Include untagged memories (NULL or empty array) OR matching tags
        clause = f"AND ({column} IS NULL OR {column} = '{{}}' OR {column} {semantics.operator} {value})"
    else:
        # Strict: only memories with matching tags (exclude NULL and empty)
        clause = f"AND {column} IS NOT NULL AND {column} != '{{}}' AND {column} {semantics.operator} {value}"

    return TagClause(clause, bound, next_offset)


def build_tags_where_clause_simple(
    tags: list[str] | None,
    param_num: int,
    table_alias: str = "",
    match: TagsMatch = "any",
) -> str:
    """
    Build a simple SQL WHERE clause for tags filtering.

    This is a convenience version that returns just the clause string,
    assuming the caller will add the tags array to their params list.

    Args:
        tags: List of tags to filter by. If None or empty, returns empty string.
        param_num: Parameter number to use in the clause.
        table_alias: Optional table alias prefix.
        match: Matching mode. Defaults to "any".

    Returns:
        SQL clause string or empty string.
    """
    column = f"{table_alias}tags" if table_alias else "tags"

    if match == "exact" and not tags:
        # Empty/absent scope = global/untagged: match only untagged rows. No bind param
        # needed (callers gate the param on truthy `tags`, so none is appended).
        return f"AND ({column} IS NULL OR {column} = '{{}}')"

    if not tags:
        return ""

    if match == "exact":
        # Set equality (order-independent): superset AND subset. Untagged rows
        # (empty array) never satisfy `@>` of a non-empty scope, so they're excluded.
        return f"AND ({column} @> ${param_num} AND {column} <@ ${param_num})"

    semantics = _parse_tags_match(match)

    if semantics.include_untagged:
        # Include untagged memories (NULL or empty array) OR matching tags
        return f"AND ({column} IS NULL OR {column} = '{{}}' OR {column} {semantics.operator} ${param_num})"
    else:
        # Strict: only memories with matching tags (exclude NULL and empty)
        return f"AND {column} IS NOT NULL AND {column} != '{{}}' AND {column} {semantics.operator} ${param_num}"


def filter_results_by_tags(
    results: list,
    tags: list[str] | None,
    match: TagsMatch = "any",
) -> list:
    """
    Filter retrieval results by tags in Python (for post-processing).

    Used when SQL filtering isn't possible (e.g., graph traversal results).

    Args:
        results: List of RetrievalResult objects with a 'tags' attribute.
        tags: List of tags to filter by. If None or empty, returns all results.
        match: Matching mode. Defaults to "any".

    Returns:
        Filtered list of results.
    """
    if match == "exact" and not tags:
        # Empty/absent scope = global/untagged: keep only untagged results.
        return [r for r in results if not getattr(r, "tags", None)]

    if not tags:
        return results

    include_untagged = _parse_tags_match(match).include_untagged
    is_any_match = match in ("any", "any_strict")

    tags_set = set(tags)
    filtered = []

    for result in results:
        result_tags = getattr(result, "tags", None)

        # Check if untagged
        is_untagged = result_tags is None or len(result_tags) == 0

        if is_untagged:
            if include_untagged:
                filtered.append(result)
            # else: skip untagged
        else:
            result_tags_set = set(result_tags)
            if match == "exact":
                # Set equality: tag set must match the scope exactly
                if result_tags_set == tags_set:
                    filtered.append(result)
            elif is_any_match:
                # Any overlap
                if result_tags_set & tags_set:
                    filtered.append(result)
            else:
                # All tags must be present
                if tags_set <= result_tags_set:
                    filtered.append(result)

    return filtered


# =============================================================================
# Compound tag group models (recursive boolean expressions)
# =============================================================================


class TagGroupLeaf(BaseModel):
    """A leaf tag filter: matches memories by tag list and match mode."""

    tags: list[str]
    match: TagsMatch = "any_strict"
    resolve: TagResolution = "exact"
    """How ``tags`` are matched: literally (``exact``, the default), or by trigram
    similarity against the bank's tags (``fuzzy``), so a filter on ``typsecript`` still
    reaches the memory tagged ``typescript``. See ``tag_resolution`` for the rule and its
    limits; it composes with every ``match`` mode."""


class TagGroupAnd(BaseModel):
    """Compound AND group: all child filters must match."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    filters: list[TagGroup] = Field(alias="and")


class TagGroupOr(BaseModel):
    """Compound OR group: at least one child filter must match."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    filters: list[TagGroup] = Field(alias="or")


class TagGroupNot(BaseModel):
    """Compound NOT group: child filter must NOT match."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    filter: TagGroup = Field(alias="not")


# TagGroup is a discriminated union; Pydantic will try left-to-right.
# TagGroupLeaf is identified by the presence of 'tags'.
# TagGroupAnd / TagGroupOr / TagGroupNot are compound (no 'tags' key).
TagGroup = Annotated[
    TagGroupLeaf | TagGroupAnd | TagGroupOr | TagGroupNot,
    Field(union_mode="left_to_right"),
]

# Rebuild forward-reference models so recursive TagGroup is resolved.
TagGroupAnd.model_rebuild()
TagGroupOr.model_rebuild()
TagGroupNot.model_rebuild()


# =============================================================================
# SQL builder for compound tag groups
# =============================================================================


def _build_group_clause(
    group: TagGroup,
    param_offset: int,
    table_alias: str,
) -> TagClause:
    """Recursively build an inner SQL clause (no leading AND/OR) for a single TagGroup.

    ``TagClause.sql`` here carries no leading ``AND`` — the caller joins the parts.
    """
    if isinstance(group, TagGroupLeaf):
        column = f"{table_alias}tags" if table_alias else "tags"
        if group.match == "exact":
            if len(group.tags) == 0:
                # Empty scope = global/untagged: match only untagged rows (no bind param).
                return TagClause(f"({column} IS NULL OR {column} = '{{}}')", [], param_offset)
            clause = f"({column} @> ${param_offset} AND {column} <@ ${param_offset})"
            return TagClause(clause, [group.tags], param_offset + 1)
        semantics = _parse_tags_match(group.match)
        if semantics.include_untagged:
            clause = f"({column} IS NULL OR {column} = '{{}}' OR {column} {semantics.operator} ${param_offset})"
        else:
            clause = f"({column} IS NOT NULL AND {column} != '{{}}' AND {column} {semantics.operator} ${param_offset})"
        return TagClause(clause, [group.tags], param_offset + 1)

    elif isinstance(group, TagGroupAnd):
        parts = []
        params: list = []
        offset = param_offset
        for child in group.filters:
            built = _build_group_clause(child, offset, table_alias)
            offset = built.next_param_offset
            parts.append(built.sql)
            params.extend(built.params)
        inner = " AND ".join(parts)
        return TagClause(f"({inner})", params, offset)

    elif isinstance(group, TagGroupOr):
        parts = []
        params = []
        offset = param_offset
        for child in group.filters:
            built = _build_group_clause(child, offset, table_alias)
            offset = built.next_param_offset
            parts.append(built.sql)
            params.extend(built.params)
        inner = " OR ".join(parts)
        return TagClause(f"({inner})", params, offset)

    elif isinstance(group, TagGroupNot):
        built = _build_group_clause(group.filter, param_offset, table_alias)
        return TagClause(f"NOT {built.sql}", built.params, built.next_param_offset)

    else:
        # Should never happen with proper Pydantic validation
        return TagClause("", [], param_offset)


def build_tag_groups_where_clause(
    tag_groups: list[TagGroup] | None,
    param_offset: int,
    table_alias: str = "",
) -> TagClause:
    """
    Build a SQL WHERE clause for compound tag group filtering.

    Top-level groups are AND-ed together. Each group is a recursive boolean
    expression (leaf, and, or, not).

    Args:
        tag_groups: List of TagGroup objects. If None or empty, returns empty clause.
        param_offset: Starting parameter number for SQL placeholders.
        table_alias: Optional table alias prefix (e.g., "mu." for "memory_units mu").

    Returns:
        A TagClause whose ``sql`` starts with "AND" (or is empty), carrying one bind
        value per leaf node.

    Example:
        >>> groups = [TagGroupLeaf(tags=["user:alice"], match="all_strict")]
        >>> built = build_tag_groups_where_clause(groups, 3)
        >>> print(built.sql)  # "AND (tags IS NOT NULL AND tags != '{}' AND tags @> $3)"
    """
    if not tag_groups:
        return TagClause("", [], param_offset)

    all_params: list = []
    all_clauses: list[str] = []
    offset = param_offset

    for group in tag_groups:
        built = _build_group_clause(group, offset, table_alias)
        offset = built.next_param_offset
        all_clauses.append(built.sql)
        all_params.extend(built.params)

    combined = " AND ".join(all_clauses)
    return TagClause(f"AND {combined}", all_params, offset)


# =============================================================================
# Python-side filter for compound tag groups (post-retrieval filtering)
# =============================================================================


def _match_group(result: object, group: TagGroup) -> bool:
    """
    Recursively evaluate a TagGroup against a retrieval result.

    Args:
        result: Any object with a 'tags' attribute (list[str] or None).
        group: The TagGroup to evaluate.

    Returns:
        True if the result matches the group, False otherwise.
    """
    if isinstance(group, TagGroupLeaf):
        result_tags = getattr(result, "tags", None)
        is_untagged = result_tags is None or len(result_tags) == 0
        if group.match == "exact" and len(group.tags) == 0:
            # Empty scope = global/untagged: match only untagged results.
            return is_untagged
        include_untagged = _parse_tags_match(group.match).include_untagged
        is_any_match = group.match in ("any", "any_strict")
        tags_set = set(group.tags)

        if is_untagged:
            return include_untagged
        else:
            result_tags_set = set(result_tags)
            if group.match == "exact":
                return result_tags_set == tags_set
            if is_any_match:
                return bool(result_tags_set & tags_set)
            else:
                return tags_set <= result_tags_set

    elif isinstance(group, TagGroupAnd):
        return all(_match_group(result, child) for child in group.filters)

    elif isinstance(group, TagGroupOr):
        return any(_match_group(result, child) for child in group.filters)

    elif isinstance(group, TagGroupNot):
        return not _match_group(result, group.filter)

    else:
        return True


def filter_results_by_tag_groups(
    results: list,
    tag_groups: list[TagGroup] | None,
) -> list:
    """
    Filter retrieval results by compound tag groups in Python (for post-processing).

    Used when SQL filtering isn't possible (e.g., graph traversal results).
    Top-level groups are AND-ed together.

    Args:
        results: List of RetrievalResult objects with a 'tags' attribute.
        tag_groups: List of TagGroup objects. If None or empty, returns all results.

    Returns:
        Filtered list of results where ALL top-level groups match.
    """
    if not tag_groups:
        return results

    return [r for r in results if all(_match_group(r, group) for group in tag_groups)]
