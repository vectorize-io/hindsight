"""
Tags filtering utilities for retrieval.

Provides SQL building functions for filtering memories by tags.
Supports four matching modes via TagsMatch enum:
- "any": OR matching, includes untagged memories (default, backward compatible)
- "all": AND matching, includes untagged memories
- "any_strict": OR matching, excludes untagged memories
- "all_strict": AND matching, excludes untagged memories

OR matching (any/any_strict): Memory matches if ANY of its tags overlap with request tags
AND matching (all/all_strict): Memory matches if ALL request tags are present in its tags
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

TagsMatch = Literal["any", "all", "any_strict", "all_strict"]


def _parse_tags_match(match: TagsMatch) -> tuple[str, bool]:
    """
    Parse TagsMatch into operator and include_untagged flag.

    Returns:
        Tuple of (operator, include_untagged)
        - operator: "&&" for any/any_strict, "@>" for all/all_strict
        - include_untagged: True for any/all, False for any_strict/all_strict
    """
    if match == "any":
        return "&&", True
    elif match == "all":
        return "@>", True
    elif match == "any_strict":
        return "&&", False
    elif match == "all_strict":
        return "@>", False
    else:
        # Default to "any" behavior
        return "&&", True


def build_tags_where_clause(
    tags: list[str] | None,
    param_offset: int = 1,
    table_alias: str = "",
    match: TagsMatch = "any",
) -> tuple[str, list, int]:
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

    Returns:
        Tuple of (sql_clause, params, next_param_offset):
        - sql_clause: SQL WHERE clause string
        - params: List of parameter values to bind
        - next_param_offset: Next available parameter number

    Example:
        >>> clause, params, next_offset = build_tags_where_clause(['user_a'], 3, 'mu.', 'any_strict')
        >>> print(clause)  # "AND mu.tags IS NOT NULL AND mu.tags != '{}' AND mu.tags && $3"
    """
    if not tags:
        return "", [], param_offset

    column = f"{table_alias}tags" if table_alias else "tags"
    operator, include_untagged = _parse_tags_match(match)

    if include_untagged:
        # Include untagged memories (NULL or empty array) OR matching tags
        clause = f"AND ({column} IS NULL OR {column} = '{{}}' OR {column} {operator} ${param_offset})"
    else:
        # Strict: only memories with matching tags (exclude NULL and empty)
        clause = f"AND {column} IS NOT NULL AND {column} != '{{}}' AND {column} {operator} ${param_offset}"

    return clause, [tags], param_offset + 1


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
    if not tags:
        return ""

    column = f"{table_alias}tags" if table_alias else "tags"
    operator, include_untagged = _parse_tags_match(match)

    if include_untagged:
        # Include untagged memories (NULL or empty array) OR matching tags
        return f"AND ({column} IS NULL OR {column} = '{{}}' OR {column} {operator} ${param_num})"
    else:
        # Strict: only memories with matching tags (exclude NULL and empty)
        return f"AND {column} IS NOT NULL AND {column} != '{{}}' AND {column} {operator} ${param_num}"


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
    if not tags:
        return results

    _, include_untagged = _parse_tags_match(match)
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
            if is_any_match:
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


class TagGroupAnd(BaseModel):
    """Compound AND group: all child filters must match."""

    model_config = ConfigDict(populate_by_name=True)
    filters: list[TagGroup] = Field(alias="and")


class TagGroupOr(BaseModel):
    """Compound OR group: at least one child filter must match."""

    model_config = ConfigDict(populate_by_name=True)
    filters: list[TagGroup] = Field(alias="or")


class TagGroupNot(BaseModel):
    """Compound NOT group: child filter must NOT match."""

    model_config = ConfigDict(populate_by_name=True)
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
) -> tuple[str, list, int]:
    """
    Recursively build an inner SQL clause (no leading AND/OR) for a single TagGroup.

    Returns:
        (inner_clause, params, next_param_offset)
    """
    if isinstance(group, TagGroupLeaf):
        column = f"{table_alias}tags" if table_alias else "tags"
        operator, include_untagged = _parse_tags_match(group.match)
        if include_untagged:
            clause = f"({column} IS NULL OR {column} = '{{}}' OR {column} {operator} ${param_offset})"
        else:
            clause = f"({column} IS NOT NULL AND {column} != '{{}}' AND {column} {operator} ${param_offset})"
        return clause, [group.tags], param_offset + 1

    elif isinstance(group, TagGroupAnd):
        parts = []
        params: list = []
        offset = param_offset
        for child in group.filters:
            child_clause, child_params, offset = _build_group_clause(child, offset, table_alias)
            parts.append(child_clause)
            params.extend(child_params)
        inner = " AND ".join(parts)
        return f"({inner})", params, offset

    elif isinstance(group, TagGroupOr):
        parts = []
        params = []
        offset = param_offset
        for child in group.filters:
            child_clause, child_params, offset = _build_group_clause(child, offset, table_alias)
            parts.append(child_clause)
            params.extend(child_params)
        inner = " OR ".join(parts)
        return f"({inner})", params, offset

    elif isinstance(group, TagGroupNot):
        child_clause, child_params, next_offset = _build_group_clause(group.filter, param_offset, table_alias)
        return f"NOT {child_clause}", child_params, next_offset

    else:
        # Should never happen with proper Pydantic validation
        return "", [], param_offset


def build_tag_groups_where_clause(
    tag_groups: list[TagGroup] | None,
    param_offset: int,
    table_alias: str = "",
) -> tuple[str, list, int]:
    """
    Build a SQL WHERE clause for compound tag group filtering.

    Top-level groups are AND-ed together. Each group is a recursive boolean
    expression (leaf, and, or, not).

    Args:
        tag_groups: List of TagGroup objects. If None or empty, returns empty clause.
        param_offset: Starting parameter number for SQL placeholders.
        table_alias: Optional table alias prefix (e.g., "mu." for "memory_units mu").

    Returns:
        Tuple of (sql_clause, params, next_param_offset):
        - sql_clause: SQL WHERE clause string starting with "AND" (or empty string)
        - params: List of parameter values to bind (one per leaf node)
        - next_param_offset: Next available parameter number

    Example:
        >>> groups = [TagGroupLeaf(tags=["user:alice"], match="all_strict")]
        >>> clause, params, next_offset = build_tag_groups_where_clause(groups, 3)
        >>> print(clause)  # "AND (tags IS NOT NULL AND tags != '{}' AND tags @> $3)"
    """
    if not tag_groups:
        return "", [], param_offset

    all_params: list = []
    all_clauses: list[str] = []
    offset = param_offset

    for group in tag_groups:
        inner_clause, group_params, offset = _build_group_clause(group, offset, table_alias)
        all_clauses.append(inner_clause)
        all_params.extend(group_params)

    combined = " AND ".join(all_clauses)
    return f"AND {combined}", all_params, offset


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
        _, include_untagged = _parse_tags_match(group.match)
        is_any_match = group.match in ("any", "any_strict")
        tags_set = set(group.tags)

        if is_untagged:
            return include_untagged
        else:
            result_tags_set = set(result_tags)
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
