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

from typing import Literal

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
