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

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

TagsMatch = Literal["any", "all", "any_strict", "all_strict", "exact"]


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
    elif match == "exact":
        # Set equality is handled by the callers via `@> AND <@`; the operator
        # here is unused. Untagged rows never equal a non-empty scope.
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
    column = f"{table_alias}tags" if table_alias else "tags"

    if match == "exact" and not tags:
        # Empty/absent scope = global/untagged: match only untagged rows. No bind param
        # needed (callers gate the param on truthy `tags`, so none is appended).
        return f"AND ({column} IS NULL OR {column} = '{{}}')", [], param_offset

    if not tags:
        return "", [], param_offset

    if match == "exact":
        # Set equality (order-independent): superset AND subset. Untagged rows
        # (empty array) never satisfy `@>` of a non-empty scope, so they're excluded.
        clause = f"AND ({column} @> ${param_offset} AND {column} <@ ${param_offset})"
        return clause, [tags], param_offset + 1

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
    if match == "exact" and not tags:
        # Empty/absent scope = global/untagged: keep only untagged results.
        return [r for r in results if not getattr(r, "tags", None)]

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


class TagGroupEntityLeaf(BaseModel):
    """A leaf ENTITY filter: matches memories by the entities they mention.

    Tags describe which *compartment* a memory lives in; entities describe what it
    is *about*. On a bank whose tag vocabulary is a handful of broad topics, a tag
    scope cannot isolate a subject — measured on a production bank (2026-08-08),
    the best tag scope for one subject reached 25% precision against a 15.5% base
    rate, while the entity association reached 94.2% precision / 86.2% recall on
    the same corpus. This leaf makes that association usable anywhere a
    ``TagGroup`` already is: mental-model refresh scope, the staleness gate, and
    retrieval filtering, through the same recursive grammar.

    ``entities`` are canonical names, matched case-insensitively — the same
    normalisation the entity registry itself enforces via its
    ``(bank_id, LOWER(canonical_name))`` uniqueness.

    Association is inheritance-aware: a memory matches if it links the entity
    directly (``unit_entities``) or through any of its ``source_memory_ids`` — the
    lane observations use, since consolidation-produced observations carry no
    direct postings by design (their entity association is transitive through
    their sources; see ``memories/pg/graph.py:_entity_rows_for_units_sql``).

    ``match="any"``: mentions at least one listed entity. ``match="all"``: mentions
    every listed entity (directly or via sources, per entity).

    Constraints, enforced by ``validate_entity_leaf_placement`` at the API edge:
    an entity leaf may not appear under ``not`` — the two permissive fallbacks
    (the Python-side post-filter and the non-memory_units surfaces that strip
    entity leaves) evaluate an unknown entity constraint as "matches", and a NOT
    over a permissive "matches" silently inverts into "exclude everything".
    """

    entities: list[str] = Field(min_length=1)
    # Distinct OpenAPI title. Without it this enum serialises as
    # title: "Match", the same title TagGroupLeaf's DIFFERENT match enum
    # already uses (any/all/any_strict/all_strict/exact). progenitor/typify
    # keys inline enums BY TITLE, so the second definition fails to conform
    # and Rust client generation dies with TypeError(InvalidValue) at
    # build.rs:199 -- reproduced 2026-08-11. The collision only appears in a
    # REGENERATED spec, which is why CI saw it and a local build of the
    # committed openapi.json did not.
    match: Literal["any", "all"] = Field(default="any", title="EntityMatch")


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
# TagGroupLeaf is identified by the presence of 'tags'; TagGroupEntityLeaf by
# the presence of 'entities'. TagGroupAnd / TagGroupOr / TagGroupNot are
# compound (neither key).
TagGroup = Annotated[
    TagGroupLeaf | TagGroupEntityLeaf | TagGroupAnd | TagGroupOr | TagGroupNot,
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
    if isinstance(group, TagGroupEntityLeaf):
        # Correlated EXISTS against the entity postings. The outer row is referenced
        # by fully-qualified table name — every builder call site that filters
        # memory_units uses an UNALIASED ``FROM {fq_table("memory_units")}``
        # (semantic/BM25 arms, the staleness read, the scoped listing and window
        # reads), and an unaliased table's qualified name is a valid correlation
        # ref in both PG and Oracle. A caller that aliases the table, or filters a
        # table other than memory_units (the mental-models search, the directives
        # listing), must NOT pass entity leaves here — strip them first with
        # ``strip_entity_leaves``, which is the permissive reading those surfaces
        # want (they carry no entity postings to match against).
        #
        # Bank scoping is deliberately absent from the subquery: entity links are
        # per-unit, and the outer query already constrains bank_id, so a
        # same-named entity in another bank has a different id and no link to
        # this row — the join cannot cross banks.
        #
        # Dialect note: ``= ANY($n)`` over an array bind matches the PG-flavoured
        # operators the tag leaves above already emit (``@>``/``&&``); Oracle
        # translation happens at the same layer it does for those.
        from ..schema import fq_table as _fq_table

        mu = _fq_table("memory_units")
        ue = _fq_table("unit_entities")
        ents = _fq_table("entities")
        row_id = f"{table_alias}id" if table_alias else f"{mu}.id"
        row_sources = f"{table_alias}source_memory_ids" if table_alias else f"{mu}.source_memory_ids"
        names = sorted({n.strip().lower() for n in group.entities if n and n.strip()})
        if not names:
            # Whitespace-only names slipping past min_length: no constraint rather
            # than invalid SQL. Mirrors the empty-groups contract (no clause).
            return "TRUE", [], param_offset
        reach = f"(ue.unit_id = {row_id} OR ({row_sources} IS NOT NULL AND ue.unit_id = ANY({row_sources})))"
        body = (
            f"FROM {ue} ue JOIN {ents} e ON e.id = ue.entity_id "
            f"WHERE LOWER(e.canonical_name) = ANY(${param_offset}) AND {reach}"
        )
        if group.match == "all":
            # Every listed entity reachable. COUNT(DISTINCT lowered-name) over the
            # same body; the target arity is a Python int, inlined as a literal.
            clause = f"((SELECT COUNT(DISTINCT LOWER(e.canonical_name)) {body}) = {len(names)})"
        else:
            clause = f"EXISTS (SELECT 1 {body})"
        return clause, [names], param_offset + 1

    if isinstance(group, TagGroupLeaf):
        column = f"{table_alias}tags" if table_alias else "tags"
        if group.match == "exact":
            if len(group.tags) == 0:
                # Empty scope = global/untagged: match only untagged rows (no bind param).
                return f"({column} IS NULL OR {column} = '{{}}')", [], param_offset
            clause = f"({column} @> ${param_offset} AND {column} <@ ${param_offset})"
            return clause, [group.tags], param_offset + 1
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
    if isinstance(group, TagGroupEntityLeaf):
        # A retrieval result may carry its entities (recall renders them as
        # names or {entity_id, canonical_name} dicts); evaluate against those
        # when present. A result that carries NO entity information passes
        # permissively — this post-filter refines rows the SQL side already
        # scoped, and graph-expansion neighbours legitimately lack entity
        # annotations. Safe only because validate_entity_leaf_placement bars
        # entity leaves under NOT (a NOT over a permissive pass would invert
        # into "drop everything").
        raw = getattr(result, "entities", None)
        if not raw:
            return True
        names: set[str] = set()
        for item in raw:
            if isinstance(item, str):
                names.add(item.lower())
            elif isinstance(item, dict):
                name = item.get("canonical_name") or item.get("name") or item.get("text")
                if isinstance(name, str):
                    names.add(name.lower())
            else:
                name = getattr(item, "canonical_name", None) or getattr(item, "name", None)
                if isinstance(name, str):
                    names.add(name.lower())
        if not names:
            return True
        wanted = {n.strip().lower() for n in group.entities if n and n.strip()}
        if not wanted:
            return True
        if group.match == "all":
            return wanted <= names
        return bool(wanted & names)

    if isinstance(group, TagGroupLeaf):
        result_tags = getattr(result, "tags", None)
        is_untagged = result_tags is None or len(result_tags) == 0
        if group.match == "exact" and len(group.tags) == 0:
            # Empty scope = global/untagged: match only untagged results.
            return is_untagged
        _, include_untagged = _parse_tags_match(group.match)
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


def validate_entity_leaf_placement(tag_groups: list[TagGroup] | None) -> None:
    """Reject entity leaves under ``not`` — raise ValueError naming the constraint.

    SQL evaluates NOT-over-entity correctly, but two surfaces cannot: the
    Python-side post-filter passes entity leaves permissively when a result
    carries no entity annotations, and the non-memory_units surfaces (the
    mental-models search, the directives listing) strip entity leaves outright.
    Under a NOT, a permissive "matches" inverts into "exclude everything" on
    exactly those surfaces, silently. Barring the placement at the API edge
    keeps every fallback sound; lift the bar only by making every consumer
    entity-aware first.
    """

    def _walk(node: TagGroup, under_not: bool) -> None:
        if isinstance(node, TagGroupEntityLeaf):
            if under_not:
                raise ValueError(
                    "an entity filter may not appear under 'not': the post-retrieval "
                    "and non-memory surfaces evaluate entity leaves permissively, and "
                    "negating a permissive match silently excludes everything there"
                )
        elif isinstance(node, (TagGroupAnd, TagGroupOr)):
            for child in node.filters:
                _walk(child, under_not)
        elif isinstance(node, TagGroupNot):
            _walk(node.filter, True)

    for group in tag_groups or []:
        _walk(group, False)


def strip_entity_leaves(tag_groups: list[TagGroup] | None) -> list[TagGroup] | None:
    """Drop entity leaves from a group tree, reading each as "matches".

    For scope filters applied to tables that carry no entity postings (the
    mental-models search in ``reflect/tools.py``, the directives listing): an
    entity constraint is unanswerable there, and the correct reading is the
    permissive one — the surviving tag constraints still apply. Dropping a
    permissive leaf from an AND keeps the siblings; from an OR it makes the
    whole OR permissive (True OR x), so the OR collapses away; a NOT over an
    entity leaf cannot reach here (barred by ``validate_entity_leaf_placement``),
    but is dropped defensively rather than inverted. Returns None when nothing
    constraining survives.
    """

    def _strip(node: TagGroup) -> TagGroup | None:
        if isinstance(node, TagGroupEntityLeaf):
            return None
        if isinstance(node, TagGroupAnd):
            kept = [c for c in (_strip(child) for child in node.filters) if c is not None]
            if not kept:
                return None
            return kept[0] if len(kept) == 1 else TagGroupAnd.model_validate({"and": [_dump(k) for k in kept]})
        if isinstance(node, TagGroupOr):
            stripped = [_strip(child) for child in node.filters]
            if any(s is None for s in stripped):
                return None  # a permissive disjunct makes the whole OR permissive
            kept = [s for s in stripped if s is not None]
            return kept[0] if len(kept) == 1 else TagGroupOr.model_validate({"or": [_dump(k) for k in kept]})
        if isinstance(node, TagGroupNot):
            inner = _strip(node.filter)
            if inner is None:
                return None  # defensive: drop, never invert, an unanswerable constraint
            return TagGroupNot.model_validate({"not": _dump(inner)})
        return node

    def _dump(node: TagGroup) -> dict:
        return node.model_dump(by_alias=True, exclude_none=True)

    if not tag_groups:
        return tag_groups
    kept_groups = [g for g in (_strip(group) for group in tag_groups) if g is not None]
    return kept_groups or None


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
