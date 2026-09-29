"""
Fuzzy tag matching.

Tag filtering is exact array containment, which is right for machine-assigned tags and
wrong for tags that hold user-facing names: a caller filtering by whatever a query
mentioned passes ``typsecript`` and the memory tagged ``typescript`` is excluded *before*
ranking is consulted, so the recall comes back empty (issue #4026).

The fix is a resolution pass, not a new matching mode. A ``TagGroupLeaf`` carrying
``resolve="fuzzy"`` has its tags treated as *tokens*: each is matched against the
bank's tag vocabulary here, and the leaf is rewritten into an equivalent tree of ordinary
``resolve="exact"`` leaves. Everything downstream — the SQL builders in ``tags.py``, their
Python mirrors used on the graph path, the ``GIN(tags)`` index, the store protocol and the
Oracle dialect — sees only exact tags and needs no knowledge of any of this.

The rewrite per leaf mode, for tokens ``t1..tn`` expanding to tag sets ``E1..En``:

- ``any`` / ``any_strict``: one leaf over ``E1 ∪ … ∪ En``. OR-of-unions is union-of-ORs,
  so flattening is exact here.
- ``all`` / ``all_strict``: ``AND`` of one ``any``-leaf per token — "has some spelling of
  t1 AND some spelling of t2". Flattening these into a single ``@>`` array would instead
  demand every spelling of every token be present, a filter that matches nothing.
- ``exact``: ``OR`` over the cross product of the expansions, each branch an ``exact``
  leaf. Combinations whose distinct tags number fewer than the tokens are dropped, so a
  memory carrying one tag cannot satisfy two tokens whose expansions overlap.

Matching is trigram similarity at ``MIN_SIMILARITY``, reusing the function entity
resolution already uses, so one notion of "similar name" governs the whole codebase.

**Similarity is length-sensitive, and short tags suffer.** Measured against this rule:
``typescropt``/``typescript`` 0.57, ``mongp``/``mongo`` 0.50, ``moongoose``/``mongoose``
0.73, ``user:alice``/``user:alicia`` 0.64 — but ``kakfa``/``kafka`` is only 0.20 and does
not resolve, because a five-character word has few trigrams and one edit destroys three of
them. Fuzzy matching is therefore effective on descriptive tags and weak on very short
ones. That same threshold is what keeps ``mango``/``mongo`` (0.33) and ``k9s``/``k8s``
(0.14) apart, which a lower one would not.
"""

from __future__ import annotations

import math
from itertools import product

from ..entity_resolver import trigram_similarity
from .tags import TagGroup, TagGroupAnd, TagGroupLeaf, TagGroupNot, TagGroupOr

#: Trigram similarity at or above which a token resolves to a tag. Shares
#: ``entity_resolver.trigram_similarity`` — verified byte-identical to Postgres
#: ``similarity()`` (#3107) — rather than growing a second notion of name similarity.
MIN_SIMILARITY = 0.45

#: Distinct tags a bank may hold before fuzzy matching refuses to run.
#: Resolution enumerates the vocabulary, and matching against a silently truncated one
#: would change which memories match with no signal to the caller, so an oversized bank is
#: rejected (422) rather than quietly given different results.
MAX_VOCABULARY = 5000

#: Ceiling on the branches an ``exact`` leaf may expand into. The branch count is the
#: product of the per-token expansions, so it grows multiplicatively where the other modes
#: grow additively. Exceeding it raises rather than truncating: dropping branches silently
#: would change which memories match, the very failure this feature exists to remove.
MAX_EXACT_COMBINATIONS = 32


class TagResolutionError(ValueError):
    """A fuzzy leaf could not be resolved. Surfaced to callers as a 422."""


def _normalize(value: str) -> str:
    """Casefold for comparison. Tags are matched case-insensitively; the stored spelling
    is what ends up in the rewritten leaf, so nothing here changes what is queried."""
    return value.strip().casefold()


def _split_namespace(tag: str) -> tuple[str, str]:
    """``("name", "k8s")`` for ``name:k8s``; ``("", tag)`` when there is no namespace.

    Similarity must be measured on the value alone. A namespace is shared by every
    candidate in it, so its trigrams count as agreement between tags that have nothing
    else in common — and the shorter the value, the larger that shared share is. Scored
    whole, ``name:the`` and ``name:ts`` reach 0.545 while ``the`` and ``ts`` score 0.167,
    so every short tag in a namespace resolves to every other. Longer namespaces
    (``customer:``, ``component:``) make it worse.
    """
    namespace, separator, value = tag.partition(":")
    return (namespace, value) if separator else ("", tag)


def expand_token(token: str, vocabulary: list[str]) -> list[str]:
    """The tags ``token`` should match, given the bank's tag vocabulary.

    An exact (case-insensitive) hit wins outright — a token that is already a real tag is
    never widened to its neighbours. Otherwise every vocabulary entry scoring at least
    ``MIN_SIMILARITY`` is returned, most similar first.

    A token that matches nothing comes back as itself rather than as an empty list. That
    keeps the leaf unsatisfiable instead of empty, which matters: the SQL builders read
    empty tags as "no filtering", so an empty expansion would drop the filter entirely and
    widen the recall to the whole bank — the opposite of what the caller asked for.
    """
    normalized = _normalize(token)
    exact = [tag for tag in vocabulary if _normalize(tag) == normalized]
    if exact:
        return exact

    token_namespace, token_value = _split_namespace(normalized)
    scored = [
        (tag, trigram_similarity(token_value, tag_value))
        for tag in vocabulary
        # Only within the same namespace: a `name:` token has no business resolving to a
        # `scope:` tag just because the values look alike.
        for tag_namespace, tag_value in [_split_namespace(_normalize(tag))]
        if tag_namespace == token_namespace
    ]
    # Closest tag first; ties keep vocabulary order, which list_tags returns by usage count.
    matches = [tag for tag, score in sorted(scored, key=lambda pair: -pair[1]) if score >= MIN_SIMILARITY]
    return matches or [token]


def _resolve_leaf(leaf: TagGroupLeaf, vocabulary: list[str]) -> TagGroup:
    """Rewrite one fuzzy leaf into exact-only groups."""
    if not leaf.tags:
        # Nothing to resolve. For `exact` this is the untagged/global scope; for every
        # other mode it means "no tag filtering". Both are unchanged by resolution.
        return TagGroupLeaf(tags=list(leaf.tags), match=leaf.match)

    expanded = [expand_token(token, vocabulary) for token in leaf.tags]

    if leaf.match in ("any", "any_strict"):
        # OR across tokens and OR within a token are the same OR: one flat union.
        union: list[str] = []
        for tags in expanded:
            for tag in tags:
                if tag not in union:
                    union.append(tag)
        return TagGroupLeaf(tags=union, match=leaf.match)

    if leaf.match in ("all", "all_strict"):
        # The caller's AND across tokens is preserved; each conjunct becomes an OR over
        # that token's spellings. The lenient `all` keeps its untagged inclusion by using
        # `any` per conjunct: (untagged ∨ E1) ∧ (untagged ∨ E2) ≡ untagged ∨ (E1 ∧ E2).
        inner_match = "any" if leaf.match == "all" else "any_strict"
        conjuncts = [TagGroupLeaf(tags=tags, match=inner_match) for tags in expanded]
        if len(conjuncts) == 1:
            return conjuncts[0]
        return TagGroupAnd.model_validate({"and": list(conjuncts)})

    # exact: set equality, so the memory's whole tag set must be one combination of the
    # expansions — one stored tag per token, nothing extra.
    branch_count = math.prod(len(tags) for tags in expanded)
    if branch_count > MAX_EXACT_COMBINATIONS:
        # Checked before enumerating, not while: most combinations may be degenerate and
        # dropped below, and counting only the survivors would walk the whole product
        # first — which is what the ceiling exists to prevent.
        raise TagResolutionError(
            f"Fuzzy resolution of an 'exact' tag filter expands to {branch_count} "
            f"candidate scopes, above the ceiling of {MAX_EXACT_COMBINATIONS}. Narrow the "
            f"tags, or use resolve='exact'."
        )

    combinations: list[list[str]] = []
    for combo in product(*expanded):
        distinct = list(dict.fromkeys(combo))
        if len(distinct) != len(combo):
            # Two tokens landed on the same tag; this combination would let a memory with
            # one tag satisfy both, which is not what an exact scope of two tokens means.
            continue
        if distinct not in combinations:
            combinations.append(distinct)

    if not combinations:
        # Every combination was degenerate. Nothing can equal this scope; keep the leaf
        # unsatisfiable rather than dropping the filter.
        return TagGroupLeaf(tags=list(leaf.tags), match="exact")
    if len(combinations) == 1:
        return TagGroupLeaf(tags=combinations[0], match="exact")
    return TagGroupOr.model_validate({"or": [TagGroupLeaf(tags=combo, match="exact") for combo in combinations]})


def _resolve_group(group: TagGroup, vocabulary: list[str]) -> TagGroup:
    """Recursively rewrite a group, leaving exact leaves and structure untouched."""
    if isinstance(group, TagGroupLeaf):
        if group.resolve == "exact":
            return group
        return _resolve_leaf(group, vocabulary)
    if isinstance(group, TagGroupAnd):
        return TagGroupAnd.model_validate({"and": [_resolve_group(child, vocabulary) for child in group.filters]})
    if isinstance(group, TagGroupOr):
        return TagGroupOr.model_validate({"or": [_resolve_group(child, vocabulary) for child in group.filters]})
    if isinstance(group, TagGroupNot):
        # Resolution widens what a NOT excludes — the negation of a wider match. That is
        # the consistent reading: the same leaf means the same thing inside or outside a NOT.
        return TagGroupNot.model_validate({"not": _resolve_group(group.filter, vocabulary)})
    return group


def needs_resolution(tag_groups: list[TagGroup] | None) -> bool:
    """Whether any leaf in the tree asks for fuzzy matching.

    Callers gate the vocabulary read on this: resolution costs a tag enumeration, and the
    overwhelming majority of requests are exact and must not pay for it.
    """
    if not tag_groups:
        return False

    def _walk(group: TagGroup) -> bool:
        if isinstance(group, TagGroupLeaf):
            return group.resolve != "exact"
        if isinstance(group, (TagGroupAnd, TagGroupOr)):
            return any(_walk(child) for child in group.filters)
        if isinstance(group, TagGroupNot):
            return _walk(group.filter)
        return False

    return any(_walk(group) for group in tag_groups)


def resolve_tag_groups(tag_groups: list[TagGroup], vocabulary: list[str]) -> list[TagGroup]:
    """Rewrite ``tag_groups`` into an exact-only tree against ``vocabulary``."""
    return [_resolve_group(group, vocabulary) for group in tag_groups]
