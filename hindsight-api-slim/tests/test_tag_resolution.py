"""
Tests for fuzzy tag resolution (#4026).

Resolution is a pure rewrite of a TagGroup tree into an exact-only tree, so it is fully
deterministic and asserted directly — no LLM, no database. The two things worth pinning
down are which tokens resolve (trigram similarity at MIN_SIMILARITY) and the per-mode
rewrite, which must preserve what each ``match`` mode means.
"""

import pytest

from hindsight_api.engine.search.tag_resolution import (
    MAX_EXACT_COMBINATIONS,
    MIN_SIMILARITY,
    TagResolutionError,
    expand_token,
    needs_resolution,
    resolve_tag_groups,
)
from hindsight_api.engine.search.tags import (
    TagGroupAnd,
    TagGroupLeaf,
    TagGroupNot,
    TagGroupOr,
    build_tag_groups_where_clause,
    filter_results_by_tag_groups,
)

VOCABULARY = ["typescript", "javascript", "kubernetes", "mongodb", "mongoose", "user:alice"]


class _Result:
    """Minimal stand-in for a retrieval result: the Python-side filters only read `tags`."""

    def __init__(self, tags):
        self.tags = tags


# =============================================================================
# Which tokens resolve
# =============================================================================


@pytest.mark.parametrize(
    "token,tag",
    [
        ("typsecript", "typescript"),  # transposition, 0.47
        ("typescropt", "typescript"),  # substitution, 0.57
        ("kubernets", "kubernetes"),  # deletion, 0.62
        ("user:alcie", "user:alice"),  # transposition inside a namespaced tag, 0.47
    ],
)
def test_typos_resolve(token, tag):
    assert expand_token(token, VOCABULARY) == [tag]


@pytest.mark.parametrize(
    "token,tag",
    [
        ("mango", "mongo"),  # a different word, not a typo (0.33)
        ("k9s", "k8s"),  # different word; short strings score far below the threshold (0.14)
        ("kafka", "kafkaesque"),  # a longer word that contains it (0.42)
        ("", "typescript"),
    ],
)
def test_non_typos_do_not_resolve(token, tag):
    assert expand_token(token, [tag]) == [token]


def test_similarity_is_length_sensitive_so_short_tags_barely_tolerate_typos():
    """A documented limitation, pinned so it stays a deliberate choice rather than a
    surprise: one edit in a five-letter tag destroys most of its trigrams (kakfa/kafka
    scores 0.20)."""
    assert expand_token("kakfa", ["kafka"]) == ["kakfa"]
    # The same class of edit in a longer tag resolves fine.
    assert expand_token("kubernets", ["kubernetes"]) == ["kubernetes"]


def test_exact_hit_wins_over_similar_tags():
    """A token that is already a real tag is never widened to its neighbours."""
    assert expand_token("mongodb", VOCABULARY) == ["mongodb"]


def test_exact_hit_is_case_insensitive():
    assert expand_token("TypeScript", VOCABULARY) == ["typescript"]


def test_a_token_can_resolve_to_several_tags_most_similar_first():
    # mongoos scores 0.70 against mongoose and 0.46 against mongodb.
    assert expand_token("mongoos", VOCABULARY) == ["mongoose", "mongodb"]


def test_unmatched_token_stays_unsatisfiable_not_empty():
    """The critical failure mode: an empty expansion would make the SQL builders read the
    leaf as 'no tag filtering' and widen the recall to the whole bank."""
    assert expand_token("nosuchtag", VOCABULARY) == ["nosuchtag"]

    resolved = resolve_tag_groups(
        [TagGroupLeaf(tags=["nosuchtag"], match="any_strict", resolve="fuzzy")],
        VOCABULARY,
    )
    clause, params, _ = build_tag_groups_where_clause(resolved, 1)
    assert clause != ""
    assert params == [["nosuchtag"]]
    assert filter_results_by_tag_groups([_Result(["typescript"])], resolved) == []


def test_threshold_is_a_sane_fraction():
    assert 0 < MIN_SIMILARITY < 1


# =============================================================================
# Per-mode rewrites
# =============================================================================


def test_any_flattens_to_a_union():
    resolved = resolve_tag_groups(
        [TagGroupLeaf(tags=["typsecript", "kubernets"], match="any_strict", resolve="fuzzy")],
        VOCABULARY,
    )
    assert resolved == [TagGroupLeaf(tags=["typescript", "kubernetes"], match="any_strict")]


def test_all_becomes_an_and_of_ors_not_a_flattened_array():
    """Flattening `all` would demand every spelling of every token be present at once — a
    filter that matches nothing. Each token must become its own OR conjunct."""
    resolved = resolve_tag_groups(
        [TagGroupLeaf(tags=["typsecript", "mongoos"], match="all_strict", resolve="fuzzy")],
        VOCABULARY,
    )
    assert resolved == [
        TagGroupAnd(
            filters=[
                TagGroupLeaf(tags=["typescript"], match="any_strict"),
                TagGroupLeaf(tags=["mongoose", "mongodb"], match="any_strict"),
            ]
        )
    ]

    # A memory tagged with one spelling of each token matches; one carrying only one does not.
    assert filter_results_by_tag_groups([_Result(["typescript", "mongoose"])], resolved)
    assert filter_results_by_tag_groups([_Result(["typescript"])], resolved) == []


def test_lenient_all_keeps_including_untagged():
    """`all` includes untagged memories; the per-conjunct rewrite must not lose that."""
    resolved = resolve_tag_groups(
        [TagGroupLeaf(tags=["typsecript", "mongoos"], match="all", resolve="fuzzy")],
        VOCABULARY,
    )
    assert filter_results_by_tag_groups([_Result([])], resolved)
    assert filter_results_by_tag_groups([_Result(None)], resolved)


def test_single_token_all_does_not_wrap_in_a_pointless_and():
    resolved = resolve_tag_groups(
        [TagGroupLeaf(tags=["mongoos"], match="all_strict", resolve="fuzzy")],
        VOCABULARY,
    )
    assert resolved == [TagGroupLeaf(tags=["mongoose", "mongodb"], match="any_strict")]


def test_exact_becomes_an_or_over_the_cross_product():
    resolved = resolve_tag_groups(
        [TagGroupLeaf(tags=["typsecript", "mongoos"], match="exact", resolve="fuzzy")],
        VOCABULARY,
    )
    assert resolved == [
        TagGroupOr(
            filters=[
                TagGroupLeaf(tags=["typescript", "mongoose"], match="exact"),
                TagGroupLeaf(tags=["typescript", "mongodb"], match="exact"),
            ]
        )
    ]

    # Set equality still holds per branch: an extra tag excludes the memory.
    assert filter_results_by_tag_groups([_Result(["typescript", "mongoose"])], resolved)
    assert filter_results_by_tag_groups([_Result(["typescript", "mongoose", "redis"])], resolved) == []


def test_exact_drops_degenerate_combinations():
    """Two tokens resolving onto the same tag must not let a single-tagged memory satisfy
    a two-token exact scope."""
    resolved = resolve_tag_groups(
        [TagGroupLeaf(tags=["typsecript", "typescropt"], match="exact", resolve="fuzzy")],
        VOCABULARY,
    )
    assert filter_results_by_tag_groups([_Result(["typescript"])], resolved) == []


def test_exact_combination_ceiling_raises_rather_than_truncating():
    # These tags differ only in their last character, so each token resolves to all ten and
    # three tokens cross-multiply past the ceiling.
    vocabulary = [f"resolution-tag-{i}" for i in range(10)]
    with pytest.raises(TagResolutionError, match="above the ceiling"):
        resolve_tag_groups(
            [
                TagGroupLeaf(
                    tags=["resolution-tag-x", "resolution-tag-y", "resolution-tag-z"],
                    match="exact",
                    resolve="fuzzy",
                )
            ],
            vocabulary,
        )
    assert MAX_EXACT_COMBINATIONS > 0


def test_empty_exact_scope_is_untouched():
    """`exact` with no tags is the untagged/global scope — there is nothing to resolve."""
    resolved = resolve_tag_groups([TagGroupLeaf(tags=[], match="exact", resolve="fuzzy")], VOCABULARY)
    assert resolved == [TagGroupLeaf(tags=[], match="exact")]
    assert filter_results_by_tag_groups([_Result([])], resolved)


# =============================================================================
# Tree structure
# =============================================================================


def test_nested_groups_resolve_and_keep_their_structure():
    resolved = resolve_tag_groups(
        [
            TagGroupAnd(
                filters=[
                    TagGroupLeaf(tags=["typsecript"], match="any_strict", resolve="fuzzy"),
                    TagGroupNot(filter=TagGroupLeaf(tags=["kubernets"], match="any_strict", resolve="fuzzy")),
                ]
            )
        ],
        VOCABULARY,
    )
    assert resolved == [
        TagGroupAnd(
            filters=[
                TagGroupLeaf(tags=["typescript"], match="any_strict"),
                TagGroupNot(filter=TagGroupLeaf(tags=["kubernetes"], match="any_strict")),
            ]
        )
    ]


def test_exact_leaves_are_left_alone():
    groups = [TagGroupLeaf(tags=["typsecript"], match="any_strict")]
    assert resolve_tag_groups(groups, VOCABULARY) == groups


def test_resolution_output_carries_no_fuzzy_leaves():
    """The whole design rests on this: nothing downstream ever sees an unresolved leaf."""
    resolved = resolve_tag_groups(
        [
            TagGroupOr(
                filters=[
                    TagGroupLeaf(tags=["typsecript"], match="all_strict", resolve="fuzzy"),
                    TagGroupLeaf(tags=["mongoos", "kubernets"], match="exact", resolve="fuzzy"),
                ]
            )
        ],
        VOCABULARY,
    )
    assert not needs_resolution(resolved)


def test_needs_resolution_walks_the_whole_tree():
    assert not needs_resolution(None)
    assert not needs_resolution([])
    assert not needs_resolution([TagGroupLeaf(tags=["a"], match="any_strict")])
    assert needs_resolution([TagGroupNot(filter=TagGroupLeaf(tags=["a"], match="any_strict", resolve="fuzzy"))])
    assert needs_resolution(
        [
            TagGroupAnd(
                filters=[
                    TagGroupLeaf(tags=["a"], match="any_strict"),
                    TagGroupOr(filters=[TagGroupLeaf(tags=["b"], match="any_strict", resolve="fuzzy")]),
                ]
            )
        ]
    )
