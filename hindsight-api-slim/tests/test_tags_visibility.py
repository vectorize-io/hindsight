"""
Tests for tags-based visibility scoping.

This module tests the tags feature which allows filtering memories by visibility tags.
Use cases:
- Multi-user agent: Agent has a single memory bank, users should only see memories from
  conversations they participated in
- Student tracking: Teacher tracks students, students should only see their own data

The tags use OR-based matching: a memory matches if ANY of its tags overlap with the request tags.
"""
from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.search.tags import (
    TagGroupAnd,
    TagGroupLeaf,
    TagGroupNot,
    TagGroupOr,
    build_tag_groups_where_clause,
    build_tags_where_clause_simple,
    filter_results_by_tag_groups,
    filter_results_by_tags,
)

# ============================================================================
# Unit Tests for tags SQL builder
# ============================================================================


class TestTagsWhereClauseBuilder:
    """Unit tests for the tags WHERE clause SQL builder."""

    def test_no_tags_returns_empty_string(self):
        """When tags is None, should return empty string (no filtering)."""
        result = build_tags_where_clause_simple(None, 5)
        assert result == ""

    def test_empty_tags_list_returns_empty_string(self):
        """When tags is an empty list, should return empty string (no filtering)."""
        result = build_tags_where_clause_simple([], 5)
        assert result == ""

    def test_tags_with_different_param_num(self):
        """Should use the provided parameter number."""
        result = build_tags_where_clause_simple(["user_a", "user_b"], 3)
        # Default is "any" which includes untagged
        assert "$3" in result

    def test_tags_with_table_alias(self):
        """Should include table alias when provided."""
        result = build_tags_where_clause_simple(["user_a"], 5, table_alias="mu.")
        assert "mu.tags" in result

    # ---- Test "any" mode (OR, includes untagged - default) ----

    def test_tags_match_any_includes_untagged(self):
        """When match='any', should include untagged memories (NULL or empty)."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="any")
        # Should use OR with NULL/empty check
        assert "IS NULL" in result
        assert "= '{}'" in result
        assert "&&" in result  # overlap operator

    def test_tags_match_any_uses_overlap(self):
        """When match='any', should use overlap operator (&&)."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="any")
        assert "&&" in result

    # ---- Test "all" mode (AND, includes untagged) ----

    def test_tags_match_all_includes_untagged(self):
        """When match='all', should include untagged memories (NULL or empty)."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="all")
        # Should use OR with NULL/empty check
        assert "IS NULL" in result
        assert "= '{}'" in result
        assert "@>" in result  # contains operator

    def test_tags_match_all_uses_contains(self):
        """When match='all', should use contains operator (@>)."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="all")
        assert "@>" in result

    # ---- Test "any_strict" mode (OR, excludes untagged) ----

    def test_tags_match_any_strict_excludes_untagged(self):
        """When match='any_strict', should exclude untagged memories."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="any_strict")
        # Should require tags to be NOT NULL and not empty
        assert "IS NOT NULL" in result
        assert "!= '{}'" in result
        assert "&&" in result  # overlap operator

    def test_tags_match_any_strict_uses_overlap(self):
        """When match='any_strict', should use overlap operator (&&)."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="any_strict")
        assert "&&" in result
        # Should NOT include untagged
        assert "IS NULL" not in result or "IS NOT NULL" in result

    # ---- Test "all_strict" mode (AND, excludes untagged) ----

    def test_tags_match_all_strict_excludes_untagged(self):
        """When match='all_strict', should exclude untagged memories."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="all_strict")
        # Should require tags to be NOT NULL and not empty
        assert "IS NOT NULL" in result
        assert "!= '{}'" in result
        assert "@>" in result  # contains operator

    def test_tags_match_all_strict_uses_contains(self):
        """When match='all_strict', should use contains operator (@>)."""
        result = build_tags_where_clause_simple(["user_a"], 5, match="all_strict")
        assert "@>" in result

    # ---- Test table alias with all modes ----

    def test_tags_match_any_with_table_alias(self):
        """Should include table alias with any mode."""
        result = build_tags_where_clause_simple(["user_a"], 3, table_alias="mu.", match="any")
        assert "mu.tags" in result

    def test_tags_match_all_strict_with_table_alias(self):
        """Should include table alias with all_strict mode."""
        result = build_tags_where_clause_simple(["user_a", "user_b"], 3, table_alias="mu.", match="all_strict")
        assert "mu.tags" in result
        assert "@>" in result
        assert "IS NOT NULL" in result


# ============================================================================
# Unit Tests for filter_results_by_tags (Python-side filtering)
# ============================================================================


class MockResult:
    """Mock result object for testing filter_results_by_tags."""

    def __init__(self, tags):
        self.tags = tags


class TestFilterResultsByTags:
    """Unit tests for the Python-side tags filter function."""

    def test_no_tags_returns_all(self):
        """When tags is None, should return all results."""
        results = [MockResult(["a"]), MockResult(["b"]), MockResult(None)]
        filtered = filter_results_by_tags(results, None)
        assert len(filtered) == 3

    def test_empty_tags_returns_all(self):
        """When tags is empty list, should return all results."""
        results = [MockResult(["a"]), MockResult(["b"]), MockResult(None)]
        filtered = filter_results_by_tags(results, [])
        assert len(filtered) == 3

    # ---- Test "any" mode (OR, includes untagged) ----

    def test_any_mode_includes_matching_tags(self):
        """'any' mode should include results with matching tags."""
        results = [MockResult(["a"]), MockResult(["b"]), MockResult(["c"])]
        filtered = filter_results_by_tags(results, ["a", "b"], match="any")
        # "a" and "b" match, "c" doesn't match and isn't untagged, so excluded
        assert len(filtered) == 2
        tags_found = [r.tags[0] for r in filtered if r.tags]
        assert "a" in tags_found
        assert "b" in tags_found
        assert "c" not in tags_found

    def test_any_mode_includes_untagged(self):
        """'any' mode should include untagged results."""
        results = [MockResult(["a"]), MockResult(None), MockResult([])]
        filtered = filter_results_by_tags(results, ["a"], match="any")
        assert len(filtered) == 3  # a matches, None is untagged, [] is untagged

    def test_any_mode_includes_partial_overlap(self):
        """'any' mode should include results with ANY overlapping tag."""
        results = [MockResult(["a", "x"]), MockResult(["b", "y"])]
        filtered = filter_results_by_tags(results, ["a"], match="any")
        # ["a", "x"] matches, ["b", "y"] doesn't, but untagged would be included
        tags_found = [r.tags for r in filtered]
        assert ["a", "x"] in tags_found

    # ---- Test "any_strict" mode (OR, excludes untagged) ----

    def test_any_strict_excludes_untagged(self):
        """'any_strict' mode should exclude untagged results."""
        results = [MockResult(["a"]), MockResult(None), MockResult([])]
        filtered = filter_results_by_tags(results, ["a"], match="any_strict")
        assert len(filtered) == 1  # Only ["a"] matches
        assert filtered[0].tags == ["a"]

    def test_any_strict_excludes_non_matching(self):
        """'any_strict' mode should exclude non-matching tagged results."""
        results = [MockResult(["a"]), MockResult(["b"]), MockResult(["c"])]
        filtered = filter_results_by_tags(results, ["a"], match="any_strict")
        assert len(filtered) == 1
        assert filtered[0].tags == ["a"]

    # ---- Test "all" mode (AND, includes untagged) ----

    def test_all_mode_requires_all_tags(self):
        """'all' mode should require ALL requested tags to be present."""
        results = [MockResult(["a", "b"]), MockResult(["a"]), MockResult(["b"])]
        filtered = filter_results_by_tags(results, ["a", "b"], match="all")
        # Only ["a", "b"] has both tags, but untagged would also be included
        tags_found = [r.tags for r in filtered]
        assert ["a", "b"] in tags_found

    def test_all_mode_includes_untagged(self):
        """'all' mode should include untagged results."""
        results = [MockResult(["a", "b"]), MockResult(None), MockResult([])]
        filtered = filter_results_by_tags(results, ["a", "b"], match="all")
        assert len(filtered) == 3  # ["a", "b"] matches, None is untagged, [] is untagged

    # ---- Test "all_strict" mode (AND, excludes untagged) ----

    def test_all_strict_requires_all_tags(self):
        """'all_strict' mode should require ALL requested tags."""
        results = [MockResult(["a", "b"]), MockResult(["a"]), MockResult(["b"])]
        filtered = filter_results_by_tags(results, ["a", "b"], match="all_strict")
        assert len(filtered) == 1
        assert filtered[0].tags == ["a", "b"]

    def test_all_strict_excludes_untagged(self):
        """'all_strict' mode should exclude untagged results."""
        results = [MockResult(["a", "b"]), MockResult(None), MockResult([])]
        filtered = filter_results_by_tags(results, ["a", "b"], match="all_strict")
        assert len(filtered) == 1
        assert filtered[0].tags == ["a", "b"]

    def test_all_strict_allows_superset(self):
        """'all_strict' mode should allow results with MORE tags than requested."""
        results = [MockResult(["a", "b", "c"]), MockResult(["a"])]
        filtered = filter_results_by_tags(results, ["a", "b"], match="all_strict")
        assert len(filtered) == 1
        assert filtered[0].tags == ["a", "b", "c"]  # Has a, b, AND c

    def test_all_strict_superset_observation_matches_incoming_memory_tags(self):
        """
        Consolidation scenario: an incoming memory with tags ['user:bob', 'session:id1']
        uses all_strict matching to find existing observations.

        An observation tagged ['user:bob', 'session:id1', 'place:online'] IS matched
        because it contains all of the incoming memory's tags (superset).
        This is NOT exact matching — an observation with extra tags is still a valid match.
        """
        # Incoming memory tags (e.g. from a new retain call)
        incoming_tags = ["user:bob", "session:id1"]

        # Candidate observations with different tag sets
        exact_match = MockResult(["user:bob", "session:id1"])
        superset_match = MockResult(["session:id1", "user:bob", "place:online"])
        different_user = MockResult(["user:alice", "session:id1"])
        missing_session = MockResult(["user:bob"])

        results = [exact_match, superset_match, different_user, missing_session]
        filtered = filter_results_by_tags(results, incoming_tags, match="all_strict")

        # Both exact_match and superset_match have all incoming tags → both match
        assert len(filtered) == 2
        assert exact_match in filtered
        assert superset_match in filtered
        # different_user and missing_session are excluded because they lack at least one tag
        assert different_user not in filtered
        assert missing_session not in filtered


# ============================================================================
# Unit Tests for build_tag_groups_where_clause (SQL builder)
# ============================================================================


class TestBuildTagGroupsWhereClause:
    """Unit tests for the compound tag group SQL builder."""

    def test_none_returns_empty(self):
        """None tag_groups returns empty clause."""
        clause, params, next_offset = build_tag_groups_where_clause(None, 3)
        assert clause == ""
        assert params == []
        assert next_offset == 3

    def test_empty_list_returns_empty(self):
        """Empty tag_groups list returns empty clause."""
        clause, params, next_offset = build_tag_groups_where_clause([], 3)
        assert clause == ""
        assert params == []
        assert next_offset == 3

    def test_single_leaf_any_strict(self):
        """Single any_strict leaf generates correct SQL."""
        groups = [TagGroupLeaf(tags=["step:5", "step:8"], match="any_strict")]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 3)
        assert clause.startswith("AND ")
        assert "$3" in clause
        assert "IS NOT NULL" in clause
        assert "!= '{}'" in clause
        assert "&&" in clause
        assert params == [["step:5", "step:8"]]
        assert next_offset == 4

    def test_single_leaf_all_strict(self):
        """Single all_strict leaf generates @> operator."""
        groups = [TagGroupLeaf(tags=["user:alice"], match="all_strict")]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 1)
        assert "@>" in clause
        assert "IS NOT NULL" in clause
        assert params == [["user:alice"]]
        assert next_offset == 2

    def test_single_leaf_any_includes_untagged(self):
        """Single any (non-strict) leaf generates NULL-inclusive clause."""
        groups = [TagGroupLeaf(tags=["user:alice"], match="any")]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 1)
        assert "IS NULL" in clause
        assert "= '{}'" in clause
        assert "&&" in clause
        assert params == [["user:alice"]]
        assert next_offset == 2

    def test_and_of_two_leaves(self):
        """AND of two leaves generates AND-joined clause."""
        groups = [
            TagGroupAnd.model_validate(
                {"and": [
                    {"tags": ["step:5"], "match": "any_strict"},
                    {"tags": ["user:ep_42"], "match": "all_strict"},
                ]}
            )
        ]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 3)
        assert "AND" in clause
        assert "$3" in clause
        assert "$4" in clause
        assert len(params) == 2
        assert params[0] == ["step:5"]
        assert params[1] == ["user:ep_42"]
        assert next_offset == 5

    def test_or_of_two_leaves(self):
        """OR of two leaves generates OR-joined clause."""
        groups = [
            TagGroupOr.model_validate(
                {"or": [
                    {"tags": ["step:5"], "match": "any_strict"},
                    {"tags": ["priority:high"], "match": "all_strict"},
                ]}
            )
        ]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 1)
        assert "OR" in clause
        assert "$1" in clause
        assert "$2" in clause
        assert len(params) == 2
        assert next_offset == 3

    def test_not_wraps_with_not(self):
        """NOT group wraps child clause with NOT."""
        groups = [
            TagGroupNot.model_validate(
                {"not": {"tags": ["archived"], "match": "any_strict"}}
            )
        ]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 2)
        assert "NOT" in clause
        assert "$2" in clause
        assert len(params) == 1
        assert next_offset == 3

    def test_nested_and_containing_or(self):
        """AND containing an OR generates correct nested SQL."""
        groups = [
            TagGroupAnd.model_validate(
                {"and": [
                    {"tags": ["user:alice"], "match": "all_strict"},
                    {"or": [
                        {"tags": ["step:5"], "match": "any_strict"},
                        {"tags": ["priority:high"], "match": "all_strict"},
                    ]},
                ]}
            )
        ]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 1)
        assert "AND" in clause
        assert "OR" in clause
        assert len(params) == 3
        assert next_offset == 4

    def test_param_numbering_sequential(self):
        """Params are numbered sequentially starting from param_offset."""
        groups = [
            TagGroupAnd.model_validate(
                {"and": [
                    {"tags": ["a"], "match": "any_strict"},
                    {"tags": ["b"], "match": "any_strict"},
                    {"tags": ["c"], "match": "any_strict"},
                ]}
            )
        ]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 5)
        assert "$5" in clause
        assert "$6" in clause
        assert "$7" in clause
        assert next_offset == 8
        assert len(params) == 3

    def test_table_alias_applied_to_leaves(self):
        """Table alias is prefixed to column name in all leaf clauses."""
        groups = [TagGroupLeaf(tags=["user:alice"], match="any_strict")]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 1, table_alias="mu.")
        assert "mu.tags" in clause

    def test_table_alias_propagates_to_nested(self):
        """Table alias propagates to nested leaves (each leaf uses the alias)."""
        groups = [
            TagGroupAnd.model_validate(
                {"and": [
                    {"tags": ["a"], "match": "any_strict"},
                    {"tags": ["b"], "match": "any_strict"},
                ]}
            )
        ]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 1, table_alias="mu.")
        # Each leaf of type any_strict references mu.tags three times (IS NOT NULL, != '{}', &&)
        # We verify that 'tags' without alias is NOT present, proving the alias is always used
        assert "mu.tags" in clause
        # No bare 'tags' keyword without the alias prefix (other than inside the alias itself)
        import re
        bare_tags = re.findall(r"(?<!\.)tags", clause)
        assert len(bare_tags) == 0, f"Found bare 'tags' references without alias: {bare_tags}"

    def test_multiple_top_level_groups_are_anded(self):
        """Multiple top-level groups are AND-ed together."""
        groups = [
            TagGroupLeaf(tags=["step:5"], match="any_strict"),
            TagGroupLeaf(tags=["user:ep_42"], match="all_strict"),
        ]
        clause, params, next_offset = build_tag_groups_where_clause(groups, 1)
        # Should start with AND and have two param refs joined by AND
        assert clause.startswith("AND ")
        assert " AND " in clause[4:]  # after the leading "AND "
        assert "$1" in clause
        assert "$2" in clause
        assert len(params) == 2
        assert next_offset == 3


# ============================================================================
# Unit Tests for filter_results_by_tag_groups (Python-side)
# ============================================================================


class TestFilterResultsByTagGroups:
    """Unit tests for the Python-side compound tag group filter."""

    def test_none_returns_all(self):
        """None tag_groups returns all results."""
        results = [MockResult(["a"]), MockResult(["b"]), MockResult(None)]
        filtered = filter_results_by_tag_groups(results, None)
        assert len(filtered) == 3

    def test_empty_list_returns_all(self):
        """Empty tag_groups list returns all results."""
        results = [MockResult(["a"]), MockResult(None)]
        filtered = filter_results_by_tag_groups(results, [])
        assert len(filtered) == 2

    def test_single_leaf_any_strict_excludes_untagged(self):
        """Single any_strict leaf excludes untagged results."""
        groups = [TagGroupLeaf(tags=["step:5"], match="any_strict")]
        results = [MockResult(["step:5"]), MockResult(["step:9"]), MockResult(None)]
        filtered = filter_results_by_tag_groups(results, groups)
        assert len(filtered) == 1
        assert filtered[0].tags == ["step:5"]

    def test_single_leaf_all_strict_matches_superset(self):
        """Single all_strict leaf matches results that contain all tags."""
        groups = [TagGroupLeaf(tags=["user:alice", "step:5"], match="all_strict")]
        results = [
            MockResult(["user:alice", "step:5"]),
            MockResult(["user:alice", "step:5", "extra"]),
            MockResult(["user:alice"]),
            MockResult(None),
        ]
        filtered = filter_results_by_tag_groups(results, groups)
        assert len(filtered) == 2

    def test_and_both_conditions_must_match(self):
        """AND group: both leaf conditions must match."""
        groups = [
            TagGroupAnd.model_validate(
                {"and": [
                    {"tags": ["user:alice"], "match": "all_strict"},
                    {"tags": ["step:5"], "match": "any_strict"},
                ]}
            )
        ]
        results = [
            MockResult(["user:alice", "step:5"]),  # matches both
            MockResult(["user:alice"]),              # only matches first
            MockResult(["step:5"]),                  # only matches second
            MockResult(None),
        ]
        filtered = filter_results_by_tag_groups(results, groups)
        assert len(filtered) == 1
        assert filtered[0].tags == ["user:alice", "step:5"]

    def test_or_either_condition_matches(self):
        """OR group: either condition matching is sufficient."""
        groups = [
            TagGroupOr.model_validate(
                {"or": [
                    {"tags": ["step:5"], "match": "any_strict"},
                    {"tags": ["priority:high"], "match": "all_strict"},
                ]}
            )
        ]
        results = [
            MockResult(["step:5"]),
            MockResult(["priority:high"]),
            MockResult(["step:5", "priority:high"]),
            MockResult(["other"]),
            MockResult(None),
        ]
        filtered = filter_results_by_tag_groups(results, groups)
        # step:5, priority:high, and step:5+priority:high all match
        assert len(filtered) == 3

    def test_not_negation(self):
        """NOT group: inverts the child match."""
        groups = [
            TagGroupNot.model_validate(
                {"not": {"tags": ["archived"], "match": "any_strict"}}
            )
        ]
        results = [
            MockResult(["archived"]),
            MockResult(["active"]),
            MockResult(["archived", "active"]),
            MockResult(None),
        ]
        filtered = filter_results_by_tag_groups(results, groups)
        # "archived" and "archived+active" should be excluded
        # "active" and None pass (None is untagged, "any_strict" for "archived" would exclude
        # untagged, so NOT(exclude untagged) = include untagged)
        tags_in_filtered = [r.tags for r in filtered]
        assert ["archived"] not in tags_in_filtered
        assert ["archived", "active"] not in tags_in_filtered

    def test_nested_and_containing_or(self):
        """AND containing OR: nested boolean logic works correctly."""
        groups = [
            TagGroupAnd.model_validate(
                {"and": [
                    {"tags": ["user:alice"], "match": "all_strict"},
                    {"or": [
                        {"tags": ["step:5"], "match": "any_strict"},
                        {"tags": ["priority:high"], "match": "any_strict"},
                    ]},
                ]}
            )
        ]
        results = [
            MockResult(["user:alice", "step:5"]),          # user:alice AND (step:5 OR ...)
            MockResult(["user:alice", "priority:high"]),   # user:alice AND (... OR priority:high)
            MockResult(["user:alice"]),                     # user:alice but neither step nor priority
            MockResult(["step:5"]),                         # step:5 but not user:alice
            MockResult(None),
        ]
        filtered = filter_results_by_tag_groups(results, groups)
        assert len(filtered) == 2

    def test_multiple_top_level_groups_are_anded(self):
        """Multiple top-level tag groups are AND-ed."""
        groups = [
            TagGroupLeaf(tags=["user:alice"], match="all_strict"),
            TagGroupLeaf(tags=["step:5"], match="any_strict"),
        ]
        results = [
            MockResult(["user:alice", "step:5"]),  # both match
            MockResult(["user:alice"]),              # only first
            MockResult(["step:5"]),                  # only second
        ]
        filtered = filter_results_by_tag_groups(results, groups)
        assert len(filtered) == 1
        assert filtered[0].tags == ["user:alice", "step:5"]


# ============================================================================
# Integration Tests for tags in retain/recall/reflect
# ============================================================================


@pytest_asyncio.fixture
async def api_client(memory):
    """Create an async test client for the FastAPI app."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_bank_id():
    """Provide a unique bank ID for this test run."""
    return f"tags_test_{datetime.now().timestamp()}"


@pytest.mark.asyncio
async def test_retain_with_tags(api_client, test_bank_id):
    """Test that memories can be stored with tags."""
    # Store memory with tags
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Alice loves hiking in the mountains.",
                    "tags": ["user_alice"]
                }
            ]
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["items_count"] == 1


@pytest.mark.asyncio
async def test_retain_with_document_tags(api_client, test_bank_id):
    """Test that document-level tags are applied to all items."""
    # Store memories with document-level tags
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "document_tags": ["session_123"],
            "items": [
                {"content": "Bob discussed the quarterly report."},
                {"content": "Charlie mentioned the new product launch."}
            ]
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["items_count"] == 2


@pytest.mark.asyncio
async def test_retain_merges_document_and_item_tags(api_client, test_bank_id):
    """Test that document tags and item tags are merged."""
    # Store memory with both document and item tags
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "document_tags": ["session_abc"],
            "items": [
                {
                    "content": "Dave talked about machine learning.",
                    "tags": ["user_dave"]
                }
            ]
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True


@pytest.mark.asyncio
async def test_recall_without_tags_returns_all_memories(api_client, test_bank_id):
    """Test that recall without tags returns all memories (no filtering)."""
    # Store memories for different users
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Eve works on natural language processing.", "tags": ["user_eve"]},
                {"content": "Frank specializes in computer vision.", "tags": ["user_frank"]},
            ]
        }
    )
    assert response.status_code == 200

    # Recall without tags - should return all
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories/recall",
        json={"query": "Who works on what?", "budget": "low"}
    )
    assert response.status_code == 200
    results = response.json()["results"]

    # Should find both Eve and Frank
    texts = [r["text"] for r in results]
    assert any("Eve" in t for t in texts), "Should find Eve"
    assert any("Frank" in t for t in texts), "Should find Frank"


@pytest.mark.asyncio
async def test_recall_with_tags_filters_memories(api_client, test_bank_id):
    """Test that recall with tags only returns matching memories."""
    # Store memories for different users
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Grace is a data scientist at Google.", "tags": ["user_grace"]},
                {"content": "Henry is a software engineer at Meta.", "tags": ["user_henry"]},
            ]
        }
    )
    assert response.status_code == 200

    # Recall with user_grace tag - should only return Grace's memory
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories/recall",
        json={"query": "Who works at which company?", "budget": "low", "tags": ["user_grace"]}
    )
    assert response.status_code == 200
    results = response.json()["results"]

    # Should find Grace but not Henry
    texts = [r["text"] for r in results]
    assert any("Grace" in t for t in texts), "Should find Grace with user_grace tag"
    # Henry should NOT be found since he has user_henry tag
    assert not any("Henry" in t for t in texts), "Should NOT find Henry (different tag)"


@pytest.mark.asyncio
async def test_recall_with_multiple_tags_uses_or_matching(api_client, test_bank_id):
    """Test that multiple tags use OR matching (any match returns the memory)."""
    # Store memories for different users
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Ivan leads the security team.", "tags": ["user_ivan"]},
                {"content": "Julia manages the design team.", "tags": ["user_julia"]},
                {"content": "Karl oversees the marketing team.", "tags": ["user_karl"]},
            ]
        }
    )
    assert response.status_code == 200

    # Recall with user_ivan OR user_julia - should return both Ivan and Julia, but not Karl
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories/recall",
        json={"query": "Who leads which team?", "budget": "low", "tags": ["user_ivan", "user_julia"]}
    )
    assert response.status_code == 200
    results = response.json()["results"]

    texts = [r["text"] for r in results]
    assert any("Ivan" in t for t in texts), "Should find Ivan (tag matches)"
    assert any("Julia" in t for t in texts), "Should find Julia (tag matches)"
    assert not any("Karl" in t for t in texts), "Should NOT find Karl (tag doesn't match)"


@pytest.mark.asyncio
async def test_recall_returns_memories_with_any_overlapping_tag(api_client, test_bank_id):
    """Test that memories with multiple tags are returned if ANY tag matches."""
    # Store memory with multiple tags
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Lisa and Mike discussed the budget in a group chat.",
                    "tags": ["user_lisa", "user_mike"]  # Memory visible to both
                },
                {"content": "Nancy reviewed the budget alone.", "tags": ["user_nancy"]},
            ]
        }
    )
    assert response.status_code == 200

    # Recall with user_lisa - should return the group chat memory
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories/recall",
        json={"query": "What was discussed about the budget?", "budget": "low", "tags": ["user_lisa"]}
    )
    assert response.status_code == 200
    results = response.json()["results"]

    texts = [r["text"] for r in results]
    assert any("Lisa" in t and "Mike" in t for t in texts), "Should find group chat (Lisa is in tags)"
    assert not any("Nancy" in t for t in texts), "Should NOT find Nancy's memory"


@pytest.mark.asyncio
async def test_reflect_with_tags_filters_memories(api_client, test_bank_id):
    """Test that reflect with tags only uses matching memories for reasoning."""
    # Store different memories for different users
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Oscar's favorite color is blue.", "tags": ["user_oscar"]},
                {"content": "Peter's favorite color is red.", "tags": ["user_peter"]},
            ]
        }
    )
    assert response.status_code == 200

    # Reflect with user_oscar tag - should only use Oscar's memories
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/reflect",
        json={
            "query": "What is the favorite color?",
            "budget": "low",
            "tags": ["user_oscar"],
            "include": {"facts": {}}  # Request facts to verify what was used
        }
    )
    assert response.status_code == 200
    result = response.json()

    # The response should mention Oscar's color (blue), not Peter's (red)
    # Note: We can check based_on facts if they're returned
    if result.get("based_on"):
        based_on = result["based_on"]
        memories = based_on.get("memories", []) if isinstance(based_on, dict) else []
        fact_texts = [f["text"] for f in memories]
        # Should use Oscar's memory (if facts are included)
        if fact_texts:
            assert any("Oscar" in t or "blue" in t for t in fact_texts), "Should use Oscar's memory"


@pytest.mark.asyncio
async def test_recall_with_empty_tags_returns_all(api_client, test_bank_id):
    """Test that empty tags list behaves same as no tags (returns all)."""
    # Store memories
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {"content": "Quinn studies mathematics.", "tags": ["user_quinn"]},
                {"content": "Rachel studies physics.", "tags": ["user_rachel"]},
            ]
        }
    )
    assert response.status_code == 200

    # Recall with empty tags list - should return all
    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories/recall",
        json={"query": "Who studies what?", "budget": "low", "tags": []}
    )
    assert response.status_code == 200
    results = response.json()["results"]

    texts = [r["text"] for r in results]
    assert any("Quinn" in t for t in texts), "Should find Quinn"
    assert any("Rachel" in t for t in texts), "Should find Rachel"


@pytest.mark.asyncio
async def test_multi_user_agent_visibility(api_client):
    """
    Test multi-user agent visibility scoping.

    Scenario:
    - Agent has one memory bank
    - Agent chats with User A (room 1) and User B (room 2) separately
    - Agent also hosts a group chat with both users (room 3)
    - User A should only see memories from rooms 1 and 3
    - User B should only see memories from rooms 2 and 3
    - Agent (no filter) should see all memories
    """
    bank_id = f"multi_user_test_{datetime.now().timestamp()}"

    # Store memories from different chat rooms
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                # Room 1: Agent + User A private chat
                {"content": "User A said they prefer morning meetings.", "tags": ["user_a"]},
                # Room 2: Agent + User B private chat
                {"content": "User B mentioned they like afternoon meetings.", "tags": ["user_b"]},
                # Room 3: Group chat with both users
                {"content": "In the group meeting, they agreed to meet at noon.", "tags": ["user_a", "user_b"]},
            ]
        }
    )
    assert response.status_code == 200

    # User A queries - should see their private chat and group chat
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "What meeting time preferences were discussed?", "budget": "low", "tags": ["user_a"]}
    )
    assert response.status_code == 200
    user_a_results = response.json()["results"]
    user_a_texts = [r["text"] for r in user_a_results]

    assert any("morning" in t for t in user_a_texts), "User A should see their own preference (morning)"
    assert any("noon" in t for t in user_a_texts), "User A should see group chat (noon)"
    assert not any("afternoon" in t for t in user_a_texts), "User A should NOT see User B's private preference"

    # User B queries - should see their private chat and group chat
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "What meeting time preferences were discussed?", "budget": "low", "tags": ["user_b"]}
    )
    assert response.status_code == 200
    user_b_results = response.json()["results"]
    user_b_texts = [r["text"] for r in user_b_results]

    assert any("afternoon" in t for t in user_b_texts), "User B should see their own preference (afternoon)"
    assert any("noon" in t for t in user_b_texts), "User B should see group chat (noon)"
    assert not any("morning" in t for t in user_b_texts), "User B should NOT see User A's private preference"

    # Agent queries (no filter) - should see everything
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "What meeting time preferences were discussed?", "budget": "low"}  # No tags
    )
    assert response.status_code == 200
    agent_results = response.json()["results"]
    agent_texts = [r["text"] for r in agent_results]

    assert any("morning" in t for t in agent_texts), "Agent should see User A's preference"
    assert any("afternoon" in t for t in agent_texts), "Agent should see User B's preference"
    assert any("noon" in t for t in agent_texts), "Agent should see group chat"


@pytest.mark.asyncio
async def test_student_tracking_visibility(api_client):
    """
    Test student tracking visibility scoping.

    Scenario:
    - Teacher bot has one memory bank
    - Teacher records observations for Student A, Student B
    - Student A should only see their own data
    - Teacher (no filter) should see all student data
    """
    bank_id = f"student_test_{datetime.now().timestamp()}"

    # Store memories for different students
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": "Student A showed improvement in algebra today.", "tags": ["student_a"]},
                {"content": "Student B struggled with geometry concepts.", "tags": ["student_b"]},
                {"content": "Student A participated actively in class discussion.", "tags": ["student_a"]},
            ]
        }
    )
    assert response.status_code == 200

    # Student A queries - should only see their own data
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "How am I doing in class?", "budget": "low", "tags": ["student_a"]}
    )
    assert response.status_code == 200
    student_a_results = response.json()["results"]
    student_a_texts = [r["text"] for r in student_a_results]

    assert any("algebra" in t for t in student_a_texts), "Student A should see their algebra progress"
    assert any("participated" in t for t in student_a_texts), "Student A should see their participation"
    assert not any("Student B" in t or "geometry" in t for t in student_a_texts), "Student A should NOT see Student B's data"

    # Teacher queries (no filter) - should see all students
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "Which students need help?", "budget": "low"}  # No tags
    )
    assert response.status_code == 200
    teacher_results = response.json()["results"]
    teacher_texts = [r["text"] for r in teacher_results]

    assert any("Student A" in t for t in teacher_texts), "Teacher should see Student A's data"
    assert any("Student B" in t for t in teacher_texts), "Teacher should see Student B's data"


# ============================================================================
# Tests for list_tags API endpoint
# ============================================================================


@pytest.mark.asyncio
async def test_list_tags_returns_all_tags(api_client):
    """Test that list_tags returns all unique tags with counts.

    Note: list_tags counts all memory units including observations.
    Observations inherit tags from their source facts (for visibility security),
    so counts may be higher than the number of stored memories.
    """
    bank_id = f"list_tags_test_{datetime.now().timestamp()}"

    # Store memories with various tags
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": "Memory 1 for user alice.", "tags": ["user:alice"]},
                {"content": "Memory 2 for user alice.", "tags": ["user:alice"]},
                {"content": "Memory 3 for user bob.", "tags": ["user:bob"]},
                {"content": "Memory 4 in session 123.", "tags": ["session:123"]},
                {"content": "Memory 5 for alice in session 456.", "tags": ["user:alice", "session:456"]},
            ]
        }
    )
    assert response.status_code == 200

    # List all tags
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags")
    assert response.status_code == 200
    result = response.json()

    # Verify structure
    assert "items" in result
    assert "total" in result
    assert "limit" in result
    assert "offset" in result

    # Verify tags exist with at least the expected counts
    # Note: Counts may be higher due to observations inheriting source fact tags
    tags_map = {item["tag"]: item["count"] for item in result["items"]}
    assert "user:alice" in tags_map
    assert tags_map["user:alice"] >= 3  # At least 3 memories have this tag
    assert "user:bob" in tags_map
    assert tags_map["user:bob"] >= 1
    assert "session:123" in tags_map
    assert tags_map["session:123"] >= 1
    assert "session:456" in tags_map
    assert tags_map["session:456"] >= 1

    assert result["total"] >= 4  # At least 4 unique tags


@pytest.mark.asyncio
async def test_list_tags_with_wildcard_prefix(api_client):
    """Test that list_tags filters with prefix wildcard pattern (user:*)."""
    bank_id = f"list_tags_wildcard_test_{datetime.now().timestamp()}"

    # Store memories with various tags
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": "Memory for alice who works at tech.", "tags": ["user:alice"]},
                {"content": "Memory for bob who is an engineer.", "tags": ["user:bob"]},
                {"content": "Memory for charlie the designer.", "tags": ["user:charlie"]},
                {"content": "Session memory about the meeting.", "tags": ["session:abc"]},
                {"content": "Room memory for conference room.", "tags": ["room:123"]},
            ]
        }
    )
    assert response.status_code == 200

    # List tags with 'user:*' wildcard pattern
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags", params={"q": "user:*"})
    assert response.status_code == 200
    result = response.json()

    # Should only return user:* tags
    tags = [item["tag"] for item in result["items"]]
    assert "user:alice" in tags
    assert "user:bob" in tags
    assert "user:charlie" in tags
    assert "session:abc" not in tags
    assert "room:123" not in tags
    assert result["total"] == 3


@pytest.mark.asyncio
async def test_list_tags_with_wildcard_suffix(api_client):
    """Test that list_tags filters with suffix wildcard pattern (*-admin)."""
    bank_id = f"list_tags_suffix_test_{datetime.now().timestamp()}"

    # Store memories with various tags - use meaningful content for reliable fact extraction
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": "John has the role-admin permission and can manage user accounts.", "tags": ["role-admin"]},
                {"content": "Sarah has super-admin access and can modify system settings.", "tags": ["super-admin"]},
                {"content": "Mike is a standard role-user who can only view content.", "tags": ["role-user"]},
                {"content": "Alice is a role-guest visitor with limited read access.", "tags": ["role-guest"]},
            ]
        }
    )
    assert response.status_code == 200

    # List tags with '*-admin' wildcard pattern (suffix match)
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags", params={"q": "*-admin"})
    assert response.status_code == 200
    result = response.json()

    # Should only return *-admin tags
    tags = [item["tag"] for item in result["items"]]
    assert "role-admin" in tags
    assert "super-admin" in tags
    assert "role-user" not in tags
    assert "role-guest" not in tags
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_list_tags_with_wildcard_middle(api_client):
    """Test that list_tags filters with middle wildcard pattern (env*-prod)."""
    bank_id = f"list_tags_middle_test_{datetime.now().timestamp()}"

    # Store memories with various tags - use meaningful content for fact extraction
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": "The production environment is configured with high availability and uses AWS infrastructure.", "tags": ["env-prod"]},
                {"content": "The enterprise environment for production runs on dedicated servers with 24/7 monitoring.", "tags": ["environment-prod"]},
                {"content": "The staging environment mirrors production but uses smaller instance sizes.", "tags": ["env-staging"]},
                {"content": "The development environment allows developers to test their code locally.", "tags": ["env-dev"]},
            ]
        }
    )
    assert response.status_code == 200

    # List tags with 'env*-prod' wildcard pattern (middle match)
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags", params={"q": "env*-prod"})
    assert response.status_code == 200
    result = response.json()

    # Should only return env*-prod tags
    tags = [item["tag"] for item in result["items"]]
    assert "env-prod" in tags
    assert "environment-prod" in tags
    assert "env-staging" not in tags
    assert "env-dev" not in tags
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_list_tags_case_insensitive(api_client):
    """Test that list_tags wildcard matching is case-insensitive."""
    bank_id = f"list_tags_case_test_{datetime.now().timestamp()}"

    # Store memories with mixed case tags - use meaningful content
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": "Alice is a software engineer who specializes in machine learning algorithms.", "tags": ["User:Alice"]},
                {"content": "Bob works as a data scientist at a large technology company.", "tags": ["user:bob"]},
                {"content": "Charlie is the lead designer responsible for the user interface.", "tags": ["USER:CHARLIE"]},
            ]
        }
    )
    assert response.status_code == 200

    # List tags with lowercase pattern - should match all cases
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags", params={"q": "user:*"})
    assert response.status_code == 200
    result = response.json()

    # Should match all user tags regardless of case
    tags = [item["tag"] for item in result["items"]]
    assert len(tags) == 3
    assert result["total"] == 3


@pytest.mark.asyncio
async def test_list_tags_pagination(api_client):
    """Test that list_tags supports pagination."""
    bank_id = f"list_tags_pagination_test_{datetime.now().timestamp()}"

    # Store memories with many tags - use meaningful content for fact extraction
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivan", "Julia"]
    items = [
        {"content": f"{name} works as a software engineer at company {i}.", "tags": [f"tag:{i:03d}"]}
        for i, name in enumerate(names)
    ]
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": items}
    )
    assert response.status_code == 200

    # Get first page (limit 3)
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags", params={"limit": 3, "offset": 0})
    assert response.status_code == 200
    result = response.json()
    assert len(result["items"]) == 3
    assert result["total"] == 10
    assert result["limit"] == 3
    assert result["offset"] == 0

    # Get second page
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags", params={"limit": 3, "offset": 3})
    assert response.status_code == 200
    result = response.json()
    assert len(result["items"]) == 3
    assert result["offset"] == 3


@pytest.mark.asyncio
async def test_list_tags_empty_bank(api_client):
    """Test that list_tags returns empty for bank with no tags."""
    bank_id = f"list_tags_empty_test_{datetime.now().timestamp()}"

    # List tags without storing anything
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags")
    assert response.status_code == 200
    result = response.json()

    assert result["items"] == []
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_list_tags_ordered_by_count(api_client):
    """Test that list_tags returns tags ordered by frequency (most used first)."""
    bank_id = f"list_tags_order_test_{datetime.now().timestamp()}"

    # Store memories with tags having different frequencies - use meaningful content
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": "Alice works at a startup company as a developer.", "tags": ["rare"]},
                {"content": "Bob is a senior engineer at Google.", "tags": ["common"]},
                {"content": "Charlie manages the marketing team at Microsoft.", "tags": ["common"]},
                {"content": "Diana leads the design department at Apple.", "tags": ["common"]},
                {"content": "Eve is a data scientist at Amazon.", "tags": ["medium"]},
                {"content": "Frank handles customer support at Meta.", "tags": ["medium"]},
            ]
        }
    )
    assert response.status_code == 200

    # List tags - should be ordered by count descending
    response = await api_client.get(f"/v1/default/banks/{bank_id}/tags")
    assert response.status_code == 200
    result = response.json()

    tags = [item["tag"] for item in result["items"]]
    # common (3) should come before medium (2) which should come before rare (1)
    assert tags.index("common") < tags.index("medium")
    assert tags.index("medium") < tags.index("rare")


@pytest.mark.asyncio
async def test_list_memories_includes_tags(api_client, test_bank_id):
    """Test that list memories endpoint returns tags for each memory unit.

    Regression test: tags were previously omitted from the SELECT query in
    list_memory_units, causing the memory dialog in the UI to show no tags
    even when memories had been stored with tags.
    """
    tags = ["user_alice", "session_xyz", "project_alpha", "team_eng", "env_prod", "region_us"]

    response = await api_client.post(
        f"/v1/default/banks/{test_bank_id}/memories",
        json={
            "items": [
                {
                    "content": "Alice is a senior engineer on the platform team.",
                    "tags": tags,
                }
            ]
        },
    )
    assert response.status_code == 200

    # List memories and verify all tags are returned
    response = await api_client.get(f"/v1/default/banks/{test_bank_id}/memories/list")
    assert response.status_code == 200
    result = response.json()

    assert result["total"] > 0
    memory_item = next((item for item in result["items"] if "Alice" in item["text"]), None)
    assert memory_item is not None, "Should find the stored memory"
    assert "tags" in memory_item, "Memory item must include a 'tags' field"
    assert set(memory_item["tags"]) == set(tags), (
        f"All {len(tags)} tags should be returned, got: {memory_item['tags']}"
    )
