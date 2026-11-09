"""The page tree and page search can be narrowed to the pages a caller may see.

Pages carry tags (``user:kate``) the same way memories do, and the reads that list
them take recall's tag filter: ``tags`` + ``tags_match``, or compound ``tag_groups``.
Before this, both reads returned every page in the bank — a page tagged for one user
showed up, with its snippet, in another user's search — and an extension that can only
allow or block a call whole had no way to trim the list.

What is pinned here is the seam between page tags and the two listing reads: a filter
must remove the page *and* any folder left with nothing in it (a folder's name says as
much as its pages), and search must rank only within the matching pages.

The pages are created in an empty bank, so they hold no synthesized body and no LLM
call is made: the filter is about tags and placement, not content.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class Pages:
    kate: str
    bob: str
    team: str
    shared: str


@pytest.fixture
async def pages(client, bank_id, settled):
    """People/Kate plans [user:kate], People/Bob plans [user:bob], Team plans [team], Shared plans []."""
    people = await client.acreate_page_folder(bank_id, name="People")
    kate = await client.acreate_page(
        bank_id, name="Kate plans", source_query="What are Kate's plans?", parent_id=people.id, tags=["user:kate"]
    )
    bob = await client.acreate_page(
        bank_id, name="Bob plans", source_query="What are Bob's plans?", parent_id=people.id, tags=["user:bob"]
    )
    team = await client.acreate_page(
        bank_id, name="Team plans", source_query="What are the team's plans?", tags=["team"]
    )
    shared = await client.acreate_page(bank_id, name="Shared plans", source_query="What is planned?")
    await settled(bank_id)
    return Pages(kate=kate.page_id, bob=bob.page_id, team=team.page_id, shared=shared.page_id)


def _shape(roots: list) -> dict[str, dict]:
    """The tree as {node name: children shape}, so the assertion covers placement too."""
    return {node.name: _shape(node.children or []) for node in roots}


async def test_unfiltered_reads_return_every_page(client, bank_id, pages):
    tree = await client.aget_page_tree(bank_id)
    assert _shape(tree.roots) == {
        "People": {"Bob plans": {}, "Kate plans": {}},
        "Shared plans": {},
        "Team plans": {},
    }
    found = await client.asearch_pages(bank_id, q="plans", limit=10)
    assert {r.id for r in found.results} == {pages.kate, pages.bob, pages.team, pages.shared}


async def test_a_strict_tag_filter_keeps_only_that_users_pages(client, bank_id, pages):
    """Kate's page — and nothing else about it — is gone from Bob's view."""
    tree = await client.aget_page_tree(bank_id, tags=["user:bob"], tags_match="any_strict")
    assert _shape(tree.roots) == {"People": {"Bob plans": {}}}

    found = await client.asearch_pages(bank_id, q="Kate plans", tags=["user:bob"], tags_match="any_strict")
    assert [r.id for r in found.results] == [pages.bob]


async def test_any_also_returns_untagged_pages_like_recall(client, bank_id, pages):
    """'any' keeps the bank's untagged (global) pages visible, as it does for memories."""
    tree = await client.aget_page_tree(bank_id, tags=["user:bob"], tags_match="any")
    assert _shape(tree.roots) == {"People": {"Bob plans": {}}, "Shared plans": {}}


async def test_a_folder_emptied_by_the_filter_is_hidden(client, bank_id, pages):
    tree = await client.aget_page_tree(bank_id, tags=["team"], tags_match="all_strict")
    assert _shape(tree.roots) == {"Team plans": {}}


async def test_tag_groups_combine_scopes(client, bank_id, pages):
    """A compound filter: Bob's pages OR the team's, through the same GET reads."""
    groups = [{"or": [{"tags": ["user:bob"], "match": "all_strict"}, {"tags": ["team"], "match": "all_strict"}]}]
    tree = await client.aget_page_tree(bank_id, tag_groups=groups)
    assert _shape(tree.roots) == {"People": {"Bob plans": {}}, "Team plans": {}}

    found = await client.asearch_pages(bank_id, q="plans", tag_groups=groups)
    assert {r.id for r in found.results} == {pages.bob, pages.team}
