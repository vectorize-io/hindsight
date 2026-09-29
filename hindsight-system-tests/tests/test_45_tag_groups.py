"""Tag filters compose: and, or, not, and nesting.

The flat `tags` parameter answers "any of these" or "all of these". Real scoping
questions are shaped differently — everything about *this project* that is not a
*draft*, anything tagged for either of two teams — and those need boolean
structure.

Each operator is checked against a corpus where getting it wrong returns a
*different non-empty set* rather than nothing. An `and` implemented as `or`
returns too much; a `not` that fails to negate returns exactly the rows it was
meant to exclude. Both look like working filters until someone reads them.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "What is Alice working on?"

# tags:                      text:
# project, urgent            SHIPPED
# project, draft             DRAFTED
# personal                   PERSONAL
SHIPPED = "Alice shipped the recall API | Involving: Alice"
DRAFTED = "Alice drafted the recall proposal | Involving: Alice"
PERSONAL = "Alice booked a holiday | Involving: Alice"


@pytest.fixture
async def tagged_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts", contains="shipped").returns(
        extracted(fact("Alice shipped the recall API", who="Alice", entities=["Alice"]))
    )
    llm.on_step("extract_facts", contains="drafted").returns(
        extracted(fact("Alice drafted the recall proposal", who="Alice", entities=["Alice"]))
    )
    llm.on_step("extract_facts", contains="holiday").returns(
        extracted(fact("Alice booked a holiday", who="Alice", entities=["Alice"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice shipped the recall API.", tags=["project", "urgent"])
    await client.aretain(bank_id=bank_id, content="Alice drafted the recall proposal.", tags=["project", "draft"])
    await client.aretain(bank_id=bank_id, content="Alice booked a holiday.", tags=["personal"])
    await settled(bank_id)
    return bank_id


async def _recall(client, bank: str, groups: list[dict]) -> list[str]:
    response = await client.arecall(bank_id=bank, query=QUERY, tag_groups=groups)
    return sorted(r.text for r in response.results)


async def test_a_single_leaf_behaves_like_the_flat_parameter(client, tagged_bank):
    """The base case, so the operators below are read against something known."""
    assert await _recall(client, tagged_bank, [{"tags": ["project"], "match": "any_strict"}]) == sorted(
        [SHIPPED, DRAFTED]
    )


async def test_and_requires_both_leaves(client, tagged_bank):
    """Only the fact carrying *both* tags. An `and` behaving like an `or` would
    return the drafted fact as well — more results, still plausible."""
    groups = [{"and": [{"tags": ["project"], "match": "any_strict"}, {"tags": ["urgent"], "match": "any_strict"}]}]

    assert await _recall(client, tagged_bank, groups) == [SHIPPED]


async def test_or_admits_either_leaf(client, tagged_bank):
    """Two unrelated tags, both wanted — the query a flat `tags` list cannot
    express once the leaves have different match modes."""
    groups = [{"or": [{"tags": ["urgent"], "match": "any_strict"}, {"tags": ["personal"], "match": "any_strict"}]}]

    assert await _recall(client, tagged_bank, groups) == sorted([SHIPPED, PERSONAL])


async def test_not_excludes_what_it_names(client, tagged_bank):
    """A `not` that quietly fails to negate returns precisely the rows it was
    meant to hide — the worst possible failure for a filter someone is using to
    keep something out of an answer."""
    groups = [{"not": {"tags": ["project"], "match": "any_strict"}}]

    assert await _recall(client, tagged_bank, groups) == [PERSONAL]


async def test_operators_nest(client, tagged_bank):
    """ "This project, but not the drafts" — the shape most real scoping takes,
    and the reason the operators have to compose rather than merely exist."""
    groups = [
        {
            "and": [
                {"tags": ["project"], "match": "any_strict"},
                {"not": {"tags": ["draft"], "match": "any_strict"}},
            ]
        }
    ]

    assert await _recall(client, tagged_bank, groups) == [SHIPPED]


async def test_a_group_matching_nothing_returns_nothing(client, tagged_bank):
    """An impossible combination is empty, not unfiltered. A group that silently
    degraded to "no filter" would hand back the whole bank."""
    groups = [{"and": [{"tags": ["personal"], "match": "any_strict"}, {"tags": ["urgent"], "match": "any_strict"}]}]

    assert await _recall(client, tagged_bank, groups) == []
