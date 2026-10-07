"""A recall query over the token cap is cut to the cap, not rejected.

`HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS` (default 500) used to answer an over-cap
REST recall with HTTP 400. A client cannot reliably stay under a *token* cap by
cutting characters: the coding-agents plugin sends the first 2000 characters of a
prompt, and code packs far more tokens per character than prose, so a pasted
snippet tipped it over and recall failed. The query now keeps its first 500
tokens and runs, so a question asked up front still finds its answer.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUESTION = "Where does Alice live? "
# Bracket-heavy code: ~1800 characters but ~1000 tokens, twice the cap while under the
# plugin's 2000-character cut. Pure punctuation, so the stub's word-overlap search sees
# only the question — the story is about the query surviving, not about ranking code.
CODE = "{[(<>)]};" * 200


async def test_a_code_heavy_query_under_2000_chars_still_recalls(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021.")
    await settled(bank_id)

    query = QUESTION + CODE
    assert len(query) < 2000

    response = await client.arecall(bank_id=bank_id, query=query)

    assert [r.text for r in response.results] == ["Alice moved to Berlin in 2021 | Involving: Alice"]
