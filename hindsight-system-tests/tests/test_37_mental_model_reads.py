"""Reflect is handed one page whole and the rest as snippets, and can read the rest.

`search_mental_models` used to answer with every hit's full text, so a bank with
five pages put all five into the prompt — and into every later turn of the loop.
It now answers the way the knowledge-page search does: the best-ranked page in
full, a snippet of the others, and `read_mental_models` for whatever else the
model decides it needs.

That is two capabilities meeting — the page layer and the reflect loop — so the
story drives the published client end to end and asserts on the tool trace the
API returns, which is the only place a caller can see which shape arrived.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

# Both pages are longer than a snippet (280 chars) and share no sentence, so a
# snippet is genuinely truncated and each page's tail identifies that page alone.
HOUSING = "Alice lives in Berlin. " + "She renewed the lease on the Kreuzberg flat. " * 8 + "The lease runs to 2027."
TRAVEL = "Alice travelled for work. " + "She was in Lisbon in March and in Porto in May. " * 8 + "Porto closed Q2."


@pytest.fixture
async def bank_with_two_pages(client, llm, bank_id, settled) -> str:
    """Two mental models with *different* text, so a search has a best hit AND a runner-up.

    Each page needs its own `llm.reset()`: stub rules match in registration order
    and persist, so registering a second `reflect_loop` without clearing would
    leave the first answer claiming every turn and give both pages the same body.
    """
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice travelled to Lisbon for work", who="Alice", entities=["Alice", "Lisbon"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=HOUSING)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Alice travelled to Lisbon for work.")
    await settled(bank_id)

    for name, query, answer in (
        ("Alice housing", "Where does Alice live?", HOUSING),
        ("Alice travel", "Where has Alice travelled?", TRAVEL),
    ):
        llm.reset()
        llm.on_step("consolidate").returns(consolidation())
        reflect_loop(llm, answer=answer)
        await client.mental_models.create_mental_model(bank_id, {"name": name, "source_query": query})
        await settled(bank_id)
    return bank_id


#: What the scripted model searches for, in both runs of the read story.
_SEARCH_QUERY = "Alice travel"


def _tool_calls(response, name: str) -> list:
    return [c for c in ((response.trace.tool_calls if response.trace else None) or []) if c.tool == name]


async def test_a_search_returns_one_page_whole_and_the_others_as_snippets(client, llm, bank_with_two_pages):
    """The bound that did not exist before: a second page costs a snippet, not a document."""
    llm.reset()
    reflect_loop(llm, answer="Alice lives in Berlin.")

    response = await client.areflect(
        bank_id=bank_with_two_pages, query="Where does Alice live?", include_tool_calls=True
    )

    (search,) = _tool_calls(response, "search_mental_models")
    pages = search.output["mental_models"]
    assert len(pages) == 2, f"both pages should be found: {pages}"
    whole = [p for p in pages if "content" in p]
    snippets = [p for p in pages if "snippet" in p]
    assert len(whole) == 1, "exactly the best-ranked page arrives in full"
    assert len(snippets) == 1, "the runner-up arrives as a snippet"
    assert snippets[0]["content_chars"] > 0, "with its size, so the model can judge what reading it would cost"
    assert "content" not in snippets[0]


async def test_the_model_can_read_a_page_it_only_saw_as_a_snippet(client, llm, bank_with_two_pages):
    """The other half of the trade: a snippet is not a dead end.

    The stub answers only once the page's TAIL is in the conversation — text a
    snippet cannot carry — so this passes only if the read actually returned the
    page, not if the model merely asked for it.
    """
    llm.reset()
    # The same search query in both runs, or a different page ranks top and the
    # page read below would be the one the search already returned whole.
    reflect_loop(llm, answer="checking", query=_SEARCH_QUERY)
    first = await client.areflect(
        bank_id=bank_with_two_pages, query="Where has Alice travelled?", include_tool_calls=True
    )
    (search,) = _tool_calls(first, "search_mental_models")
    snippet_page = next(p for p in search.output["mental_models"] if "snippet" in p)

    page = await client.mental_models.get_mental_model(bank_with_two_pages, snippet_page["id"], detail="full")
    tail = page.content.strip()[-40:]
    assert tail not in str(search.output), "the snippet already carries the tail, so a read would prove nothing"

    llm.reset()
    # Finish only when the read has put the tail in the conversation...
    llm.on_step("reflect", contains=tail).returns_tool_call("done", answer="Alice travelled to Lisbon and Porto.")
    # ...until then, read the page the search only summarised.
    llm.on_step("reflect", tool="read_mental_models").returns_tool_call(
        "read_mental_models", mental_model_ids=[snippet_page["id"]]
    )
    llm.on_step("reflect").calls_the_offered_tool(query=_SEARCH_QUERY)
    llm.on_step("reflect").returns_text("Alice travelled to Lisbon and Porto.")

    response = await client.areflect(
        bank_id=bank_with_two_pages,
        query="Where has Alice travelled?",
        include_tool_calls=True,
        include_facts=True,
    )

    reads = _tool_calls(response, "read_mental_models")
    assert reads, "the model asked for the page, so the read must have run"
    (read_page,) = reads[0].output["mental_models"]
    assert read_page["id"] == snippet_page["id"]
    assert read_page["content"].strip().endswith(tail), "reading a page returns its text, not another snippet"

    # The citation carries what the model actually read. Built from the search
    # output alone it carried the truncated snippet — or, for a page that only
    # arrived as a snippet, nothing at all but the page's name.
    cited = [m for m in (response.based_on.mental_models or []) if snippet_page["id"] in (m.id or "")]
    assert cited, f"the page the answer was read from must be cited: {response.based_on.mental_models}"
    assert tail in cited[0].text, "the citation shows the page's full text, not the snippet"
