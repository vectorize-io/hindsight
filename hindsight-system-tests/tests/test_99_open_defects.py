"""Contracts the product does not currently honour.

Every test in this file **fails today**, on purpose. Each asserts the behaviour
we want, names the issue tracking it, and goes green when that issue is fixed.

They live together rather than in their topical stories because they are
temporary: when an issue lands, its test moves into the story it belongs to and
this file shrinks. When the file is empty, delete it.

Two things this file is deliberately *not*:

- **Not `xfail`.** An expected-failure marker is a note that the suite has
  agreed to look away. It keeps the run green, so nothing forces the question,
  and the marker outlives the bug by months.
- **Not a pin of current behaviour.** A test asserting today's wrong answer
  fails the day someone fixes it, which teaches people to delete tests instead
  of reading them.

So these are red, and the suite is red with them, until the product changes.
"""

from __future__ import annotations

import inspect

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# #4217 — the recall trace reports wall-clock time, not the caller's anchor
# ---------------------------------------------------------------------------


async def test_the_trace_reports_the_anchor_the_query_actually_used(client, llm, bank_id, settled):
    """A trace exists to explain a ranking, so it has to report the inputs that
    produced it.

    `query_timestamp` moves the "now" that recency decays from, and it does reach
    the engine — story 07 asserts its effect. But `tracer.py:396` stamps the
    trace with `datetime.now(UTC)`, so someone debugging "why did my 2019 memory
    rank low when I asked as of 2020?" reads today's date and concludes their
    anchor was ignored. The field points away from the answer.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    response = await client.arecall(
        bank_id=bank_id, query="Where does Alice live?", query_timestamp="2020-01-01T00:00:00", trace=True
    )

    assert response.trace["query"]["timestamp"].startswith("2020-01-01"), (
        "the trace reports when it was built, not the anchor the query ran against — #4217"
    )


# ---------------------------------------------------------------------------
# #4218 — list endpoints return untyped rows, unlike their single-fetch siblings
# ---------------------------------------------------------------------------


async def test_document_list_rows_are_typed_like_the_single_fetch(client, llm, bank_id, settled):
    """`get_document` returns a model; `list_documents` returns dicts.

    Same resource, two conventions, so a caller cannot learn one. The natural
    `listing.items[0].id` raises `AttributeError`, and a field renamed
    server-side is a compile error for one and a runtime `KeyError` for the
    other — found in production rather than at build time.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="d1")
    await settled(bank_id)

    listing = await client.documents.list_documents(bank_id)

    assert listing.items[0].id == "d1", "list rows are untyped dicts — #4218"


async def test_memory_list_rows_are_typed(client, llm, bank_id, settled):
    """The same defect on the endpoint most callers touch first."""
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=10)

    assert memories.items[0].text, "memory list rows are untyped dicts — #4218"


# ---------------------------------------------------------------------------
# #4221 — most wrapper methods have no async variant, despite the docstring
# ---------------------------------------------------------------------------

# `close`/`aclose` manage the client's own connection pool rather than calling
# the API. Both halves are excluded, not just `close` — excluding one orphans the
# other, which then reads as a method missing *its* async twin.
_NOT_API_CALLS = {"close", "aclose"}


def _convenience_methods() -> list[str]:
    """Public methods on the wrapper that are not already the async half of a pair."""
    names = [
        name
        for name, _ in inspect.getmembers(Hindsight, callable)
        if not name.startswith("_") and name not in _NOT_API_CALLS
    ]
    async_variants = {name for name in names if name.startswith("a") and name[1:] in names}
    return sorted(name for name in names if name not in async_variants)


async def test_every_convenience_method_has_an_async_variant():
    """The wrapper's own docstring says so:

        "Every convenience method has an async counterpart prefixed with `a`
        ... **Prefer the async variants** whenever you are inside an async
        context (`async def`, event loops, frameworks like
        FastAPI/LangGraph/CrewAI)."

    27 do not. And they are not merely inconvenient there — they route through
    `loop.run_until_complete`, which raises inside a running loop. So from
    exactly the frameworks that sentence names, mental models, knowledge pages,
    directives and bank config are unreachable through the wrapper.

    Written over the whole family rather than as a list of the 27, so the *next*
    method added without a twin fails here too — the structural-guard shape from
    the code-review skill (§9a).
    """
    missing = [name for name in _convenience_methods() if not hasattr(Hindsight, f"a{name}")]

    assert missing == [], f"{len(missing)} convenience methods have no async variant — #4221: {missing}"


# ---------------------------------------------------------------------------
# #4230 — a failed structured-output extraction is silent
# ---------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


async def test_a_failed_structured_extraction_is_distinguishable_from_an_empty_one(client, llm, bank_id, settled):
    """`structured_output: null` currently means three different things.

    The extraction call errored, or returned unparseable text, or the answer
    genuinely had nothing matching the schema. The first two are retryable and
    worth alerting on; the third is normal. Collapsing them means a caller either
    retries every empty result or none, and an operator watching for "structured
    output stopped working" has no signal — the request succeeded.

    Returning the prose answer rather than failing the whole reflect is right.
    Saying nothing about the half that did not happen is not.
    """
    from hindsight_system_tests import reflect_loop

    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    reflect_loop(llm, answer="Alice lives in Berlin.")
    llm.on_step("reflect_structured").returns_text("absolutely not json")

    response = await client.areflect(bank_id=bank_id, query="Where does Alice live?", response_schema=SCHEMA)

    assert response.structured_output is None
    assert getattr(response, "structured_output_error", None), (
        "a failed extraction is reported exactly like an empty one — #4230"
    )
