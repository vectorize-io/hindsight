"""A date written in the question narrows recall to that time.

"Where did Alice hike last month?" names a window. Recall reads it out of the query
(text2date by default, HINDSIGHT_API_QUERY_ANALYZER), relative to the caller's
`query_timestamp`, and the temporal arm ranks the facts by how close they happened
to it. This story pins the window the server read and the temporal arm's order; a
question with no date must leave the window empty.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


async def test_a_date_in_the_query_becomes_the_recall_window(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(
            fact(
                "Alice hiked the Dolomites",
                who="Alice",
                fact_kind="event",
                occurred_start="2025-10-14T00:00:00Z",
                occurred_end="2025-10-14T23:59:59Z",
                entities=["Alice", "Dolomites"],
            ),
            fact(
                "Alice hiked the Pyrenees",
                who="Alice",
                fact_kind="event",
                occurred_start="2023-09-02T00:00:00Z",
                occurred_end="2023-09-02T23:59:59Z",
                entities=["Alice", "Pyrenees"],
            ),
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice hiked the Dolomites and, years before, the Pyrenees.")
    await settled(bank_id)

    # Asked on 18 November 2025: "last month" is all of October 2025.
    response = await client.arecall(
        bank_id=bank_id,
        query="Where did Alice hike last month?",
        query_timestamp="2025-11-18T12:00:00Z",
        trace=True,
    )
    window = response.trace["query"]["temporal_constraint"]
    assert window is not None, "the date in the query must become a window"
    assert window["start"].startswith("2025-10-01T00:00:00")
    assert window["end"].startswith("2025-10-31T23:59:59")

    # Both facts match the query's words equally; only the temporal arm can tell them
    # apart, and it must put the October hike ahead of the 2023 one.
    # The reranker does not see that score yet: fusion drops it (#6991).
    temporal_arm = next(
        a for a in response.trace["retrieval_results"] if (a["method_name"], a["fact_type"]) == ("temporal", "world")
    )
    assert [r["text"] for r in temporal_arm["results"]] == [
        "Alice hiked the Dolomites | Involving: Alice",
        "Alice hiked the Pyrenees | Involving: Alice",
    ]
    assert temporal_arm["results"][0]["score"] > temporal_arm["results"][1]["score"]
    assert [r.text for r in response.results] == [
        "Alice hiked the Dolomites | Involving: Alice",
        "Alice hiked the Pyrenees | Involving: Alice",
    ]

    # No date in the question: no window.
    plain = await client.arecall(
        bank_id=bank_id, query="Where did Alice hike?", query_timestamp="2025-11-18T12:00:00Z", trace=True
    )
    assert plain.trace["query"]["temporal_constraint"] is None
    assert all(a["results"] == [] for a in plain.trace["retrieval_results"] if a["method_name"] == "temporal")
