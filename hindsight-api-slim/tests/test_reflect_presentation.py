"""What the reflect model reads of a tool result (engine/reflect/presentation.py)."""

from hindsight_api.engine.reflect.presentation import ToolResultPresenter, compact_timestamp


def _observations(*ids: str) -> dict:
    return {
        "query": "q",
        "count": len(ids),
        "observations": [
            {
                "id": i,
                "text": f"fact {i}",
                "fact_type": "observation",
                "mentioned_at": "2026-03-07T12:34:56.789+00:00",
                "occurred_start": "2026-03-01T00:00:00+00:00",
                "occurred_end": "2026-03-01T00:00:00+00:00",
                "source_fact_ids": [f"src-{i}"],
            }
            for i in ids
        ],
    }


def test_dates_keep_their_minute():
    assert compact_timestamp("2026-03-07T12:34:56.789+00:00") == "2026-03-07 12:34"
    assert compact_timestamp("2026-03-07T12:34:56Z") == "2026-03-07 12:34"
    # Not a UTC timestamp: left exactly as it was rather than silently re-zoned.
    assert compact_timestamp("2026-03-07T12:34:56+02:00") == "2026-03-07T12:34:56+02:00"
    assert compact_timestamp(None) is None


def test_observation_is_written_short_but_keeps_every_fact():
    item = ToolResultPresenter().present(_observations("uuid-a"))["observations"][0]
    assert item == {
        "id": "o1",
        "text": "fact uuid-a",
        "mentioned_at": "2026-03-07 12:34",
        "occurred_start": "2026-03-01 00:00",
        "source_fact_ids": ["f1"],
    }


def test_a_distinct_occurred_end_and_a_mixed_fact_type_are_kept():
    out = ToolResultPresenter().present(
        {
            "memories": [
                {
                    "id": "m",
                    "text": "t",
                    "fact_type": "experience",
                    "occurred_start": "2026-03-01T00:00:00+00:00",
                    "occurred_end": "2026-03-04T00:00:00+00:00",
                }
            ]
        }
    )
    assert out["memories"][0]["fact_type"] == "experience"
    assert out["memories"][0]["occurred_end"] == "2026-03-04 00:00"


def test_aliases_the_model_writes_back_resolve_to_real_ids():
    presenter = ToolResultPresenter()
    presenter.present(_observations("uuid-a", "uuid-b"))
    args = {"memory_ids": ["o2", "o1", "unknown"], "query": "o1 again", "reason": "o2"}
    assert presenter.resolve(args) == {
        "memory_ids": ["uuid-b", "uuid-a", "unknown"],
        "query": "o1 again",
        "reason": "o2",
    }


def test_an_item_already_shown_is_referenced_not_repeated():
    presenter = ToolResultPresenter()
    presenter.present(_observations("uuid-a"))
    again = presenter.present(_observations("uuid-a", "uuid-b"))
    assert [o["id"] for o in again["observations"]] == ["o2"]
    assert again["already_shown"] == {"observations": ["o1"]}


def test_reading_a_page_seen_as_a_snippet_returns_its_text():
    """The dedupe keys on the shape, not the id alone.

    `search_mental_models` shows the best hit whole and the rest as snippets, so
    the SAME page id legitimately comes back twice: once truncated, once in full
    from the `read_mental_models` the model then issues. Keying on the id alone
    answered that read with `already_shown` and an empty list — the model asked
    for the page and was handed nothing, which the system story
    `test_37_mental_model_reads` caught end to end.
    """
    presenter = ToolResultPresenter()
    presenter.present({"mental_models": [{"id": "uuid-a", "snippet": "Alice lives in", "content_chars": 407}]})

    read = presenter.present({"mental_models": [{"id": "uuid-a", "content": "Alice lives in Berlin."}]})

    assert [p["content"] for p in read["mental_models"]] == ["Alice lives in Berlin."]
    assert "already_shown" not in read
    # A second read of the same page is still a back-reference.
    assert presenter.present({"mental_models": [{"id": "uuid-a", "content": "Alice lives in Berlin."}]}) == {
        "mental_models": [],
        "already_shown": {"mental_models": ["p1"]},
    }


def test_chunks_lose_their_long_keys_and_bookkeeping():
    out = ToolResultPresenter().present(
        {
            "memories": [],
            "chunks": {
                "bank_doc-uuid_3": {"chunk_text": "a", "chunk_index": 3, "truncated": False},
                "bank_doc-uuid_4": {"chunk_text": "b", "chunk_index": 4, "truncated": True},
            },
        }
    )
    assert out["chunks"] == {"c1": {"chunk_text": "a"}, "c2": {"chunk_text": "b", "truncated": True}}


def test_error_results_pass_through_untouched():
    error = {"error": "recall requires a query parameter"}
    assert ToolResultPresenter().present(error) is error
