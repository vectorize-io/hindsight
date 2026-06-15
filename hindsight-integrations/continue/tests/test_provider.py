"""Unit tests for build_context_items against Continue's request contract."""

from __future__ import annotations

import pytest

from hindsight_continue import (
    HindsightError,
    build_context_items,
    configure,
    serialize,
)

from .conftest import continue_request, make_client


class TestBuildContextItems:
    def test_recalls_against_query_and_returns_one_item(self):
        client = make_client(["User prefers dark mode", "Project uses pnpm"])
        configure(bank_id="proj-1")

        items = build_context_items(continue_request(query="ui preferences"), client=client)

        client.recall.assert_called_once()
        kwargs = client.recall.call_args.kwargs
        assert kwargs["bank_id"] == "proj-1"
        assert kwargs["query"] == "ui preferences"
        assert kwargs["budget"] == "mid"
        assert kwargs["max_tokens"] == 2048

        assert len(items) == 1
        assert items[0].name == "Hindsight Memory"
        assert "1. User prefers dark mode" in items[0].content
        assert "2. Project uses pnpm" in items[0].content

    def test_falls_back_to_full_input_when_query_empty(self):
        client = make_client(["a memory"])
        configure(bank_id="proj-1")

        build_context_items(
            continue_request(query="", full_input="what did we decide about auth?"),
            client=client,
        )

        assert client.recall.call_args.kwargs["query"] == "what did we decide about auth?"

    def test_empty_input_returns_no_items_and_skips_recall(self):
        client = make_client(["unused"])
        configure(bank_id="proj-1")

        items = build_context_items(continue_request(query="", full_input="   "), client=client)

        assert items == []
        client.recall.assert_not_called()

    def test_no_results_returns_empty_list(self):
        client = make_client([])
        configure(bank_id="proj-1")

        items = build_context_items(continue_request(query="anything"), client=client)

        assert items == []

    def test_missing_bank_id_raises(self):
        client = make_client(["x"])
        configure()  # no bank id, no env

        with pytest.raises(HindsightError, match="bank id"):
            build_context_items(continue_request(query="hello"), client=client)
        client.recall.assert_not_called()

    def test_request_option_overrides_configured_bank(self):
        client = make_client(["x"])
        configure(bank_id="default-bank")

        build_context_items(
            continue_request(query="hi", options={"bankId": "override-bank"}),
            client=client,
        )

        assert client.recall.call_args.kwargs["bank_id"] == "override-bank"

    def test_recall_failure_is_wrapped(self):
        client = make_client()
        client.recall.side_effect = RuntimeError("boom")
        configure(bank_id="proj-1")

        with pytest.raises(HindsightError, match="Recall failed"):
            build_context_items(continue_request(query="hi"), client=client)

    def test_http_status_surfaced_in_error(self):
        # hindsight_client raises an ApiException with a ``status`` attribute;
        # the adapter should surface it so a bad token (401) is distinguishable.
        client = make_client()
        err = RuntimeError("unauthorized")
        err.status = 401
        client.recall.side_effect = err
        configure(bank_id="proj-1")

        with pytest.raises(HindsightError, match="HTTP 401"):
            build_context_items(continue_request(query="hi"), client=client)

    def test_types_and_tags_passed_through_when_configured(self):
        client = make_client(["m"])
        configure(
            bank_id="proj-1",
            recall_types=["world", "experience"],
            recall_tags=["team-a"],
            recall_tags_match="all",
        )

        build_context_items(continue_request(query="hi"), client=client)

        kwargs = client.recall.call_args.kwargs
        assert kwargs["types"] == ["world", "experience"]
        assert kwargs["tags"] == ["team-a"]
        assert kwargs["tags_match"] == "all"

    def test_types_and_tags_omitted_when_unset(self):
        client = make_client(["m"])
        configure(bank_id="proj-1")

        build_context_items(continue_request(query="hi"), client=client)

        kwargs = client.recall.call_args.kwargs
        assert "types" not in kwargs
        assert "tags" not in kwargs


class TestSerialize:
    def test_serialize_matches_continue_context_item_shape(self):
        client = make_client(["memory one"])
        configure(bank_id="proj-1")

        items = build_context_items(continue_request(query="hi"), client=client)
        payload = serialize(items)

        assert isinstance(payload, list)
        assert set(payload[0].keys()) == {"name", "description", "content"}
        assert all(isinstance(v, str) for v in payload[0].values())
