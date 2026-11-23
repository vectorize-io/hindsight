"""The wrapper builds correct knowledge-bank requests.

Mirrors the TypeScript wrapper's ``knowledge_banks_mapping`` tests so the two
hand-written clients stay at parity.
"""

import json
from unittest.mock import MagicMock

import pytest

from hindsight_client import Hindsight


@pytest.fixture
def client():
    return Hindsight(base_url="http://example.invalid")


def _capture(monkeypatch, client, method):
    captured: dict[str, object] = {}

    async def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(client._knowledge_banks_api, method, fake)
    return captured


def test_search_maps_every_option(monkeypatch, client):
    captured = _capture(monkeypatch, client, "search_knowledge_bank")

    client.search_knowledge_bank(
        "kb-1",
        query="return window",
        query_image="data:image/png;base64,AAAA",
        query_video="data:video/mp4;base64,BBBB",
        budget="high",
        max_tokens=2048,
        tags=["policy"],
        tags_match="all_strict",
        tag_groups=[{"tags": ["x200"], "match": "any"}],
        document_ids=["returns-policy"],
        media_types=["text", "document"],
        enable_reranking=False,
        trace=True,
    )

    bank_id, request = captured["args"]
    assert bank_id == "kb-1"
    assert request.to_dict() == {
        "query": "return window",
        "query_image": "data:image/png;base64,AAAA",
        "query_video": "data:video/mp4;base64,BBBB",
        "budget": "high",
        "max_tokens": 2048,
        "tags": ["policy"],
        "tags_match": "all_strict",
        "tag_groups": [{"tags": ["x200"], "match": "any", "resolve": "exact"}],
        "document_ids": ["returns-policy"],
        "media_types": ["text", "document"],
        "enable_reranking": False,
        "trace": True,
    }


def test_ingest_texts_maps_items_and_async(monkeypatch, client):
    captured = _capture(monkeypatch, client, "ingest_knowledge_texts")

    client.ingest_knowledge_texts(
        "kb-1",
        [{"text": "# Returns", "document_id": "returns", "filename": "r.md", "tags": ["policy"], "strategy": "s"}],
        ingest_async=True,
    )

    bank_id, request = captured["args"]
    assert bank_id == "kb-1"
    assert request.to_dict() == {
        "items": [
            {"text": "# Returns", "document_id": "returns", "filename": "r.md", "tags": ["policy"], "strategy": "s"}
        ],
        "async": True,
    }


def test_ingest_files_reads_paths_and_takes_bytes(monkeypatch, client, tmp_path):
    captured = _capture(monkeypatch, client, "ingest_knowledge_files")
    manual = tmp_path / "manual.pdf"
    manual.write_bytes(b"%PDF-1.4")

    client.ingest_knowledge_files(
        "kb-1",
        [manual, ("quick.md", b"# Quick start")],
        files_options=[{"document_id": "manual", "tags": ["x200"]}, {"document_id": "quick"}],
    )

    kwargs = captured["kwargs"]
    assert kwargs["bank_id"] == "kb-1"
    assert kwargs["files"] == [("manual.pdf", b"%PDF-1.4"), ("quick.md", b"# Quick start")]
    assert json.loads(kwargs["request"]) == {
        "files": [{"document_id": "manual", "tags": ["x200"]}, {"document_id": "quick"}]
    }


def test_ingest_files_without_options_sends_no_request_field(monkeypatch, client):
    captured = _capture(monkeypatch, client, "ingest_knowledge_files")

    client.ingest_knowledge_files("kb-1", [("a.md", b"a")])

    assert captured["kwargs"]["request"] is None


def test_ingest_files_refuses_options_that_do_not_match_the_files(client):
    with pytest.raises(ValueError, match="one entry per file"):
        client.ingest_knowledge_files("kb-1", [("a.md", b"a")], files_options=[{}, {}])


def test_list_documents_maps_filters(monkeypatch, client):
    captured = _capture(monkeypatch, client, "list_knowledge_documents")

    client.list_knowledge_documents(
        "kb-1", q="x200", status="ready", media_type="document", tags=["x200"], limit=10, offset=20
    )

    assert captured["args"] == ("kb-1",)
    kwargs = captured["kwargs"]
    assert {k: kwargs[k] for k in ("q", "status", "media_type", "tags", "limit", "offset")} == {
        "q": "x200",
        "status": "ready",
        "media_type": "document",
        "tags": ["x200"],
        "limit": 10,
        "offset": 20,
    }


def test_update_config_sends_updates_including_clears(monkeypatch, client):
    captured = _capture(monkeypatch, client, "update_knowledge_bank_config")

    client.update_knowledge_bank_config("kb-1", {"text_passage_tokens": 256, "disabled_media_types": None})

    bank_id, request = captured["args"]
    assert bank_id == "kb-1"
    assert request.updates == {"text_passage_tokens": 256, "disabled_media_types": None}
