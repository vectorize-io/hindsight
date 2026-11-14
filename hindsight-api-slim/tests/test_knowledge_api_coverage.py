"""Knowledge banks HTTP API: the parameters, filters and error answers the other suites leave out.

Every test drives the real route through ``api_client``; the embedder is the deterministic fake
from ``test_knowledge_banks``.
"""

import base64
import uuid

import pytest

from hindsight_api.config import clear_config_cache
from tests.test_knowledge_banks import (  # noqa: F401  (fixtures)
    KB,
    _pdf,
    _png,
    _search,
    _tone,
    _upload,
    fake_embedder,
    kb,
    needs_ffmpeg,
)


async def _ingest(client, bank_id: str, *items: dict) -> None:
    r = await client.post(f"{KB}/{bank_id}/ingest", json={"items": list(items)})
    assert r.status_code == 200, r.text


def _ids(results: list[dict]) -> list[str]:
    return sorted(r["document_id"] for r in results)


# --- banks ----------------------------------------------------------------------------------------


async def test_a_new_bank_starts_empty_and_creating_it_again_returns_it_unchanged(api_client, kb):
    await _ingest(api_client, kb, {"document_id": "a", "text": "red"})
    r = await api_client.post(KB, json={"bank_id": kb})
    assert r.status_code == 200, r.text
    assert (r.json()["bank_id"], r.json()["document_count"], r.json()["ready_count"]) == (kb, 1, 1)

    fresh = f"kb-{uuid.uuid4().hex[:8]}"
    try:
        body = (await api_client.post(KB, json={"bank_id": fresh})).json()
        assert body["bank_id"] == fresh and body["created_at"]
        assert {k: body[k] for k in ("document_count", "passage_count", "pending_count", "failed_count")} == {
            "document_count": 0,
            "passage_count": 0,
            "pending_count": 0,
            "failed_count": 0,
        }
        assert body["total_bytes"] == 0 and body["documents_by_media_type"] == {}
    finally:
        await api_client.delete(f"{KB}/{fresh}")


async def test_a_bank_needs_a_valid_id(api_client, fake_embedder):
    assert (await api_client.post(KB, json={})).status_code == 422
    r = await api_client.post(KB, json={"bank_id": "_reserved"})
    assert r.status_code == 400 and "reserved" in r.json()["detail"]
    assert (await api_client.get(f"{KB}/_reserved")).status_code == 404


async def test_the_bank_list_refuses_a_negative_page(api_client, fake_embedder):
    assert (await api_client.get(KB, params={"limit": -1})).status_code == 422
    assert (await api_client.get(KB, params={"offset": -1})).status_code == 422
    body = (await api_client.get(KB, params={"limit": 0})).json()
    assert body["banks"] == [] and body["limit"] == 0 and body["total"] >= 0


async def test_deleting_a_missing_bank_is_a_404(api_client, fake_embedder):
    r = await api_client.delete(f"{KB}/missing-{uuid.uuid4().hex[:8]}")
    assert r.status_code == 404 and "not found" in r.json()["detail"]


async def test_a_deleted_bank_takes_its_documents_with_it(api_client, kb):
    await _ingest(api_client, kb, {"document_id": "a", "text": "red"})
    r = await api_client.delete(f"{KB}/{kb}")
    assert r.status_code == 200 and r.json() == {"deleted": True}
    assert (await api_client.get(f"{KB}/{kb}/documents/a")).status_code == 404
    assert (await api_client.get(f"{KB}/{kb}/documents")).status_code == 404
    assert (await api_client.post(f"{KB}/{kb}/search", json={"query": "red"})).status_code == 404


# --- config ---------------------------------------------------------------------------------------


async def test_a_config_patch_needs_an_updates_object(api_client, kb):
    assert (await api_client.patch(f"{KB}/{kb}/config", json={})).status_code == 422
    assert (await api_client.patch(f"{KB}/{kb}/config", json={"updates": [1]})).status_code == 422


async def test_config_writes_are_refused_when_the_bank_config_api_is_off(api_client, kb, monkeypatch):
    await api_client.patch(f"{KB}/{kb}/config", json={"updates": {"text_passage_tokens": 64}})
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_BANK_CONFIG_API", "false")
    clear_config_cache()
    try:
        r = await api_client.patch(f"{KB}/{kb}/config", json={"updates": {"text_passage_tokens": 32}})
        assert r.status_code == 404 and "Bank configuration API is disabled" in r.json()["detail"]
        assert (await api_client.delete(f"{KB}/{kb}/config")).status_code == 404
        # Reads still answer, and nothing was written or reset.
        r = await api_client.get(f"{KB}/{kb}/config")
        assert r.status_code == 200 and r.json()["overrides"] == {"text_passage_tokens": 64}
    finally:
        monkeypatch.undo()
        clear_config_cache()


# --- timeseries -----------------------------------------------------------------------------------


async def test_timeseries_defaults_to_seven_days_and_buckets_one_day_by_the_hour(api_client, kb):
    await _ingest(api_client, kb, {"document_id": "a", "text": "red"}, {"document_id": "b", "text": "blue"})
    default = (await api_client.get(f"{KB}/{kb}/stats/timeseries")).json()
    assert (default["bank_id"], default["period"], default["trunc"]) == (kb, "7d", "day")
    for period, trunc in (("1d", "hour"), ("30d", "day"), ("90d", "day")):
        body = (await api_client.get(f"{KB}/{kb}/stats/timeseries", params={"period": period})).json()
        assert (body["period"], body["trunc"]) == (period, trunc)
        assert sum(b["documents"] for b in body["buckets"]) == 2
        assert sum(b["passages"] for b in body["buckets"]) == 2


async def test_timeseries_refuses_an_unknown_media_type_and_a_missing_bank(api_client, kb):
    r = await api_client.get(f"{KB}/{kb}/stats/timeseries", params={"media_type": "hologram"})
    assert r.status_code == 422
    r = await api_client.get(f"{KB}/missing-{uuid.uuid4().hex[:8]}/stats/timeseries")
    assert r.status_code == 404


# --- text ingest ----------------------------------------------------------------------------------


async def test_a_text_without_an_id_or_filename_gets_both_and_keeps_its_tags(api_client, kb):
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"text": "red notes", "tags": ["t1"]}]})
    assert r.status_code == 200, r.text
    (document_id,) = r.json()["document_ids"]
    uuid.UUID(document_id)
    doc = (await api_client.get(f"{KB}/{kb}/documents/{document_id}")).json()
    assert (doc["filename"], doc["content_type"], doc["media_type"]) == (f"{document_id}.md", "text/markdown", "text")
    assert doc["tags"] == ["t1"] and doc["status"] == "ready"


async def test_a_text_ingest_refuses_an_empty_or_oversized_batch(api_client, kb):
    url = f"{KB}/{kb}/ingest"
    assert (await api_client.post(url, json={"items": []})).status_code == 422
    assert (await api_client.post(url, json={"items": [{"text": ""}]})).status_code == 422
    too_many = [{"text": f"n{i}"} for i in range(101)]
    assert (await api_client.post(url, json={"items": too_many})).status_code == 422
    assert (await api_client.get(f"{KB}/{kb}/documents")).json()["total"] == 0


async def test_ingesting_into_a_missing_bank_is_a_404(api_client, fake_embedder):
    missing = f"missing-{uuid.uuid4().hex[:8]}"
    r = await api_client.post(f"{KB}/{missing}/ingest", json={"items": [{"text": "red"}]})
    assert r.status_code == 404
    r = await _upload(api_client, missing, [("a.md", b"red", "text/markdown")])
    assert r.status_code == 404


# --- file ingest ----------------------------------------------------------------------------------


async def test_a_file_ingest_refuses_a_bad_request_field(api_client, kb):
    files = [("files", ("a.md", b"red", "text/markdown"))]
    r = await api_client.post(f"{KB}/{kb}/files/ingest", files=files, data={"request": "{not json"})
    assert r.status_code == 400 and "invalid 'request' field" in r.json()["detail"]
    r = await api_client.post(f"{KB}/{kb}/files/ingest", files=files, data={"request": '{"files": [{"tags": 5}]}'})
    assert r.status_code == 400 and "invalid 'request' field" in r.json()["detail"]
    r = await _upload(api_client, kb, [("a.md", b"red", "text/markdown")], [{"document_id": "a"}, {"document_id": "b"}])
    assert r.status_code == 400 and "one entry per file" in r.json()["detail"]
    assert (await api_client.post(f"{KB}/{kb}/files/ingest")).status_code == 422
    assert (await api_client.get(f"{KB}/{kb}/documents")).json()["total"] == 0


async def test_an_untyped_upload_is_typed_from_its_filename(api_client, kb):
    r = await _upload(api_client, kb, [("photo.png", _png((255, 0, 0)), "application/octet-stream")])
    assert r.status_code == 202, r.text
    (document_id,) = r.json()["document_ids"]
    doc = (await api_client.get(f"{KB}/{kb}/documents/{document_id}")).json()
    assert (doc["filename"], doc["content_type"], doc["media_type"]) == ("photo.png", "image/png", "image")
    file = await api_client.get(doc["document_url"])
    assert file.headers["content-type"] == "image/png"


async def test_an_empty_or_too_big_file_is_refused(api_client, kb, monkeypatch):
    r = await _upload(api_client, kb, [("empty.md", b"", "text/markdown")])
    assert r.status_code == 400 and "is empty" in r.json()["detail"]
    monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_MAX_FILE_MB", "1")
    clear_config_cache()
    try:
        r = await _upload(api_client, kb, [("big.md", b"x" * (1024 * 1024 + 1), "text/markdown")])
        assert r.status_code == 413 and "the limit is 1MB" in r.json()["detail"]
    finally:
        monkeypatch.undo()
        clear_config_cache()
    assert (await api_client.get(f"{KB}/{kb}/documents")).json()["total"] == 0


# --- documents ------------------------------------------------------------------------------------


@pytest.fixture
async def mixed_bank(api_client, kb):
    """Three documents of three media types and tags: text, an image and a two-page PDF."""
    await _ingest(api_client, kb, {"document_id": "notes", "text": "red notes", "filename": "notes.md", "tags": ["a"]})
    r = await _upload(
        api_client,
        kb,
        [
            ("photo.png", _png((255, 0, 0)), "image/png"),
            ("manual.pdf", _pdf((250, 0, 0), (0, 0, 250)), "application/pdf"),
        ],
        [{"document_id": "photo", "tags": ["a", "b"]}, {"document_id": "manual", "tags": ["c"]}],
    )
    assert r.status_code == 202, r.text
    return kb


async def test_the_document_list_filters_by_text_media_type_tags_and_status(api_client, mixed_bank):
    async def listed(**params) -> list[str]:
        r = await api_client.get(f"{KB}/{mixed_bank}/documents", params=params)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == len(body["documents"])
        return sorted(d["document_id"] for d in body["documents"])

    assert await listed() == ["manual", "notes", "photo"]
    assert await listed(q="PHOTO") == ["photo"]  # the id, case-insensitive
    assert await listed(q="manual.pdf") == ["manual"]  # the filename
    assert await listed(media_type="document") == ["manual"]
    assert await listed(media_type="image") == ["photo"]
    assert await listed(media_type="text") == ["notes"]
    assert await listed(tags="b") == ["photo"]
    assert await listed(tags=["b", "c"]) == ["manual", "photo"]  # any of them
    assert await listed(status="ready") == ["manual", "notes", "photo"]
    assert await listed(status="failed") == []
    assert await listed(q="photo", media_type="text") == []


async def test_the_document_list_pages_newest_first(api_client, mixed_bank):
    first = (await api_client.get(f"{KB}/{mixed_bank}/documents", params={"limit": 2})).json()
    rest = (await api_client.get(f"{KB}/{mixed_bank}/documents", params={"limit": 2, "offset": 2})).json()
    assert first["total"] == 3 and rest["total"] == 3
    assert len(first["documents"]) == 2 and len(rest["documents"]) == 1
    seen = [d["document_id"] for d in first["documents"] + rest["documents"]]
    assert sorted(seen) == ["manual", "notes", "photo"]
    updated = [d["updated_at"] for d in first["documents"] + rest["documents"]]
    assert updated == sorted(updated, reverse=True)


async def test_the_document_list_refuses_a_page_out_of_bounds(api_client, kb):
    url = f"{KB}/{kb}/documents"
    for params in ({"limit": 0}, {"limit": 501}, {"offset": -1}):
        assert (await api_client.get(url, params=params)).status_code == 422, params
    assert (await api_client.get(url, params={"limit": 500})).status_code == 200


async def test_a_document_detail_lists_its_passages_with_media_links(api_client, mixed_bank):
    doc = (await api_client.get(f"{KB}/{mixed_bank}/documents/manual")).json()
    assert doc["document_url"] == f"{KB}/{mixed_bank}/documents/manual/file"
    assert doc["size_bytes"] > 0 and doc["tags"] == ["c"] and doc["error"] is None
    assert [(p["index"], p["page"], p["media_type"]) for p in doc["passages"]] == [
        (0, 1, "document"),
        (1, 2, "document"),
    ]
    assert doc["passages"][1]["passage_url"] == f"{KB}/{mixed_bank}/documents/manual/passages/1/media"
    text = (await api_client.get(f"{KB}/{mixed_bank}/documents/notes")).json()
    assert text["passages"][0]["passage_url"] is None and text["passages"][0]["text"] == "red notes"


async def test_deleting_a_missing_document_is_a_404(api_client, kb):
    r = await api_client.delete(f"{KB}/{kb}/documents/nope")
    assert r.status_code == 404 and "not found" in r.json()["detail"]


# --- file and passage media -----------------------------------------------------------------------


async def test_the_original_file_comes_back_typed_and_named(api_client, kb):
    await _ingest(api_client, kb, {"document_id": "n", "text": "red notes", "filename": "my notes é.md"})
    r = await api_client.get(f"{KB}/{kb}/documents/n/file")
    assert r.status_code == 200 and r.content == b"red notes"
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.headers["content-disposition"] == "inline; filename*=UTF-8''my%20notes%20%C3%A9.md"


async def test_a_range_request_on_a_file_gets_the_whole_file(api_client, kb):
    # By design: the dataplane does not answer Range; the control plane's proxy slices media itself.
    await _ingest(api_client, kb, {"document_id": "n", "text": "red notes"})
    r = await api_client.get(f"{KB}/{kb}/documents/n/file", headers={"Range": "bytes=0-2"})
    assert r.status_code == 200 and r.content == b"red notes"


async def test_a_missing_file_is_a_404(api_client, kb):
    r = await api_client.get(f"{KB}/{kb}/documents/nope/file")
    assert r.status_code == 404 and "not found" in r.json()["detail"]
    r = await api_client.get(f"{KB}/missing-{uuid.uuid4().hex[:8]}/documents/nope/file")
    assert r.status_code == 404


async def test_passage_media_is_cacheable_and_a_passage_without_media_is_a_404(api_client, mixed_bank):
    r = await api_client.get(f"{KB}/{mixed_bank}/documents/photo/passages/0/media")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "private, max-age=3600"
    for path in ("notes/passages/0/media", "photo/passages/9/media", "nope/passages/0/media"):
        r = await api_client.get(f"{KB}/{mixed_bank}/documents/{path}")
        assert r.status_code == 404 and r.json()["detail"] == "this passage has no media", path
    assert (await api_client.get(f"{KB}/{mixed_bank}/documents/photo/passages/x/media")).status_code == 422


# --- search ---------------------------------------------------------------------------------------


async def test_a_search_hit_names_its_document_and_where_to_fetch_it(api_client, mixed_bank):
    (hit,) = await _search(api_client, mixed_bank, query="red", media_types=["text"])
    assert hit == {
        "index": 0,
        "media_type": "text",
        "text": "red notes",
        "page": None,
        "start_ms": None,
        "end_ms": None,
        "passage_url": None,
        "document_id": "notes",
        "filename": "notes.md",
        "tags": ["a"],
        "document_url": f"{KB}/{mixed_bank}/documents/notes/file",
    }
    (image,) = await _search(api_client, mixed_bank, query="red", media_types=["image"])
    assert image["passage_url"] == f"{KB}/{mixed_bank}/documents/photo/passages/0/media"


async def test_each_media_type_filter_keeps_only_that_type(api_client, mixed_bank):
    async def found(*media_types: str) -> list[str]:
        return _ids(await _search(api_client, mixed_bank, query="red", media_types=list(media_types)))

    assert await found("text") == ["notes"]
    assert await found("image") == ["photo"]
    assert sorted(set(await found("document"))) == ["manual"]
    assert sorted(set(await found("image", "text"))) == ["notes", "photo"]
    assert await found("audio") == [] and await found("video") == []


@needs_ffmpeg
async def test_audio_and_video_media_type_filters_keep_only_that_type(api_client, kb):
    await _ingest(api_client, kb, {"document_id": "notes", "text": "audio video notes"})
    await _upload(
        api_client,
        kb,
        [("talk.m4a", _tone(3, video=False), "audio/mp4"), ("demo.mp4", _tone(3, video=True), "video/mp4")],
        [{"document_id": "talk"}, {"document_id": "demo"}],
    )
    query = "audio video"
    assert _ids(await _search(api_client, kb, query=query, media_types=["audio"])) == ["talk"]
    assert _ids(await _search(api_client, kb, query=query, media_types=["video"])) == ["demo"]
    assert _ids(await _search(api_client, kb, query=query, media_types=["audio", "video"])) == ["demo", "talk"]


async def test_search_filters_combine_as_and(api_client, mixed_bank):
    async def found(**filters) -> list[str]:
        return sorted(set(_ids(await _search(api_client, mixed_bank, query="red", **filters))))

    assert await found(tags=["a"]) == ["notes", "photo"]
    assert await found(tags=["a"], media_types=["image"]) == ["photo"]
    assert await found(tags=["a"], document_ids=["notes", "manual"]) == ["notes"]
    assert await found(tags=["a", "b"], tags_match="all") == ["photo"]
    assert await found(tags=["a", "c"], tags_match="all") == []
    assert await found(document_ids=["nope"]) == []


async def test_search_refuses_out_of_range_parameters(api_client, kb):
    url = f"{KB}/{kb}/search"
    for body in (
        {"query": "red", "max_tokens": 0},
        {"query": "red", "max_tokens": 1_000_001},
        {"query": "red", "tags_match": "some"},
        {"query": "red", "tags": "a"},
    ):
        assert (await api_client.post(url, json=body)).status_code == 422, body
    assert (await api_client.post(url, json={"query": "red", "max_tokens": 1_000_000})).status_code == 200


async def test_a_query_image_must_be_base64(api_client, kb):
    r = await api_client.post(f"{KB}/{kb}/search", json={"query_image": "not base64!"})
    assert r.status_code == 400 and r.json()["detail"] == "query_image must be base64"
    image = base64.b64encode(_png((255, 0, 0))).decode()
    assert (await api_client.post(f"{KB}/{kb}/search", json={"query_image": image})).status_code == 200
