"""Knowledge banks: multimodal chunking, write/search over HTTP, and the memory-bank link.

The embedder is a deterministic fake (keywords and colours mapped to fixed axes), so these
tests check the plumbing — what a file becomes, what is stored, what a filter keeps, what
reflect reads from a linked bank — not the model. Retrieval quality is measured by
hindsight-dev/benchmarks/knowledge_banks_mm.
"""

import asyncio
import io
import json
import math
import shutil
import subprocess
import uuid
from dataclasses import replace

import pytest
from PIL import Image

from hindsight_api.engine.token_encoding import count_tokens
from hindsight_api.knowledge import media
from hindsight_api.knowledge.embedders import EmbedInput, KnowledgeEmbedder
from hindsight_api.knowledge.service import _EMBED_ROUND, _Work, get_knowledge_service, run_ingest_task
from hindsight_api.knowledge.store import StatusUpdate

KB = "/v1/default/knowledge-banks"
_AXES = ("red", "blue", "key", "password", "invoice", "audio", "video", "green")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class FakeEmbedder(KnowledgeEmbedder):
    provider = "fake"
    model = "fake"
    dimension = len(_AXES)
    modalities = frozenset({"text", "image", "audio", "video"})

    def __init__(self, modalities: frozenset[str] | None = None) -> None:
        if modalities is not None:
            self.modalities = modalities  # type: ignore[assignment]
        self.calls: list[list[EmbedInput]] = []

    @staticmethod
    def _text(text: str) -> list[float]:
        words = text.lower()
        return [float(words.count(axis)) for axis in _AXES]

    @staticmethod
    def _norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else [1.0 / math.sqrt(len(v))] * len(v)

    def _one(self, item: EmbedInput) -> list[float]:
        if item.modality == "text":
            return self._norm(self._text(item.text or ""))
        if item.modality == "image":
            r, g, b = Image.open(io.BytesIO(item.data or b"")).convert("RGB").resize((1, 1)).getpixel((0, 0))
            v = [0.0] * len(_AXES)
            v[0], v[1], v[7] = r / 255, b / 255, g / 255
            return self._norm(v)
        v = [0.0] * len(_AXES)
        v[_AXES.index(item.modality)] = 1.0
        return v

    async def embed_documents(self, items: list[EmbedInput]) -> list[list[float]]:
        self.calls.append(items)
        return [self._one(i) for i in items]

    async def embed_query(
        self, text: str | None, image: bytes | None = None, video: bytes | None = None
    ) -> list[float]:
        """One vector for the whole query, as the real models give: here the sum of its parts."""
        parts = [self._text(text)] if text else []
        if image is not None:
            parts.append(self._one(EmbedInput(modality="image", data=image)))
        if video is not None:
            parts.append(self._one(EmbedInput(modality="video", data=video)))
        return self._norm([sum(xs) for xs in zip(*parts)] if parts else self._text(""))


def _png(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


def _pdf(*colors: tuple[int, int, int]) -> bytes:
    pages = [Image.new("RGB", (200, 280), c) for c in colors]
    out = io.BytesIO()
    pages[0].save(out, format="PDF", save_all=True, append_images=pages[1:])
    return out.getvalue()


def _text_pdf(text: str) -> bytes:
    """A one-page PDF with a real text layer (PIL-made PDFs are images only)."""
    stream = f"BT /F1 24 Tf 40 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>stream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    out.writelines(b"%010d 00000 n \n" % o for o in offsets)
    out.write(b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref))
    return out.getvalue()


def _tone(seconds: int, *, video: bool) -> bytes:
    with __import__("tempfile").TemporaryDirectory() as tmp:
        out = f"{tmp}/clip.{'mp4' if video else 'm4a'}"
        args = ["ffmpeg", "-loglevel", "error", "-y"]
        if video:
            args += ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=5"]
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
        if video:
            args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        subprocess.run([*args, "-c:a", "aac", "-shortest", out], check=True)
        return open(out, "rb").read()


# --- chunking (no database) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content_type", "filename", "kind"),
    [
        ("application/octet-stream", "manual.PDF", "pdf"),
        ("application/octet-stream", "photo.jpeg", "image"),
        ("audio/mpeg", None, "audio"),
        ("application/octet-stream", "talk.mov", "video"),
        ("text/plain", "notes", "text"),
        ("application/octet-stream", "spec.docx", "text"),
        ("application/json", None, "text"),
        ("application/zip", "archive.zip", None),
    ],
)
def test_classify_uses_extension_then_content_type(content_type, filename, kind):
    assert media.classify(content_type, filename) == kind


def test_windows_cover_the_media_with_overlap_and_fold_a_short_tail():
    spans = lambda *args: [(w.start, w.end) for w in media.windows(*args)]  # noqa: E731
    assert spans(0, 60, 5) == []
    assert spans(42, 60, 5) == [(0.0, 42)]
    assert spans(130, 60, 5) == [(0.0, 60), (55.0, 115.0), (110.0, 130)]
    # A 5s tail after the last full window is folded into it rather than standing alone.
    assert spans(120, 60, 5)[-1][1] == 120
    assert all(end - start >= 20 for start, end in spans(121, 60, 5))


def test_pdf_becomes_one_page_image_per_passage():
    drafts = list(media.split_pdf(_pdf((255, 0, 0), (0, 0, 255)), page_pixels=256, max_pages=10))
    assert [d.page for d in drafts] == [1, 2]
    assert all(d.modality == "image" and d.mime == "image/png" for d in drafts)
    assert max(Image.open(io.BytesIO(drafts[0].data or b"")).size) == 256


def test_pdf_page_keeps_its_text_layer_for_display():
    (page,) = media.split_pdf(_text_pdf("Invoices are emailed monthly"), page_pixels=128, max_pages=5)
    assert page.text == "Invoices are emailed monthly" and page.modality == "image"


def test_pdf_over_the_page_limit_is_refused():
    with pytest.raises(media.MediaError, match="limit is 1"):
        list(media.split_pdf(_pdf((1, 1, 1), (2, 2, 2)), page_pixels=64, max_pages=1))


def test_image_is_kept_as_is_or_converted_when_an_api_would_not_take_it():
    png = _png((0, 255, 0), size=(300, 300))
    assert media.split_image(png)[0].data == png
    bmp = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 9, 9)).save(bmp, format="BMP")
    converted = media.split_image(bmp.getvalue())[0]
    assert converted.mime == "image/png" and converted.data != bmp.getvalue()
    with pytest.raises(media.MediaError):
        media.split_image(b"not an image")


def test_text_is_split_into_token_sized_passages():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(10))
    drafts = media.split_text(text, tokens=128)
    assert len(drafts) > 3 and all(d.modality == "text" and d.text for d in drafts)


def test_passages_follow_structure_and_keep_exact_order():
    from hindsight_api.knowledge.text_passages import split_into_passages

    para = "The tenant pays rent monthly. Late payment accrues interest. " * 3
    doc = (
        f"# Lease\n\n{para}\n\nARTICLE II\n\n{para}\n\n"
        "5. TERMINATION. Either party may end the lease with notice.\n\n" + "Long clause sentence here. " * 60
    )
    chunks = split_into_passages(doc, passage_size=64)
    # Every passage is an exact slice of the input, in order.
    cursor = 0
    for c in chunks:
        cursor = doc.index(c, cursor) + len(c)
    assert all(count_tokens(c) <= 64 for c in chunks)
    # Headings open passages: nothing merges across "ARTICLE II" or the numbered "5. TERMINATION".
    assert any(c.startswith("ARTICLE II") for c in chunks)
    assert any(c.startswith("5. TERMINATION") for c in chunks)


@needs_ffmpeg
def test_audio_and_video_are_cut_into_timed_windows():
    audio = list(
        media.split_media(
            _tone(70, video=False),
            "audio",
            filename="a.m4a",
            window_seconds=30,
            overlap_seconds=5,
            max_seconds=3600,
        )
    )
    assert [(d.start_ms, d.modality) for d in audio] == [(0, "audio"), (25000, "audio"), (50000, "audio")]
    assert audio[-1].end_ms == pytest.approx(70000, abs=200)
    video = list(
        media.split_media(
            _tone(8, video=True), "video", filename="v.mp4", window_seconds=30, overlap_seconds=5, max_seconds=3600
        )
    )
    assert len(video) == 1 and video[0].mime == "video/mp4"
    assert media.decode_video_frames(video[0].data or b"").shape[1:] == (448, 448, 3)
    assert media.decode_audio_pcm(video[0].data or b"").size > 16000 * 7
    with pytest.raises(media.MediaError, match="limit is 5s"):
        list(media.split_media(_tone(8, video=False), "audio", filename="a.m4a", window_seconds=30, overlap_seconds=5, max_seconds=5))  # fmt: skip


# --- HTTP ---------------------------------------------------------------------------------------


@pytest.fixture
def fake_embedder(memory):
    service = get_knowledge_service(memory)
    previous = service._embedder
    service._embedder = FakeEmbedder()
    yield service._embedder
    service._embedder = previous


@pytest.fixture
async def kb(api_client, fake_embedder):
    bank_id = f"kb-{uuid.uuid4().hex[:8]}"
    r = await api_client.post(KB, json={"bank_id": bank_id})
    assert r.status_code == 200, r.text
    yield bank_id
    await api_client.delete(f"{KB}/{bank_id}")


async def _upload(client, bank_id: str, files: list[tuple[str, bytes, str]], documents: list[dict] | None = None):
    form = {"request": json.dumps({"files": documents})} if documents else None
    return await client.post(
        f"{KB}/{bank_id}/files/ingest", files=[("files", (n, d, t)) for n, d, t in files], data=form
    )


async def _search(client, bank_id: str, **body) -> list[dict]:
    r = await client.post(f"{KB}/{bank_id}/search", json=body)
    assert r.status_code == 200, r.text
    return r.json()["results"]


async def test_bank_lifecycle_and_separation_from_memory_banks(api_client, memory, kb, request_context):
    banks = (await api_client.get(KB, params={"q": kb})).json()["banks"]
    assert kb in [b["bank_id"] for b in banks]
    memory_banks = (await api_client.get("/v1/default/banks")).json()["banks"]
    assert kb not in [b["bank_id"] for b in memory_banks]

    # A memory write cannot land in a knowledge bank, and a memory bank cannot become one.
    r = await api_client.post(f"/v1/default/banks/{kb}/memories", json={"items": [{"content": "hello"}]})
    assert r.status_code == 400 and "knowledge bank" in r.text
    mem_bank = f"mem-{uuid.uuid4().hex[:8]}"
    await memory._ensure_bank_exists(mem_bank, request_context)
    assert (await api_client.post(KB, json={"bank_id": mem_bank})).status_code == 409
    assert (await api_client.get(f"{KB}/{mem_bank}")).status_code == 404

    assert (await api_client.delete(f"{KB}/{kb}")).status_code == 200
    assert (await api_client.get(f"{KB}/{kb}")).status_code == 404


_LONG_TEXT = "\n\n".join(f"Section {n}. " + ("Unrelated filler words. " * 20) for n in range(8))


async def _passage_count(client, bank_id: str, document_id: str) -> int:
    detail = (await client.get(f"{KB}/{bank_id}/documents/{document_id}")).json()
    assert detail["status"] == "ready", detail
    return len(detail["passages"])


async def test_a_bank_config_starts_on_the_server_defaults_and_shows_no_system_settings(api_client, kb):
    from hindsight_api.config import get_config

    r = await api_client.get(f"{KB}/{kb}/config")
    assert r.status_code == 200, r.text
    c = get_config()
    assert r.json() == {
        "bank_id": kb,
        "config": {
            "text_passage_tokens": c.knowledge_text_passage_tokens,
            "pdf_page_pixels": c.knowledge_pdf_page_pixels,
            "pdf_bands": c.knowledge_pdf_bands,
            "image_tokens": c.knowledge_image_tokens,
            "audio_window_seconds": c.knowledge_audio_window_seconds,
            "video_window_seconds": c.knowledge_video_window_seconds,
            "media_overlap_seconds": c.knowledge_media_overlap_seconds,
            "max_media_seconds": c.knowledge_max_media_seconds,
            "max_pdf_pages": c.knowledge_max_pdf_pages,
            "disabled_media_types": sorted(c.knowledge_disabled_media_types),
        },
        "overrides": {},
    }
    # The embedder and the server's limits are not anybody's to read: the route that told them is gone.
    assert (await api_client.get(f"{KB}/config")).status_code == 404


async def test_a_bank_override_is_applied_at_ingest_and_a_reset_restores_the_default(api_client, kb):
    await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "m", "text": _LONG_TEXT}]})
    default = await _passage_count(api_client, kb, "m")

    r = await api_client.patch(f"{KB}/{kb}/config", json={"updates": {"text_passage_tokens": 32}})
    assert r.status_code == 200, r.text
    assert r.json()["config"]["text_passage_tokens"] == 32 and r.json()["overrides"] == {"text_passage_tokens": 32}
    assert (await api_client.get(f"{KB}/{kb}/config")).json()["overrides"] == {"text_passage_tokens": 32}
    # The same text again is not "unchanged": the bank's settings are part of its content hash.
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "m", "text": _LONG_TEXT}]})
    assert r.json()["unchanged_document_ids"] == []
    assert await _passage_count(api_client, kb, "m") > default

    # Another override merges with the first; null clears one.
    r = await api_client.patch(f"{KB}/{kb}/config", json={"updates": {"pdf_bands": True}})
    assert r.json()["overrides"] == {"text_passage_tokens": 32, "pdf_bands": True}
    r = await api_client.patch(f"{KB}/{kb}/config", json={"updates": {"pdf_bands": None}})
    assert r.json()["overrides"] == {"text_passage_tokens": 32}

    r = await api_client.delete(f"{KB}/{kb}/config")
    assert r.status_code == 200 and r.json()["overrides"] == {}
    assert r.json() == (await api_client.get(f"{KB}/{kb}/config")).json()
    await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "m", "text": _LONG_TEXT}]})
    assert await _passage_count(api_client, kb, "m") == default


async def test_a_bank_override_reaches_the_media_splitters_and_the_embedder(api_client, kb, fake_embedder):
    r = await api_client.patch(
        f"{KB}/{kb}/config",
        json={"updates": {"max_pdf_pages": 1, "image_tokens": 560, "disabled_media_types": ["audio"]}},
    )
    assert r.status_code == 200, r.text
    r = await _upload(
        api_client, kb, [("two.pdf", _pdf((250, 0, 0), (0, 0, 250)), "application/pdf")], [{"document_id": "p"}]
    )
    detail = (await api_client.get(f"{KB}/{kb}/documents/p")).json()
    assert detail["status"] == "failed" and "the limit is 1" in detail["error"], detail
    fake_embedder.calls.clear()
    r = await _upload(api_client, kb, [("red.png", _png((255, 0, 0)), "image/png")], [{"document_id": "red"}])
    assert r.is_success, r.text
    assert [i.image_tokens for call in fake_embedder.calls for i in call if i.modality == "image"] == [560]
    r = await _upload(api_client, kb, [("a.m4a", b"not really audio", "audio/mp4")])
    assert r.status_code == 415 and "this bank does not ingest" in r.text


async def test_a_bank_config_refuses_what_a_bank_may_not_set(api_client, kb):
    from hindsight_api.config import get_config

    async def refused(updates: dict) -> str:
        r = await api_client.patch(f"{KB}/{kb}/config", json={"updates": updates})
        assert r.status_code == 400, r.text
        return r.json()["detail"]

    # Models, providers, keys and the server's limits are the server's: named back, nothing written.
    detail = await refused({"embeddings_model": "x", "max_file_mb": 1, "text_passage_tokens": 64})
    assert "embeddings_model" in detail and "max_file_mb" in detail
    assert "knowledge_text_passage_tokens" in await refused({"knowledge_text_passage_tokens": 64})
    assert "text_passage_tokens" in await refused({"text_passage_tokens": 0})
    assert "text_passage_tokens" in await refused({"text_passage_tokens": "64"})
    assert "image_tokens" in await refused({"image_tokens": 5000})
    assert "disabled_media_types" in await refused({"disabled_media_types": ["pictures"]})
    assert "media_overlap_seconds" in await refused({"media_overlap_seconds": 30, "video_window_seconds": 30})
    over = get_config().knowledge_max_pdf_pages + 1
    assert "server's limit" in await refused({"max_pdf_pages": over})
    assert (await api_client.get(f"{KB}/{kb}/config")).json()["overrides"] == {}


async def test_a_bank_config_answers_only_for_its_own_kind_of_bank(api_client, memory, kb, request_context):
    mem_bank = f"mem-{uuid.uuid4().hex[:8]}"
    await memory._ensure_bank_exists(mem_bank, request_context)
    for bank_id in (mem_bank, f"missing-{uuid.uuid4().hex[:8]}"):
        assert (await api_client.get(f"{KB}/{bank_id}/config")).status_code == 404
        r = await api_client.patch(f"{KB}/{bank_id}/config", json={"updates": {"text_passage_tokens": 64}})
        assert r.status_code == 404
        assert (await api_client.delete(f"{KB}/{bank_id}/config")).status_code == 404
    # And the memory banks' config route does not read, write or wipe a knowledge bank's settings.
    await api_client.patch(f"{KB}/{kb}/config", json={"updates": {"text_passage_tokens": 64}})
    assert (await api_client.get(f"/v1/default/banks/{kb}/config")).status_code == 400
    r = await api_client.patch(f"/v1/default/banks/{kb}/config", json={"updates": {"retain_chunk_size": 1000}})
    assert r.status_code == 400 and "knowledge bank" in r.text
    assert (await api_client.delete(f"/v1/default/banks/{kb}/config")).status_code == 400
    assert (await api_client.get(f"{KB}/{kb}/config")).json()["overrides"] == {"text_passage_tokens": 64}


async def test_files_become_passages_that_search_finds_by_meaning(api_client, kb, fake_embedder):
    r = await _upload(
        api_client,
        kb,
        [
            ("manual.pdf", _pdf((250, 10, 10), (10, 10, 250)), "application/pdf"),
            ("photo.png", _png((10, 240, 10)), "image/png"),
        ],
        documents=[{"document_id": "manual", "tags": ["v2"]}, {"document_id": "photo"}],
    )
    assert r.status_code == 202, r.text
    assert r.json()["document_ids"] == ["manual", "photo"]
    r = await api_client.post(
        f"{KB}/{kb}/ingest",
        json={
            "items": [{"document_id": "faq", "text": "To rotate a key open settings. key key", "filename": "faq.md"}]
        },
    )
    assert r.status_code == 200, r.text

    detail = (await api_client.get(f"{KB}/{kb}/documents/manual")).json()
    assert detail["status"] == "ready" and detail["media_type"] == "document" and detail["passage_count"] == 2
    assert [p["page"] for p in detail["passages"]] == [1, 2]

    top = (await _search(api_client, kb, query="red"))[0]
    assert (top["document_id"], top["page"], top["media_type"]) == ("manual", 1, "document")
    assert (await _search(api_client, kb, query="blue"))[0]["page"] == 2
    assert (await _search(api_client, kb, query="green"))[0]["document_id"] == "photo"
    assert (await _search(api_client, kb, query="how do I rotate a key"))[0]["document_id"] == "faq"
    # An image is a query too.
    import base64

    by_image = await _search(api_client, kb, query_image=base64.b64encode(_png((0, 0, 255))).decode())
    assert (by_image[0]["document_id"], by_image[0]["page"]) == ("manual", 2)

    # The page image and the original file are both downloadable from the URLs search hands back.
    media_resp = await api_client.get(top["passage_url"])
    assert media_resp.status_code == 200 and media_resp.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(media_resp.content)).size[0] > 0
    original = await api_client.get(top["document_url"])
    assert original.status_code == 200 and original.content.startswith(b"%PDF")

    bank = (await api_client.get(f"{KB}/{kb}")).json()
    assert (bank["document_count"], bank["passage_count"], bank["ready_count"]) == (3, 4, 3)
    assert bank["documents_by_media_type"] == {"document": 1, "image": 1, "text": 1}


async def test_tag_and_media_type_filters_only_keep_what_the_caller_asked_for(api_client, kb):
    await _upload(
        api_client,
        kb,
        [("a.png", _png((255, 0, 0)), "image/png"), ("b.png", _png((250, 5, 5)), "image/png")],
        documents=[
            {"document_id": "a", "tags": ["cloud", "v2"]},
            {"document_id": "b", "tags": ["oss"]},
        ],
    )
    ids = lambda results: sorted(r["document_id"] for r in results)  # noqa: E731
    assert ids(await _search(api_client, kb, query="red", tags=["oss"])) == ["b"]
    assert ids(await _search(api_client, kb, query="red", tags=["cloud", "oss"])) == ["a", "b"]
    assert ids(await _search(api_client, kb, query="red", tags=["cloud", "v2"], tags_match="all")) == ["a"]
    assert ids(await _search(api_client, kb, query="red", document_ids=["a"])) == ["a"]
    assert await _search(api_client, kb, query="red", media_types=["audio"]) == []
    bad = await api_client.post(f"{KB}/{kb}/search", json={"query": "red", "media_types": ["hologram"]})
    assert bad.status_code == 422


async def test_rewriting_a_document_is_a_noop_when_unchanged_and_a_replace_otherwise(api_client, kb, fake_embedder):
    first = await _upload(api_client, kb, [("p.png", _png((255, 0, 0)), "image/png")], [{"document_id": "p"}])
    assert len(first.json()["operation_ids"]) == 1
    calls = len(fake_embedder.calls)
    again = await _upload(api_client, kb, [("p.png", _png((255, 0, 0)), "image/png")], [{"document_id": "p"}])
    assert again.json()["operation_ids"] == [] and again.json()["unchanged_document_ids"] == ["p"]
    assert len(fake_embedder.calls) == calls

    await _upload(
        api_client, kb, [("p.pdf", _pdf((0, 0, 255), (0, 0, 250)), "application/pdf")], [{"document_id": "p"}]
    )
    detail = (await api_client.get(f"{KB}/{kb}/documents/p")).json()
    assert (detail["media_type"], detail["passage_count"]) == ("document", 2)
    assert (await _search(api_client, kb, query="blue"))[0]["document_id"] == "p"

    assert (await api_client.delete(f"{KB}/{kb}/documents/p")).status_code == 200
    assert (await api_client.get(f"{KB}/{kb}/documents/p")).status_code == 404
    assert await _search(api_client, kb, query="blue") == []


async def test_bad_writes_are_refused_or_reported_never_silently_dropped(api_client, kb, fake_embedder):
    r = await _upload(api_client, kb, [("x.zip", b"PK..", "application/zip")])
    assert r.status_code == 415
    # A file that cannot be split is accepted, then reported as failed with the reason.
    r = await _upload(
        api_client, kb, [("broken.pdf", b"%PDF-1.4 garbage", "application/pdf")], [{"document_id": "bad"}]
    )
    assert r.status_code == 202
    doc = (await api_client.get(f"{KB}/{kb}/documents/bad")).json()
    assert doc["status"] == "failed" and "PDF" in doc["error"]
    failed = (await api_client.get(f"{KB}/{kb}/documents", params={"status": "failed"})).json()
    assert [d["document_id"] for d in failed["documents"]] == ["bad"]

    # A text-only embedder refuses non-text writes up front.
    fake_embedder.modalities = frozenset({"text"})
    r = await _upload(api_client, kb, [("p.png", _png((1, 2, 3)), "image/png")])
    assert r.status_code == 415 and "only embeds text" in r.text
    assert (await api_client.post(f"{KB}/{kb}/search", json={})).status_code == 400


@needs_ffmpeg
async def test_audio_and_video_passages_carry_their_time_ranges_and_clips(api_client, kb):
    await _upload(
        api_client,
        kb,
        [("talk.m4a", _tone(130, video=False), "audio/mp4"), ("demo.mp4", _tone(6, video=True), "video/mp4")],
        [{"document_id": "talk"}, {"document_id": "demo"}],
    )
    talk = (await api_client.get(f"{KB}/{kb}/documents/talk")).json()
    assert talk["status"] == "ready", talk["error"]
    # 60s windows with 5s overlap (the defaults).
    assert [p["start_ms"] for p in talk["passages"]] == [0, 55000, 110000]
    assert talk["passages"][-1]["end_ms"] == pytest.approx(130000, abs=200)
    hit = (await _search(api_client, kb, query="video"))[0]
    assert (hit["document_id"], hit["media_type"], hit["start_ms"]) == ("demo", "video", 0)
    clip = await api_client.get(hit["passage_url"])
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4" and len(clip.content) > 1000
    # A clip is a query too, alone or with text and an image: all of them make one vector.
    import base64

    clip64 = base64.b64encode(_tone(3, video=True)).decode()
    assert (await _search(api_client, kb, query_video=clip64))[0]["document_id"] == "demo"
    both = await _search(
        api_client, kb, query="video", query_image=base64.b64encode(_png((255, 0, 0))).decode(), query_video=clip64
    )
    assert both[0]["document_id"] == "demo"
    bad = await api_client.post(f"{KB}/{kb}/search", json={"query_video": "not base64!"})
    assert bad.status_code == 400 and "query_video" in bad.text


async def test_reflect_recall_tool_returns_linked_documents_as_the_authority(api_client, memory, kb, request_context):
    from hindsight_api.engine.reflect.tools import tool_recall

    await _upload(api_client, kb, [("keys.png", _png((255, 0, 0)), "image/png")], [{"document_id": "keys"}])
    mem_bank = f"mem-{uuid.uuid4().hex[:8]}"
    await api_client.post(
        f"/v1/default/banks/{mem_bank}/memories", json={"items": [{"content": "The user loves red cars."}]}
    )
    plain = await tool_recall(memory, mem_bank, "red", request_context, include_documents=True)
    assert "documents" not in plain

    r = await api_client.patch(f"/v1/default/banks/{mem_bank}/config", json={"updates": {"knowledge_banks": [kb]}})
    assert r.status_code == 200, r.text
    linked = await tool_recall(memory, mem_bank, "red", request_context, include_documents=True)
    assert [d["document_id"] for d in linked["documents"]] == ["keys"]
    assert linked["documents_note"] and linked["memories"]

    # The public recall and internal recalls (consolidation, mental models) never read documents.
    recall = (await api_client.post(f"/v1/default/banks/{mem_bank}/memories/recall", json={"query": "red"})).json()
    assert "documents" not in recall and recall["results"]
    assert "documents" not in await tool_recall(memory, mem_bank, "red", request_context)
    internal = await memory.recall_async(mem_bank, "red", request_context=request_context)
    assert internal.documents is None
    await memory.delete_bank(mem_bank, request_context=request_context)


async def test_bank_list_pages_and_searches_by_id(api_client, fake_embedder):
    tag = uuid.uuid4().hex[:8]
    ids = [f"kb-{tag}-{n}" for n in range(3)]
    for bank_id in ids:
        assert (await api_client.post(KB, json={"bank_id": bank_id})).status_code == 200
    try:
        first = (await api_client.get(KB, params={"q": tag.upper(), "limit": 2})).json()
        assert first["total"] == 3 and first["limit"] == 2 and first["offset"] == 0
        second = (await api_client.get(KB, params={"q": tag, "limit": 2, "offset": 2})).json()
        # Newest first, and the two pages together are every match exactly once.
        assert [b["bank_id"] for b in first["banks"] + second["banks"]] == ids[::-1]
        past = (await api_client.get(KB, params={"q": tag, "offset": 10})).json()
        assert past["banks"] == [] and past["total"] == 3
    finally:
        for bank_id in ids:
            await api_client.delete(f"{KB}/{bank_id}")


async def test_knowledge_banks_are_hidden_when_the_flag_is_off(api_client, memory, kb, monkeypatch, request_context):
    from hindsight_api.config import clear_config_cache

    mem_bank = f"mem-{uuid.uuid4().hex[:8]}"
    await api_client.post(f"/v1/default/banks/{mem_bank}/memories", json={"items": [{"content": "Red cars."}]})
    await api_client.patch(f"/v1/default/banks/{mem_bank}/config", json={"updates": {"knowledge_banks": [kb]}})

    monkeypatch.setenv("HINDSIGHT_API_ENABLE_KNOWLEDGE_BANKS", "false")
    clear_config_cache()
    assert (await api_client.get("/v1/version")).json()["features"]["knowledge_banks"] is False
    assert (await api_client.get(KB)).status_code == 404
    assert (await api_client.get(f"{KB}/{kb}")).status_code == 404
    assert (await api_client.post(f"{KB}/{kb}/search", json={"query": "red"})).status_code == 404
    from hindsight_api.engine.reflect.tools import tool_recall

    assert "documents" not in await tool_recall(memory, mem_bank, "red", request_context, include_documents=True)

    monkeypatch.setenv("HINDSIGHT_API_ENABLE_KNOWLEDGE_BANKS", "true")
    clear_config_cache()
    assert (await api_client.get("/v1/version")).json()["features"]["knowledge_banks"] is True
    assert (await api_client.get(f"{KB}/{kb}")).status_code == 200
    await api_client.delete(f"/v1/default/banks/{mem_bank}")


async def _scored(memory, bank_id: str, query: str) -> list:
    """The ranked hits with their scores: the search response carries ranks only."""
    from hindsight_api.knowledge.service import SearchRequest
    from hindsight_api.knowledge.store import SearchFilters

    request = SearchRequest(query=query, query_image=None, filters=SearchFilters())
    service = get_knowledge_service(memory)
    return await service._search(bank_id, request, await service._embed_query(request))


async def test_pdf_pages_are_scored_from_image_bands_and_text_layer(api_client, memory, kb, monkeypatch):
    from hindsight_api.config import clear_config_cache

    # 200+ characters of words: a real text layer (a near-empty one leans on the image).
    text = " ".join(["invoice"] * 30)
    await _upload(api_client, kb, [("rates.pdf", _text_pdf(text), "application/pdf")])
    doc = (await api_client.get(f"{KB}/{kb}/documents")).json()["documents"][0]
    assert doc["passage_count"] == 1  # one page = one passage, however many vectors it is stored as
    # Text weight 0: the score is the page image (+ bands) alone, which says nothing about invoices to the fake.
    monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_PDF_TEXT_WEIGHT", "0")
    clear_config_cache()
    assert (await _scored(memory, kb, "invoice"))[0].score < 0.5
    # The default weight brings in the page's text layer (whole page and 128-token units).
    monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_PDF_TEXT_WEIGHT", "0.6")
    clear_config_cache()
    top = (await _scored(memory, kb, "invoice"))[0]
    assert top.score > 0.5 and top.passage.text == text and top.passage.page == 1
    assert "score" not in (await _search(api_client, kb, query="invoice"))[0]


def test_page_score_follows_the_recipe():
    from hindsight_api.knowledge.store import page_score

    assert page_score(0.8, {}, 0.6) == 0.8  # anything with one vector scores as that vector
    image = 0.5 * 0.4 + 0.5 * 0.9  # page image averaged with its best band
    assert page_score(0.4, {"band": [0.1, 0.9]}, 0.6) == image
    roles = {"band": [0.9], "page_text": [0.3], "unit": [0.7, 0.5]}
    assert abs(page_score(0.4, roles, 0.6) - (0.4 * image + 0.6 * 0.7)) < 1e-9


async def test_openai_compatible_embedder_sends_text_as_input_and_media_as_messages():
    """The remote provider speaks vLLM's OpenAI embeddings dialect: texts batched, media one per request."""
    from aiohttp import web

    from hindsight_api.knowledge.embedders import OpenAIEmbedder, UnsupportedModalityError
    from tests.aiohttp_stub import stub_server

    seen: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        seen.append(body)
        n = len(body["input"]) if "input" in body else 1
        return web.json_response({"data": [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(n))]})

    async with stub_server(handler) as base_url:
        embedder = OpenAIEmbedder(
            "m", base_url=base_url + "/v1", api_key=None, dimensions=None,
            modalities=frozenset({"text", "image"}), batch_size=2,
        )  # fmt: skip
        vectors = await embedder.embed_documents(
            [
                EmbedInput(modality="text", text="a", title="T"),
                EmbedInput(modality="image", data=_png((1, 2, 3)), mime="image/png"),
                EmbedInput(modality="text", text="b"),
            ]
        )
        assert vectors == [[0.6, 0.8]] * 3 and embedder.dimension == 2
        assert seen[0] == {"model": "m", "input": ["title: T | text: a", "title: none | text: b"]}
        part = seen[1]["messages"][0]["content"][0]
        assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
        with pytest.raises(UnsupportedModalityError):
            await embedder.embed_documents([EmbedInput(modality="audio", data=b"x", mime="audio/mp4")])


async def test_timeseries_counts_documents_and_passages_per_day(api_client, kb):
    await _upload(
        api_client,
        kb,
        [("a.pdf", _pdf((1, 1, 1), (2, 2, 2)), "application/pdf"), ("b.png", _png((3, 3, 3)), "image/png")],
    )
    r = await api_client.get(f"{KB}/{kb}/stats/timeseries", params={"period": "7d"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trunc"] == "day"
    assert sum(b["documents"] for b in body["buckets"]) == 2 and sum(b["passages"] for b in body["buckets"]) == 3
    assert (await api_client.get(f"{KB}/{kb}/stats/timeseries", params={"period": "2y"})).status_code == 422
    # One media type: the PDF alone (a document, its two pages), then the image alone.
    for media_type, documents, passages in (("document", 1, 2), ("image", 1, 1)):
        r = await api_client.get(f"{KB}/{kb}/stats/timeseries", params={"period": "7d", "media_type": media_type})
        buckets = r.json()["buckets"]
        assert (sum(b["documents"] for b in buckets), sum(b["passages"] for b in buckets)) == (documents, passages)


async def test_an_unreachable_remote_embedder_is_a_503_not_a_crash(api_client, kb, fake_embedder, monkeypatch):
    import aiohttp

    async def down(*args, **kwargs):
        raise aiohttp.ClientConnectionError("connection refused")

    monkeypatch.setattr(fake_embedder, "embed_query", down)
    r = await api_client.post(f"{KB}/{kb}/search", json={"query": "red"})
    assert r.status_code == 503 and "embedder unavailable" in r.text


async def test_text_hits_come_back_with_neighbours_and_overlapping_windows_merge(api_client, kb):
    # Each paragraph fills one passage at the default size (256 tokens), so passage i is paragraph i.
    paragraphs = [f"Paragraph {i}: " + ("invoice " if i in (4, 5) else "filler ") * 180 for i in range(10)]
    r = await api_client.post(
        f"{KB}/{kb}/ingest",
        json={"items": [{"document_id": "long", "text": "\n\n".join(paragraphs), "filename": "long.md"}]},
    )
    assert r.status_code == 200, r.text
    passages = (await api_client.get(f"{KB}/{kb}/documents/long")).json()["passages"]
    weight = {p["index"]: count_tokens(p["text"]) for p in passages}

    # Room for four passages: passage 4 with one on each side, then passage 5 joins that window
    # instead of coming back on its own, and nothing else fits.
    def paragraphs_in(text: str) -> list[int]:
        return [i for i in range(10) if f"Paragraph {i}:" in text]

    hits = await _search(api_client, kb, query="invoice", max_tokens=sum(weight[i] for i in (3, 4, 5, 6)))
    assert len(hits) == 1 and hits[0]["index"] in (4, 5) and "context" not in hits[0]
    assert paragraphs_in(hits[0]["text"]) == [3, 4, 5, 6]  # the hit's text is the run the budget gave it
    # More room widens it, and no passage comes back twice.
    wide = await _search(api_client, kb, query="invoice", max_tokens=sum(weight.values()))
    assert paragraphs_in(wide[0]["text"])[0] < 3 and paragraphs_in(wide[0]["text"])[-1] > 6
    returned = [i for h in wide for i in paragraphs_in(h["text"])]
    assert len(returned) == len(set(returned))
    # Less than one passage: nothing fits.
    assert await _search(api_client, kb, query="invoice", max_tokens=1) == []


def test_pdfs_split_concurrently_from_many_threads_do_not_crash():
    """pdfium is not thread-safe; the worker ingests PDFs on parallel threads (exit 134 before the lock)."""
    from concurrent.futures import ThreadPoolExecutor

    pdf = _pdf(*[(i * 20, 0, 0) for i in range(6)])
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: len(list(media.split_pdf(pdf, page_pixels=256, max_pages=10))), range(16)))
    assert results == [6] * 16


def test_small_or_transparent_images_are_upscaled_onto_a_contrasting_background():
    sprite = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for x in range(8, 24):
        for y in range(8, 24):
            sprite.putpixel((x, y), (250, 250, 250, 255))  # a white sprite: a white background would erase it
    out = io.BytesIO()
    sprite.save(out, format="PNG")
    (draft,) = media.split_image(out.getvalue())
    image = Image.open(io.BytesIO(draft.data or b""))
    assert min(image.size) >= 256 and image.mode == "RGB"
    assert image.getpixel((0, 0)) == (32, 32, 32)  # dark behind a light sprite
    photo = _png((10, 200, 10), size=(512, 400))
    assert media.split_image(photo)[0].data == photo  # big opaque images pass through untouched


async def test_media_is_embedded_with_its_name_and_tags_in_one_input():
    from aiohttp import web

    from hindsight_api.knowledge.embedders import OpenAIEmbedder
    from tests.aiohttp_stub import stub_server

    seen: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(await request.json())
        return web.json_response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async with stub_server(handler) as base_url:
        embedder = OpenAIEmbedder(
            "m", base_url=base_url + "/v1", api_key=None, dimensions=None,
            modalities=frozenset({"text", "audio"}), batch_size=4,
        )  # fmt: skip
        await embedder.embed_documents(
            [
                EmbedInput(
                    modality="audio", data=b"x", mime="audio/mp4", title="footstep_gravel_03", tags=("sfx", "walk")
                )
            ]
        )
        await embedder.embed_documents([EmbedInput(modality="audio", data=b"x", mime="audio/mp4")])
    named, bare = seen[0]["messages"][0]["content"], seen[1]["messages"][0]["content"]
    assert named[0] == {"type": "text", "text": "title: footstep_gravel_03 | tags: sfx, walk | text: "}
    assert named[1]["type"] == "audio_url" and [p["type"] for p in bare] == ["audio_url"]


async def test_remote_embedder_retries_a_dropped_connection():
    from aiohttp import web

    from hindsight_api.knowledge.embedders import OpenAIEmbedder
    from tests.aiohttp_stub import stub_server

    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return web.Response(status=503, text="busy")
        return web.json_response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    async with stub_server(handler) as base_url:
        embedder = OpenAIEmbedder(
            "m", base_url=base_url + "/v1", api_key=None, dimensions=None, modalities=frozenset({"text"}), batch_size=4
        )
        assert await embedder.embed_query("hello") == [1.0, 0.0]
    assert calls["n"] == 2


def test_language_detection_and_keyword_fusion():
    from hindsight_api.knowledge.language import detect, text_search_config
    from hindsight_api.knowledge.store import fuse_keyword

    assert detect("The rotation of the keys is done by the admin in the console") == "english"
    assert detect("La rotation des clés est faite par l'administrateur dans la console") == "french"
    assert detect("ERR-4417") is None
    assert (text_search_config("german"), text_search_config(None)) == ("german", "simple")
    # No keyword hit: dense scores pass through untouched.
    assert fuse_keyword([0.5, 0.4], [0.0, 0.0], 0.2) == [0.5, 0.4]
    # A keyword hit can lift a slightly weaker dense match over the best one.
    fused = fuse_keyword([0.50, 0.49, 0.10], [0.0, 1.0, 0.0], 0.2)
    assert fused[1] > fused[0] > fused[2]


async def test_keyword_matches_count_only_for_documents_in_the_query_language(api_client, kb):
    # The fake embedder sees none of its axis words here, so every passage ties on meaning
    # and only the keyword match can order them.
    docs = [
        ("en-hit", "The gateway returns the code zx4417 when the token of the session has ended."),
        ("en-miss", "The gateway returns a timeout when the upstream of the session has ended."),
        ("fr-hit", "La passerelle renvoie le code zx4417 quand le jeton de la session est fini."),
    ]
    r = await api_client.post(
        f"{KB}/{kb}/ingest",
        json={"items": [{"document_id": d, "text": t} for d, t in docs]},
    )
    assert r.status_code == 200, r.text
    hits = await _search(api_client, kb, query="what does the code zx4417 of the gateway mean")
    # en-miss shares only "gateway"; the French page shares the code but not the query's
    # language, so it gets no keyword credit at all.
    assert [h["document_id"] for h in hits] == ["en-hit", "en-miss", "fr-hit"]


async def test_a_search_needs_a_query_and_a_bank(api_client, kb):
    r = await api_client.post(f"{KB}/{kb}/search", json={})
    assert r.status_code == 400 and "query" in r.text
    assert (await api_client.post(f"{KB}/kb-missing/search", json={"query": "x"})).status_code == 404
    body = (await api_client.post(f"{KB}/{kb}/search", json={"query": "x"})).json()
    assert set(body) == {"results"}  # ranks, no timing and no scores


def test_running_headers_and_footers_are_not_embedded():
    topics = ["Entropy", "Spin", "Lasers", "Relativity", "Chaos"]
    # Page numbers differ, so digits are ignored when lines are compared.
    pages = [f"Physics for everyone - 2024-2025 {i + 1}\n{t} explained\n{i + 1}" for i, t in enumerate(topics)]
    stripped = media.strip_repeated_lines([*pages, None])
    assert stripped[:5] == [f"{t} explained" for t in topics]
    assert stripped[5] is None
    # Two short documents: nothing is frequent enough to call a header.
    assert media.strip_repeated_lines(["Title\nA", "Title\nB"]) == ["Title\nA", "Title\nB"]


def test_the_file_name_is_the_title_unless_it_was_made_up_from_the_id():
    from types import SimpleNamespace

    from hindsight_api.knowledge.service import _title_hint

    def title(document_id: str, filename: str | None) -> str | None:
        return _title_hint(SimpleNamespace(document_id=document_id, filename=filename))  # type: ignore[arg-type]

    assert title("annual-report", "annual-report.pdf") == "annual report"  # named like its id: kept
    assert title("doc-1", "footstep_gravel_03.wav") == "footstep gravel 03"
    assert title("3f2a", "3f2a.md") is None  # the name a text write gets when it has none
    assert title("x", None) is None
    # A name with no word in it says nothing about the content (Flickr30k: R@1 79.8 with it, 93.3 without).
    for meaningless in ("1007129816.jpg", "IMG_4821.JPG", "video7020.mp4", "scan_003.pdf", "deadbeef01.png"):
        assert title("x", meaningless) is None, meaningless
    assert title("x", "face.png") == "face"  # a short word that happens to be hex is still a word


def test_pdf_page_text_is_packed_by_whole_lines():
    lines = [f"Row {i}: revenue grew in region {i}" for i in range(60)]
    units = media.pack_lines("\r\n".join(lines), tokens=40)
    assert all(count_tokens(u) <= 40 for u in units)
    assert [line for u in units for line in u.split("\n")] == lines  # whole lines, in order, none lost
    long = " ".join(["word"] * 200)
    assert all(count_tokens(u) <= 40 for u in media.pack_lines(long, tokens=40))


def test_pdf_sentences_reference_pages_and_text_batches():
    from hindsight_api.knowledge.embedders import _token_batches
    from hindsight_api.knowledge.store import is_reference_page

    text = "Employment rose sharply in 2023 across the EU.\nThe share of tempo-\nrary contracts fell to 12.6% (from 15.9%).\nShort.\nAnother sentence follows with enough words to stand alone."
    assert media.sentences(text) == [
        "Employment rose sharply in 2023 across the EU.",
        "The share of temporary contracts fell to 12.6% (from 15.9%).",  # joined across the line break
        "Short. Another sentence follows with enough words to stand alone.",  # too short alone: merged forward
    ]
    references = "\n".join(f"Author {i}, A. ({2000 + i}), 'A title', Journal, vol. {i}, pp. 1-9." for i in range(8))
    assert is_reference_page(references)
    assert not is_reference_page("The employment rate (2023) rose.\n" + "Prose without citations here.\n" * 6)
    # Batches are cut by size, not count: many short sentences share a request, a long page text does not.
    short = [(i, "s" * 100) for i in range(60)]
    assert [len(b) for b in _token_batches(short, 8)] == [60]
    long = [(i, "p" * 5000) for i in range(3)]
    assert [len(b) for b in _token_batches(long, 8)] == [1, 1, 1]
    assert sum(len(b) for b in _token_batches([(i, "x") for i in range(200)], 8)) == 200  # nothing lost past the cap


def test_a_reference_page_is_demoted_without_squashing_the_other_scores():
    from hindsight_api.knowledge.store import final_scores

    # Two real pages, the better one with fewer keyword hits, and a reference list. Demoting the list inside
    # the keyword fusion stretched the dense range so far that keywords alone decided the real pages.
    dense = [0.80, 0.70, 0.79]
    keyword = [1.0, 3.0, 9.0]
    scores = final_scores(dense, keyword, 0.2, reference=[False, False, True])
    assert scores[0] > scores[1] > scores[2]  # dense still ranks the real pages; the list goes last
    same = final_scores(dense, keyword, 0.2, reference=[False] * 3)
    assert scores[:2] == same[:2]  # the demotion does not move anyone else


def test_pages_lean_on_the_image_when_their_text_is_junk_and_on_their_document():
    from hindsight_api.knowledge.store import page_text_weight, with_document_prior

    prose = "The employment rate of women rose across the member states in the last decade. " * 4
    equations = "∂ψ/∂t = −iħ⁻¹ Hψ ; E = mc² ; λ = h/p ; 𝜓(𝑟,𝑡) = Σ cₙ φₙ e^(−iEₙt/ħ) ; " * 6
    assert page_text_weight(prose, 0.6) == 0.6
    assert page_text_weight(equations, 0.6) == 0.3
    assert page_text_weight("short", 0.6) == 0.3  # almost no text layer
    # A page of the best-matching PDF overtakes a slightly better page of a weaker PDF...
    scores = with_document_prior(["deck", "deck", "other"], [0.80, 0.70, 0.72])
    assert scores[1] > scores[2]
    # ...and a one-passage document keeps its plain similarity.
    assert with_document_prior(["img"], [0.42]) == [pytest.approx(0.42)]


async def test_text_ingest_waits_by_default_and_queues_when_async(api_client, kb):
    items = [{"document_id": "a", "text": "red"}, {"document_id": "b", "text": "blue"}]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.status_code == 200, r.text
    assert r.json() == {"document_ids": ["a", "b"], "operation_ids": [], "unchanged_document_ids": []}
    for d in ("a", "b"):
        assert (await api_client.get(f"{KB}/{kb}/documents/{d}")).json()["status"] == "ready"
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.json()["unchanged_document_ids"] == ["a", "b"]

    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "c", "text": "red"}], "async": True})
    assert r.status_code == 202 and len(r.json()["operation_ids"]) == 1, r.text


async def test_a_text_that_fails_a_sync_ingest_fails_the_request(api_client, kb):
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "e", "text": " \n "}]})
    assert r.status_code == 400 and "nothing to search" in r.json()["detail"], r.text
    assert (await api_client.get(f"{KB}/{kb}/documents/e")).json()["status"] == "failed"


async def test_a_document_waiting_for_a_retry_is_pending_not_failed(api_client, memory, kb, fake_embedder, monkeypatch):
    import aiohttp

    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "red"}], "async": True})
    assert r.status_code == 202, r.text
    service = get_knowledge_service(memory)
    storage_key = (await service.store.get_documents(kb, ["d"]))["d"].storage_key

    async def down(*args, **kwargs):
        raise aiohttp.ClientConnectionError("connection reset")

    monkeypatch.setattr(fake_embedder, "embed_documents", down)
    await service.store.set_statuses(kb, [StatusUpdate("d", "pending")])  # not yet processed
    with pytest.raises(aiohttp.ClientConnectionError):
        await service.process_batch(kb, [_Work("d", storage_key)], retries_left=True)
    doc = (await service.store.get_documents(kb, ["d"]))["d"]
    assert (doc.status, doc.error) == ("pending", "retrying after: ClientConnectionError('connection reset')")
    with pytest.raises(aiohttp.ClientConnectionError):
        await service.process_batch(kb, [_Work("d", storage_key)], retries_left=False)  # the last attempt
    assert (await service.store.get_documents(kb, ["d"]))["d"].status == "failed"


def test_an_abandoned_pdf_split_can_be_closed_while_pdfium_is_in_use():
    # A failed ingest leaves its page generator half-read; closing it (the PDF, under pdfium's lock)
    # from a thread already inside that lock used to deadlock.
    drafts = media.split_pdf(_pdf((255, 0, 0), (0, 0, 255)), page_pixels=256, max_pages=10)
    next(drafts)
    with media._PDFIUM_LOCK:
        drafts.close()


async def test_remote_embedder_keeps_a_few_media_requests_in_flight():
    import asyncio

    from hindsight_api.knowledge import embedders

    embedder = embedders.OpenAIEmbedder(
        "m", base_url="http://unused/v1", api_key=None, dimensions=3, modalities=frozenset({"image"}), batch_size=8
    )
    in_flight = peak = 0

    async def one(item: EmbedInput) -> list[float]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return [float(item.title or 0), 0.0, 0.0]

    embedder._embed_media = one  # type: ignore[method-assign]
    pages = [EmbedInput(modality="image", data=b"x", mime="image/png", title=str(n)) for n in range(20)]
    vectors = await embedder.embed_documents(pages)
    assert [v[0] for v in vectors] == list(range(20))  # every page, in order
    assert peak == embedders._MEDIA_CONCURRENCY


async def test_the_store_reads_and_writes_many_documents_per_call(api_client, memory, kb, fake_embedder):
    from hindsight_api.knowledge.store import KnowledgePassage, PassageReplacement, StatusUpdate

    store = get_knowledge_service(memory).store
    r = await api_client.post(
        f"{KB}/{kb}/ingest", json={"items": [{"document_id": "a", "text": "red"}, {"document_id": "b", "text": "blue"}]}
    )
    assert r.status_code == 200, r.text
    docs = await store.get_documents(kb, ["a", "b", "missing"])
    assert sorted(docs) == ["a", "b"]

    previous = await store.put_documents(
        kb, [replace(docs["a"], storage_key="new-a"), replace(docs["b"], document_id="c")]
    )
    assert previous == {"a": docs["a"].storage_key, "c": None}

    await store.set_statuses(kb, [StatusUpdate("a", "failed", "boom"), StatusUpdate("c", "processing")])
    after = await store.get_documents(kb, ["a", "c"])
    assert (after["a"].status, after["a"].error, after["c"].status) == ("failed", "boom", "processing")

    passages = await store.list_passages(kb, ["a", "b", "c"])
    assert [p.text for p in passages["a"]] == ["red"] and passages["c"] == []
    with_vectors = [
        KnowledgePassage(
            index=n, modality="text", text=f"c{n}", embedding=[1.0] + [0.0] * (fake_embedder.dimension - 1)
        )
        for n in range(4)
    ]
    await store.replace_passages(kb, [PassageReplacement("c", with_vectors), PassageReplacement("a", [])])
    ranges = await store.passage_ranges(kb, [("c", 1, 2), ("c", 3, 9), ("a", 0, 5), ("c", 1, 2)])
    assert [p.text for p in ranges[("c", 1, 2)]] == ["c1", "c2"] and [p.text for p in ranges[("c", 3, 9)]] == ["c3"]
    assert ranges[("a", 0, 5)] == [] and (await store.get_documents(kb, ["c"]))["c"].status == "ready"

    deleted = await store.delete_documents(kb, ["a", "c", "missing"])
    assert sorted(d.document_id for d in deleted) == ["a", "c"]


async def test_an_ingest_naming_one_document_twice_is_refused(api_client, kb):
    items = [{"document_id": "x", "text": "red"}, {"document_id": "x", "text": "blue"}]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.status_code == 400 and "more than once" in r.json()["detail"]


@pytest.fixture
def counted(memory, monkeypatch):
    """Counts the store writes and queued operations an ingest makes, and keeps each task's payload."""
    store = get_knowledge_service(memory).store
    counts = {"put_documents": 0, "replace_passages": 0}
    payloads: list[dict] = []
    for name in counts:
        real = getattr(store, name)

        async def spy(*args, _real=real, _name=name, **kwargs):
            counts[_name] += 1
            return await _real(*args, **kwargs)

        monkeypatch.setattr(store, name, spy)
    # Every queued task reaches the task backend, whichever path wrote its row.
    submit = memory._task_backend.submit_task

    async def submit_spy(payload, *args, **kwargs):
        if payload.get("type") == "knowledge_ingest":
            payloads.append(payload)
        return await submit(payload, *args, **kwargs)

    monkeypatch.setattr(memory._task_backend, "submit_task", submit_spy)
    return counts, payloads


async def test_an_ingest_batch_reports_where_its_time_went(api_client, kb, fake_embedder, caplog):
    items = [{"document_id": f"t{n}", "text": f"note {n} about blue"} for n in range(20)]
    with caplog.at_level("INFO", logger="hindsight_api.engine.retain.timing"):
        r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.is_success, r.text
    lines = [m.getMessage() for m in caplog.records if "[knowledge_ingest-timing]" in m.getMessage()]
    assert len(lines) == 1 and "docs=20" in lines[0], lines
    # Every stage a batch passes through is attributed, with its round-trip count.
    phases = dict(p.split("=", 1) for p in lines[0].split(" | ", 1)[1].split())
    for phase in ("store.read", "store.status", "file.read", "split", "embed", "store.write", "language"):
        assert phase in phases, (phase, lines[0])
    assert phases["file.read"].split("/")[1] == "20c"  # one stored file read per document


async def test_an_ingest_of_many_documents_costs_a_fixed_number_of_calls(api_client, kb, fake_embedder, counted):
    counts, payloads = counted
    items = [{"document_id": f"d{n}", "text": f"note {n} about red"} for n in range(100)]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items, "async": True})
    assert r.status_code == 202, r.text
    # One operation, its id given for every document it carries.
    assert len(set(r.json()["operation_ids"])) == 1 and len(r.json()["operation_ids"]) == 100
    assert len(payloads) == 1 and len(payloads[0]["documents"]) == 100
    page = (await api_client.get(f"{KB}/{kb}/documents", params={"status": "ready", "limit": 1})).json()
    assert page["total"] == 100
    # One row write, one passage write, and embed calls in rounds across documents (100 passages / 16).
    assert counts == {"put_documents": 1, "replace_passages": 1}
    assert len(fake_embedder.calls) == math.ceil(100 / _EMBED_ROUND)


async def test_a_bad_document_fails_alone_and_a_sync_ingest_reports_it(api_client, kb):
    items = [
        {"document_id": "ok1", "text": "red"},
        {"document_id": "bad", "text": " \n "},
        {"document_id": "ok2", "text": "blue"},
    ]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.status_code == 400 and "1 of 3 documents failed; 'bad'" in r.json()["detail"], r.text
    statuses = {d: (await api_client.get(f"{KB}/{kb}/documents/{d}")).json()["status"] for d in ("ok1", "bad", "ok2")}
    assert statuses == {"ok1": "ready", "bad": "failed", "ok2": "ready"}
    assert (await _search(api_client, kb, query="blue"))[0]["document_id"] == "ok2"


@pytest.fixture
def child_bytes(monkeypatch):
    """Sets the async-ingest child budget, and leaves the config as it found it."""

    from hindsight_api.config import clear_config_cache

    def set_budget(n: int) -> None:
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_INGEST_CHILD_BYTES", str(n))
        clear_config_cache()

    yield set_budget
    monkeypatch.delenv("HINDSIGHT_API_KNOWLEDGE_INGEST_CHILD_BYTES", raising=False)
    clear_config_cache()


async def test_an_async_ingest_is_one_parent_split_into_children_by_size(
    api_client, memory, kb, fake_embedder, counted, child_bytes
):
    _, payloads = counted
    child_bytes(100)
    small = [{"document_id": f"s{n}", "text": f"small note {n} " + "x" * 20} for n in range(6)]  # 33 B each
    big = {"document_id": "big", "text": "big " * 100}  # 400 B: over the budget
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [*small[:3], big, *small[3:]], "async": True})
    assert r.status_code == 202, r.text
    # Every document reports the one parent operation.
    parent = r.json()["operation_ids"][0]
    assert set(r.json()["operation_ids"]) == {parent}
    # Children keep the documents in order, never split one, and a document over the budget is alone.
    assert [[d["document_id"] for d in p["documents"]] for p in payloads] == [
        ["s0", "s1", "s2"],
        ["big"],
        ["s3", "s4", "s5"],
    ]
    assert {p["operation_id"] for p in payloads}.isdisjoint({parent})
    page = (await api_client.get(f"{KB}/{kb}/documents", params={"status": "ready", "limit": 1})).json()
    assert page["total"] == 7
    # The parent completes once its children have, as a batch retain's does.
    from hindsight_api.engine.db_utils import acquire_with_retry
    from hindsight_api.engine.schema import fq_table

    async with acquire_with_retry(await memory._get_backend()) as conn:
        rows = await conn.fetch(
            f"SELECT operation_id, status, result_metadata FROM {fq_table('async_operations')} WHERE bank_id = $1",
            kb,
        )
    by_id = {str(row["operation_id"]): row for row in rows}
    assert by_id[parent]["status"] == "completed"
    for p in payloads:
        meta = by_id[p["operation_id"]]["result_metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        assert meta["parent_operation_id"] == parent and by_id[p["operation_id"]]["status"] == "completed"


async def test_a_sync_ingest_is_still_processed_in_one_batch(api_client, kb, fake_embedder, counted, child_bytes):
    counts, payloads = counted
    child_bytes(10)
    items = [{"document_id": f"n{n}", "text": f"note {n} about green"} for n in range(5)]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.is_success and not any(r.json()["operation_ids"]), r.text
    # Nothing queued: the five documents are one batch, written once, however small the budget.
    assert payloads == [] and counts["replace_passages"] == 1


@pytest.fixture
def memory_budget_mb(monkeypatch):
    """Sets the knowledge-ingest memory budget, and leaves the config as it found it."""
    from hindsight_api.config import clear_config_cache

    def set_budget(n: int) -> None:
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_INGEST_MEMORY_BUDGET_MB", str(n))
        clear_config_cache()

    yield set_budget
    monkeypatch.delenv("HINDSIGHT_API_KNOWLEDGE_INGEST_MEMORY_BUDGET_MB", raising=False)
    clear_config_cache()


def _peak_mb(caplog) -> int:
    line = next(m.getMessage() for m in caplog.records if "holding at most" in m.getMessage())
    return int(line.split("holding at most ")[1].split(" MB")[0])


async def test_finished_documents_commit_in_groups_as_they_finish(api_client, kb, fake_embedder, counted):
    from hindsight_api.knowledge import service as service_module

    counts, _ = counted
    parts = service_module._COMMIT_GROUP_PASSAGES // 10  # passages per document, under the early-commit size
    assert parts < service_module._EARLY_COMMIT_PASSAGES
    text = "\n\n".join(f"# Part {n}\n\npassage {n} about red" for n in range(parts))
    items = [{"document_id": f"g{n}", "text": text + f" {n}"} for n in range(12)]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.is_success, r.text
    # 11 documents fill a group, which commits while the last one is cut; that one commits at the end.
    assert counts["replace_passages"] == 2
    page = (await api_client.get(f"{KB}/{kb}/documents", params={"status": "ready", "limit": 1})).json()
    assert page["total"] == 12


async def test_a_batch_over_its_memory_budget_commits_as_it_goes_and_stays_under_it(
    api_client, kb, fake_embedder, counted, memory_budget_mb, caplog
):
    counts, _ = counted
    memory_budget_mb(1)
    # 60 documents of ~60 KB of text: ~3.6 MB of passages against a 1 MB budget.
    items = [{"document_id": f"m{n}", "text": f"note {n} about red " + "word " * 12_000} for n in range(60)]
    with caplog.at_level("INFO", logger="hindsight_api.knowledge.service"):
        r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.is_success, r.text
    assert counts["replace_passages"] >= 4  # committed in pieces, not once at the end
    assert _peak_mb(caplog) <= 1
    page = (await api_client.get(f"{KB}/{kb}/documents", params={"status": "ready", "limit": 1})).json()
    assert page["total"] == 60


async def test_one_document_over_the_memory_budget_goes_in_staged_pieces(
    api_client, memory, kb, fake_embedder, memory_budget_mb, caplog, monkeypatch
):
    memory_budget_mb(1)
    store = get_knowledge_service(memory).store
    pieces: list[int] = []
    real = store.stage_passages

    async def spy(bank_id, items):
        pieces.extend(len(i.passages) for i in items)
        return await real(bank_id, items)

    monkeypatch.setattr(store, "stage_passages", spy)
    big = {"document_id": "big", "text": "a long report about red " + "word " * 400_000}  # ~2 MB
    with caplog.at_level("INFO", logger="hindsight_api.knowledge.service"):
        r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [big]})
    assert r.is_success, r.text
    # Staged in pieces, and never held whole: the batch stays within its budget.
    assert len(pieces) >= 2
    assert _peak_mb(caplog) <= 1
    doc = (await api_client.get(f"{KB}/{kb}/documents/big")).json()
    passages = (await store.list_passages(kb, ["big"]))["big"]
    assert doc["status"] == "ready" and doc["passage_count"] == len(passages) > sum(pieces)
    assert [p.index for p in passages] == list(range(len(passages)))


async def test_a_failed_round_stops_a_splitter_waiting_for_memory(
    api_client, kb, fake_embedder, memory_budget_mb, monkeypatch
):
    memory_budget_mb(1)
    real = fake_embedder.embed_documents
    calls = {"n": 0}

    async def flaky(items):
        calls["n"] += 1
        if calls["n"] >= 2:  # down from the second round on: nothing more commits, so nothing frees room
            raise RuntimeError("embedder down")
        return await real(items)

    monkeypatch.setattr(fake_embedder, "embed_documents", flaky)
    items = [{"document_id": f"f{n}", "text": f"note {n} about red " + "word " * 12_000} for n in range(40)]
    # The round's failure ends the batch (and reaches the caller); without it the splitter would wait
    # for room nobody frees, and this would time out.
    with pytest.raises(RuntimeError, match="embedder down"):
        await asyncio.wait_for(api_client.post(f"{KB}/{kb}/ingest", json={"items": items}), timeout=60)


async def test_a_retried_batch_skips_the_documents_it_already_committed(api_client, memory, kb, fake_embedder, counted):
    _, payloads = counted
    items = [{"document_id": d, "text": f"{d} red"} for d in ("a", "b", "c")]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items, "async": True})
    assert r.status_code == 202, r.text
    service = get_knowledge_service(memory)
    assert {d.status for d in (await service.store.get_documents(kb, ["a", "b", "c"])).values()} == {"ready"}
    # The worker retries the task after "c" was left behind (it never committed).
    await service.store.set_statuses(kb, [StatusUpdate("c", "pending")])
    before = (await service.store.list_passages(kb, ["a"]))["a"]
    fake_embedder.calls.clear()
    await run_ingest_task(memory, {"bank_id": kb, **payloads[0]})
    assert [len(call) for call in fake_embedder.calls] == [1]  # only "c" embedded again
    assert (await service.store.list_passages(kb, ["a"]))["a"] == before
    assert (await service.store.get_documents(kb, ["c"]))["c"].status == "ready"


async def test_the_memory_api_refuses_to_delete_a_knowledge_bank(api_client, kb):
    r = await api_client.delete(f"/v1/default/banks/{kb}")
    assert r.status_code == 400 and "is a knowledge bank" in r.json()["detail"], r.text
    assert (await api_client.get(f"{KB}/{kb}")).status_code == 200


async def test_a_knowledge_bank_cannot_take_a_memory_banks_alias(api_client, fake_embedder):
    memory_bank, alias = f"mem-{uuid.uuid4().hex[:8]}", f"alias-{uuid.uuid4().hex[:8]}"
    assert (await api_client.put(f"/v1/default/banks/{memory_bank}", json={})).is_success
    r = await api_client.post(f"/v1/default/banks/{memory_bank}/aliases", json={"alias": alias})
    assert r.is_success, r.text
    r = await api_client.post(KB, json={"bank_id": alias})
    assert r.status_code == 409 and "is an alias" in r.json()["detail"], r.text
    await api_client.delete(f"/v1/default/banks/{memory_bank}")


async def test_an_embed_round_refused_for_one_document_fails_only_that_one(api_client, kb, fake_embedder, monkeypatch):
    from hindsight_api.knowledge.embedders import UnsupportedModalityError

    real = fake_embedder.embed_documents

    async def picky(items):
        if any("poison" in (i.text or "") for i in items):
            raise UnsupportedModalityError("this model will not embed poison")
        return await real(items)

    monkeypatch.setattr(fake_embedder, "embed_documents", picky)
    items = [{"document_id": d, "text": f"{d} red"} for d in ("a", "poison", "b")]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items})
    assert r.status_code == 400 and "'poison': this model will not embed poison" in r.json()["detail"], r.text
    statuses = {d: (await api_client.get(f"{KB}/{kb}/documents/{d}")).json()["status"] for d in ("a", "poison", "b")}
    assert statuses == {"a": "ready", "poison": "failed", "b": "ready"}


async def test_writes_for_a_replaced_version_leave_the_newer_one_alone(api_client, memory, kb):
    from hindsight_api.knowledge.store import KnowledgePassage, PassageReplacement

    store = get_knowledge_service(memory).store
    for text in ("first red", "second blue"):
        r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": text}]})
        assert r.status_code == 200, r.text
    newer = (await store.get_documents(kb, ["d"]))["d"]
    # A batch still working on the first version fails it, or commits its passages, too late.
    await store.set_statuses(kb, [StatusUpdate("d", "failed", "old version broke", storage_key="gone")])
    stale = KnowledgePassage(index=0, modality="text", text="first red", embedding=[1.0] + [0.0] * (len(_AXES) - 1))
    replaced = await store.replace_passages(kb, [PassageReplacement("d", [stale], storage_key="gone")])
    assert (replaced.stale_media_keys, replaced.skipped_document_ids) == ([], ["d"])
    after = (await store.get_documents(kb, ["d"]))["d"]
    assert (after.status, after.error, after.storage_key) == ("ready", None, newer.storage_key)
    assert [p.text for p in (await store.list_passages(kb, ["d"]))["d"]] == ["second blue"]


def _passages(start: int, *texts: str):
    from hindsight_api.knowledge.store import KnowledgePassage

    return [
        KnowledgePassage(index=start + n, modality="text", text=t, embedding=FakeEmbedder._norm(FakeEmbedder._text(t)))
        for n, t in enumerate(texts)
    ]


async def test_staged_pieces_stay_out_of_sight_until_their_commit(api_client, memory, kb, fake_embedder):
    from hindsight_api.knowledge.store import PassageReplacement, PassageStage

    store = get_knowledge_service(memory).store
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "old blue"}]})
    assert r.status_code == 200, r.text
    key = (await store.get_documents(kb, ["d"]))["d"].storage_key
    assert (
        await store.stage_passages(kb, [PassageStage("d", _passages(0, "new red one", "new red two"), "en", key)]) == []
    )
    assert await store.stage_passages(kb, [PassageStage("d", _passages(2, "new red three"), "en", key, staged=2)]) == []
    # Until the commit, every read sees the version it replaces, whole.
    assert [p.text for p in (await store.list_passages(kb, ["d"]))["d"]] == ["old blue"]
    assert [h["text"] for h in await _search(api_client, kb, query="red")] in ([], ["old blue"])
    done = await store.replace_passages(
        kb, [PassageReplacement("d", _passages(3, "new red four"), "en", key, staged=3)]
    )
    assert done.skipped_document_ids == []
    texts = [p.text for p in (await store.list_passages(kb, ["d"]))["d"]]
    assert texts == ["new red one", "new red two", "new red three", "new red four"]
    assert (await store.get_documents(kb, ["d"]))["d"].passage_count == 4
    assert "old blue" not in [h["text"] for h in await _search(api_client, kb, query="blue")]


async def test_a_piece_for_a_replaced_file_is_skipped(api_client, memory, kb):
    from hindsight_api.knowledge.store import PassageStage

    store = get_knowledge_service(memory).store
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "old blue"}]})
    assert r.status_code == 200, r.text
    assert await store.stage_passages(kb, [PassageStage("d", _passages(0, "stale red"), "en", "gone")]) == ["d"]
    assert await store.stage_passages(kb, [PassageStage("nope", _passages(0, "x"), "en")]) == ["nope"]


async def test_what_an_ingest_that_never_committed_staged_is_dropped(api_client, memory, kb, fake_embedder):
    from hindsight_api.knowledge.store import PassageReplacement, PassageStage

    store = get_knowledge_service(memory).store
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "old blue"}]})
    assert r.status_code == 200, r.text
    key = (await store.get_documents(kb, ["d"]))["d"].storage_key
    # An ingest staged five passages and died before committing them.
    await store.stage_passages(kb, [PassageStage("d", _passages(0, *(f"lost red {n}" for n in range(5))), "en", key)])
    # A version started over (a staged=0 piece) drops them...
    await store.stage_passages(kb, [PassageStage("d", _passages(0, "kept red"), "en", key)])
    await store.replace_passages(kb, [PassageReplacement("d", [], "en", key, staged=1)])
    assert [p.text for p in (await store.list_passages(kb, ["d"]))["d"]] == ["kept red"]
    # ...and so does a version written in one go, which never names them.
    await store.stage_passages(kb, [PassageStage("d", _passages(0, *(f"lost red {n}" for n in range(5))), "en", key)])
    await store.replace_passages(kb, [PassageReplacement("d", _passages(0, "one blue", "two blue"), "en", key)])
    assert [p.text for p in (await store.list_passages(kb, ["d"]))["d"]] == ["one blue", "two blue"]
    hits = await _search(api_client, kb, query="red")
    assert not any("lost" in h["text"] or "kept" in h["text"] for h in hits), hits


async def test_a_document_that_fails_drops_what_it_staged(api_client, memory, kb, fake_embedder):
    from hindsight_api.knowledge.store import PassageReplacement, PassageStage, StatusUpdate

    store = get_knowledge_service(memory).store
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "old blue"}]})
    assert r.status_code == 200, r.text
    key = (await store.get_documents(kb, ["d"]))["d"].storage_key
    await store.stage_passages(kb, [PassageStage("d", _passages(0, *(f"lost red {n}" for n in range(3))), "en", key)])
    # A failure for another version (an older file) leaves what this one staged.
    await store.set_statuses(kb, [StatusUpdate("d", "failed", "boom", "some-other-key")])
    await store.replace_passages(kb, [PassageReplacement("d", [], "en", key, staged=3)])
    assert len((await store.list_passages(kb, ["d"]))["d"]) == 3
    # Failing for good drops it: a commit naming the staged pieces finds none. The store has no read of
    # staged rows, so a commit that would move them in is how the test sees they are gone.
    await store.stage_passages(kb, [PassageStage("d", _passages(0, *(f"lost red {n}" for n in range(3))), "en", key)])
    await store.set_statuses(kb, [StatusUpdate("d", "failed", "boom", key)])
    assert (await store.get_documents(kb, ["d"]))["d"].status == "failed"
    await store.replace_passages(kb, [PassageReplacement("d", [], "en", key, staged=3)])
    assert (await store.list_passages(kb, ["d"]))["d"] == []


async def test_a_file_replaced_while_its_ingest_reads_it_is_skipped_not_retried(
    api_client, memory, kb, fake_embedder, monkeypatch
):
    from hindsight_api.knowledge import media
    from hindsight_api.knowledge.service import _Work
    from hindsight_api.knowledge.store import StatusUpdate

    service = get_knowledge_service(memory)
    store = service.store
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "old blue"}]})
    assert r.status_code == 200, r.text
    old_key = (await store.get_documents(kb, ["d"]))["d"].storage_key
    await store.set_statuses(kb, [StatusUpdate("d", "pending", None, old_key)])  # as if its task were queued
    real = memory._file_storage.retrieve
    replaced = False

    async def retrieve(key: str) -> bytes:
        # The newer upload lands between the batch's check and its read, deleting the old file.
        nonlocal replaced
        if key == old_key and not replaced:
            replaced = True
            r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "new red"}]})
            assert r.status_code == 200, r.text
            raise FileNotFoundError(key)
        return await real(key)

    monkeypatch.setattr(memory._file_storage, "retrieve", retrieve)
    await service.process_batch(kb, [_Work("d", old_key)])  # no error, so no retry
    doc = (await store.get_documents(kb, ["d"]))["d"]
    assert doc.status == "ready" and doc.storage_key != old_key
    assert [p.text for p in (await store.list_passages(kb, ["d"]))["d"]] == ["new red"]

    # Gone under the current version, the file cannot come back: the document fails, once.
    await store.set_statuses(kb, [StatusUpdate("d", "pending", None, doc.storage_key)])

    async def gone(key: str) -> bytes:
        raise FileNotFoundError(key)

    monkeypatch.setattr(memory._file_storage, "retrieve", gone)
    with pytest.raises(media.MediaError):  # with retries left, a transient error would leave it pending
        await service.process_batch(kb, [_Work("d", doc.storage_key)], retries_left=True)
    failed = (await store.get_documents(kb, ["d"]))["d"]
    assert failed.status == "failed" and doc.storage_key in (failed.error or "")


async def test_renaming_a_file_ingests_it_again(api_client, kb):
    for name in ("a.md", "b.md"):
        r = await api_client.post(
            f"{KB}/{kb}/ingest", json={"items": [{"document_id": "d", "text": "same red", "filename": name}]}
        )
        assert r.status_code == 200 and r.json()["unchanged_document_ids"] == [], r.text
    assert (await api_client.get(f"{KB}/{kb}/documents/d")).json()["filename"] == "b.md"


async def test_neighbour_windows_that_grow_into_each_other_are_merged(api_client, memory, kb):
    from hindsight_api.knowledge.store import SearchHit

    text = "\n\n".join(f"# Part {n}\n\npassage {n} red" for n in range(8))
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "w", "text": text}]})
    assert r.status_code == 200, r.text
    service = get_knowledge_service(memory)
    document = (await service.store.get_documents(kb, ["w"]))["w"]
    passages = (await service.store.list_passages(kb, ["w"]))["w"]
    assert len(passages) >= 6
    # Hits at 0, then 4, then 2: windows 0-1 and 3-5, which passage 2's window 1-3 joins into one.
    hits = [SearchHit(passages[i], document, 1.0 - i / 10) for i in (0, 4, 2)]
    # A budget that holds those six passages and no more: the windows meet, nothing widens past them.
    budget = sum(count_tokens(p.text or "") for p in passages[:6])
    items = await service._within_budget(kb, hits, budget)
    assert [(i.context_first, i.context_last) for i in items] == [(0, 5)]


async def test_folded_windows_keep_the_rank_of_the_best_hit(api_client, memory, kb):
    from hindsight_api.knowledge.store import SearchHit

    text = "\n\n".join(f"# Part {n}\n\npassage {n} red" for n in range(10))
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "f", "text": text}]})
    assert r.status_code == 200, r.text
    service = get_knowledge_service(memory)
    document = (await service.store.get_documents(kb, ["f"]))["f"]
    passages = (await service.store.list_passages(kb, ["f"]))["f"]
    # Hits at 2 (best), 7, then 5: 5 joins 7's window, which then meets 2's. The merged window is the
    # best hit's: it keeps passage 2 as its hit, first in the results.
    hits = [SearchHit(passages[i], document, 1.0 - n / 10) for n, i in enumerate((2, 7, 5))]
    budget = sum(count_tokens(p.text or "") for p in passages[1:9])
    items = await service._within_budget(kb, hits, budget)
    assert [(i.hit.passage.index, i.context_first, i.context_last) for i in items] == [(2, 1, 8)]


async def test_a_document_replaced_during_its_batch_is_not_committed(memory, kb, monkeypatch):
    from hindsight_api.knowledge.service import IngestItem
    from hindsight_api.models import RequestContext

    service = get_knowledge_service(memory)
    media_keys: list[str] = []
    real_store = memory._file_storage.store

    async def store_spy(*, file_data, key, metadata=None):
        if "/kb/media/" in key:
            media_keys.append(key)
        return await real_store(file_data=file_data, key=key, metadata=metadata)

    monkeypatch.setattr(memory._file_storage, "store", store_spy)
    real_replace = service.store.replace_passages

    async def replace_after_a_newer_ingest(bank_id, items):
        # A newer ingest of the same document lands at the last moment: after the batch embedded it,
        # just before its passages are written.
        row = (await service.store.get_documents(kb, ["p"]))["p"]
        await service.store.put_documents(kb, [replace(row, storage_key="newer", status="pending")])
        return await real_replace(bank_id, items)

    monkeypatch.setattr(service.store, "replace_passages", replace_after_a_newer_ingest)
    png = IngestItem(data=_png((255, 0, 0)), filename="p.png", content_type="image/png", document_id="p")
    await service.ingest(kb, [png], RequestContext(), wait=True)

    after = (await service.store.get_documents(kb, ["p"]))["p"]
    assert (after.storage_key, after.status) == ("newer", "pending")  # not marked ready by the older batch
    assert (await service.store.list_passages(kb, ["p"]))["p"] == []
    # Its passage media was stored, then deleted again: nobody owns it.
    assert media_keys and not [k for k in media_keys if await memory._file_storage.exists(k)]


async def test_a_refused_document_stops_being_split(api_client, kb, fake_embedder, monkeypatch):
    from hindsight_api.knowledge import service as service_module
    from hindsight_api.knowledge.embedders import UnsupportedModalityError

    pulled = {"n": 0}
    real_next = service_module._next_draft

    def counting(drafts):
        pulled["n"] += 1
        return real_next(drafts)

    async def refuse(items):
        raise UnsupportedModalityError("refused")

    monkeypatch.setattr(service_module, "_next_draft", counting)
    monkeypatch.setattr(fake_embedder, "embed_documents", refuse)
    rounds = 4 * (service_module._EMBED_ROUNDS_IN_FLIGHT + 1)
    text = "\n\n".join(f"# Part {n}\n\npassage {n}" for n in range(rounds * _EMBED_ROUND))
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "long", "text": text}]})
    assert r.status_code == 400, r.text
    # The splitter runs ahead of the embedder by at most the rounds in flight; once the first round is
    # refused it stops, so most of the document's passages were never cut.
    assert pulled["n"] <= (service_module._EMBED_ROUNDS_IN_FLIGHT + 1) * _EMBED_ROUND + 1
    assert pulled["n"] < rounds * _EMBED_ROUND // 2


async def test_a_new_content_type_ingests_the_same_file_again(api_client, kb):
    first = await _upload(api_client, kb, [("n.txt", b"red notes", "text/plain")], [{"document_id": "n"}])
    assert first.is_success, first.text
    again = await _upload(api_client, kb, [("n.txt", b"red notes", "text/markdown")], [{"document_id": "n"}])
    assert again.json()["unchanged_document_ids"] == []
    assert (await api_client.get(f"{KB}/{kb}/documents/n")).json()["content_type"] == "text/markdown"


async def _run_as_worker(memory, poller, child: dict, retry_count: int) -> dict:
    """Run a queued child operation the way the worker would, on its ``retry_count``-th attempt, and read
    back its operation row. Raw SQL on purpose: the row is marked claimed (as a worker's claim would mark
    it) so the poller's terminal write applies, and no API exposes ``retry_count`` or the defer counts."""
    from hindsight_api.worker.poller import ClaimedTask

    pool = await memory._get_pool()
    await pool.execute(
        "UPDATE async_operations SET status = 'processing', worker_id = 'kb-worker', claimed_at = now() "
        "WHERE operation_id = $1",
        uuid.UUID(child["operation_id"]),
    )
    await poller.execute_task(
        ClaimedTask(operation_id=child["operation_id"], task_dict={**child, "_retry_count": retry_count}, schema=None)
    )
    assert await poller.wait_for_active_tasks(timeout=30.0)
    row = await pool.fetchrow(
        "SELECT status, retry_count, result_metadata FROM async_operations WHERE operation_id = $1",
        uuid.UUID(child["operation_id"]),
    )
    meta = row["result_metadata"]
    return {**dict(row), "result_metadata": json.loads(meta) if isinstance(meta, str) else meta}


@pytest.mark.xdist_group("worker_tests")
async def test_a_rate_limited_embedder_defers_the_batch_instead_of_failing_it(
    api_client, memory, kb, fake_embedder, counted, monkeypatch
):
    from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
    from hindsight_api.worker import WorkerPoller

    _, payloads = counted
    items = [{"document_id": f"r{n}", "text": f"note {n} about red"} for n in range(3)]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items, "async": True})
    assert r.status_code == 202, r.text
    child = payloads[0]
    service = get_knowledge_service(memory)
    # Run the child again as the worker would, its documents back to pending, against an embedder that
    # keeps answering 429 after its own retries.
    await service.store.set_statuses(kb, [StatusUpdate(i["document_id"], "pending") for i in items])

    async def busy(_items):
        raise UpstreamHTTPError(429, "Too Many Requests", {}, "http://gateway/v1/embeddings")

    monkeypatch.setattr(fake_embedder, "embed_documents", busy)
    poller = WorkerPoller(backend=await memory._get_backend(), worker_id="kb-worker", executor=memory.execute_task)

    async def attempt(retry_count: int) -> dict:
        return await _run_as_worker(memory, poller, child, retry_count)

    # On its last retry, too: a rate limit is "not now", not a failure, so nothing is used up or lost.
    row = await attempt(retry_count=3)
    assert row["status"] == "pending" and (row["retry_count"] or 0) == 0, row
    assert row["result_metadata"]["rate_limit_defers"] == 1
    assert {d.status for d in (await service.store.get_documents(kb, [i["document_id"] for i in items])).values()} == {
        "pending"
    }
    assert (await attempt(retry_count=0))["result_metadata"]["rate_limit_defers"] == 2  # counted, and backing off

    # Anything else still uses up a retry, as before.
    async def broken(_items):
        raise UpstreamHTTPError(500, "Internal Server Error", {}, "http://gateway/v1/embeddings")

    monkeypatch.setattr(fake_embedder, "embed_documents", broken)
    row = await attempt(retry_count=0)
    assert row["status"] == "pending" and row["retry_count"] == 1, row


async def test_a_store_shedding_an_upload_answers_503_with_retry_after(api_client, memory, kb, monkeypatch):
    from hindsight_api.engine.memories.base import StoreWriteUnavailable

    store = get_knowledge_service(memory).store

    async def shed(*_a, **_k):
        raise StoreWriteUnavailable("fold behind: the bank's indexing is behind its writers")

    monkeypatch.setattr(store, "put_documents", shed)
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "s", "text": "x"}], "async": True})
    # "Not now", which a bulk ingest retries, rather than a 500 that says the request is broken.
    assert r.status_code == 503 and r.headers["retry-after"] == "30", r.text


@pytest.mark.xdist_group("worker_tests")
async def test_a_store_shedding_a_batch_defers_it_without_using_a_retry(
    api_client, memory, kb, fake_embedder, counted, monkeypatch
):
    from hindsight_api.worker import WorkerPoller

    _, payloads = counted
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "b", "text": "red"}], "async": True})
    assert r.status_code == 202, r.text
    child = payloads[0]
    service = get_knowledge_service(memory)
    await service.store.set_statuses(kb, [StatusUpdate("b", "pending")])

    async def shed(*_a, **_k):
        raise RuntimeError("Write: fold behind: un-folded tail holds 15603736 bytes, over the write bound")

    monkeypatch.setattr(service.store, "replace_passages", shed)
    poller = WorkerPoller(backend=await memory._get_backend(), worker_id="kb-worker", executor=memory.execute_task)
    row = await _run_as_worker(memory, poller, child, retry_count=0)
    # Deferred, not retried: the backlog clears by itself, and a counted retry is how a long ingest's
    # batches ran out of attempts while it did.
    assert row["status"] == "pending" and (row["retry_count"] or 0) == 0, row
    assert (await service.store.get_documents(kb, ["b"]))["b"].status == "pending"


async def test_a_batch_that_fails_for_good_fails_its_documents_not_leaves_them_pending(
    api_client, memory, kb, fake_embedder, monkeypatch
):
    store = get_knowledge_service(memory).store

    async def other_size(*_a, **_k):
        raise ValueError("knowledge bank holds vectors of 768 dimensions, but this ingest makes 384")

    monkeypatch.setattr(store, "replace_passages", other_size)
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"document_id": "v", "text": "red"}], "async": True})
    assert r.status_code == 202, r.text
    # The engine does not retry a ValueError, so a pending document would wait for ever.
    doc = (await api_client.get(f"{KB}/{kb}/documents/v")).json()
    assert doc["status"] == "failed" and "768 dimensions" in (doc["error"] or ""), doc


def test_an_engine_with_a_knowledge_service_can_still_be_freed():
    """The service is kept on its engine, not in a process-wide map: a weak map whose value (the service)
    held its own key (the engine) kept every engine a test made alive until the CI runner ran out of memory."""
    import gc
    import weakref

    class _Engine:
        _knowledge_service = None

    engine = _Engine()
    service = get_knowledge_service(engine)  # type: ignore[arg-type]
    assert get_knowledge_service(engine) is service  # type: ignore[arg-type]  # one per engine
    gone = weakref.ref(engine)
    del engine, service
    gc.collect()
    assert gone() is None


async def test_media_hits_cost_a_fixed_amount_of_the_budget(api_client, kb):
    from hindsight_api.knowledge.service import MEDIA_HIT_TOKENS

    for n, color in enumerate(((255, 0, 0), (0, 0, 255), (0, 255, 0))):
        await _upload(api_client, kb, [(f"swatch{n}.png", _png(color), "image/png")], [{"document_id": f"swatch{n}"}])
    # Images carry no text: each costs MEDIA_HIT_TOKENS, so this budget holds two of the three.
    hits = await _search(api_client, kb, query="red", max_tokens=2 * MEDIA_HIT_TOKENS)
    assert len(hits) == 2
    assert len(await _search(api_client, kb, query="red", max_tokens=3 * MEDIA_HIT_TOKENS)) == 3


async def test_the_search_budget_sets_how_deep_it_ranks(api_client, kb):
    from hindsight_api.knowledge.service import SEARCH_CANDIDATES

    items = [{"document_id": f"n{i}", "text": f"note {i} about red"} for i in range(SEARCH_CANDIDATES["low"] + 10)]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": items[:100]})
    assert r.status_code == 200, r.text
    deep = {"query": "red", "max_tokens": 1_000_000}
    assert len(await _search(api_client, kb, budget="low", **deep)) == SEARCH_CANDIDATES["low"]
    assert len(await _search(api_client, kb, **deep)) == len(items)  # mid, the default, holds them all
    bad = await api_client.post(f"{KB}/{kb}/search", json={"query": "red", "budget": "huge"})
    assert bad.status_code == 422


async def test_a_text_with_a_nul_byte_is_stored_and_spares_its_batch(api_client, kb):
    # Postgres text cannot hold NUL: one such text used to fail every document ingested with it.
    r = await api_client.post(
        f"{KB}/{kb}/ingest",
        json={"items": [{"document_id": "nul", "text": "debt\x00 ratio"}, {"document_id": "ok", "text": "fine"}]},
    )
    assert r.status_code == 200, r.text
    for doc in ("nul", "ok"):
        detail = (await api_client.get(f"{KB}/{kb}/documents/{doc}")).json()
        assert detail["status"] == "ready", detail
    assert (await api_client.get(f"{KB}/{kb}/documents/nul")).json()["passages"][0]["text"] == "debt ratio"
