"""Knowledge-bank embedders: each provider's request and response, against a stub of its API, and
the server settings that decide what a knowledge bank can ingest (modalities, opt-outs, text-only).
"""

from dataclasses import replace

import pytest
from aiohttp import web

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.knowledge.embedders import (
    CohereEmbedder,
    EmbedInput,
    OpenAIEmbedder,
    TextEmbedder,
    UnsupportedModalityError,
    VoyageEmbedder,
    create_knowledge_embedder,
)
from tests.aiohttp_stub import stub_server
from tests.test_knowledge_banks import KB, _png, _text_pdf, _upload, fake_embedder, kb  # noqa: F401  (fixtures)

PNG = _png((255, 0, 0))
MP4 = b"\x00\x00\x00\x18ftypmp42"  # sent as bytes, never decoded by a remote provider


def _recorder(reply):
    """A stub API that records each request's path and body, answering ``reply(body)``."""
    seen: list[tuple[str, dict]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        seen.append((request.path, body))
        return web.json_response(reply(body))

    return handler, seen


# --- Voyage ----------------------------------------------------------------------------------------


def _voyage_reply(body: dict) -> dict:
    return {"data": [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["inputs"]))]}


async def test_voyage_sends_media_with_its_name_and_tags_in_one_input():
    handler, seen = _recorder(_voyage_reply)
    async with stub_server(handler) as base_url:
        embedder = VoyageEmbedder(
            "voyage-multimodal-3.5", api_key="k", base_url=base_url, dimensions=None, batch_size=8
        )
        vectors = await embedder.embed_documents(
            [
                EmbedInput(modality="text", text="Keys expire.", title="api keys"),
                EmbedInput(modality="image", data=PNG, mime="image/png", title="red car", tags=("vehicle",)),
                EmbedInput(modality="video", data=MP4, mime="video/mp4"),
            ]
        )
        query = await embedder.embed_query("red car", PNG)
    assert vectors == [[0.6, 0.8]] * 3 and query == [0.6, 0.8] and embedder.dimension == 2
    path, body = seen[0]
    assert path == "/multimodalembeddings" and body["input_type"] == "document"
    text, image, video = (i["content"] for i in body["inputs"])
    assert text == [{"type": "text", "text": "api keys\n\nKeys expire."}]
    assert image[0] == {"type": "text", "text": "red car | vehicle"}
    assert image[1]["type"] == "image_base64" and image[1]["image_base64"].startswith("data:image/png;base64,")
    assert video[0]["type"] == "video_base64" and video[0]["video_base64"].startswith("data:video/mp4;base64,")
    assert seen[1][1]["input_type"] == "query" and [c["type"] for c in seen[1][1]["inputs"][0]["content"]] == [
        "text",
        "image_base64",
    ]


async def test_voyage_refuses_audio_and_honours_the_dimensions():
    handler, seen = _recorder(_voyage_reply)
    async with stub_server(handler) as base_url:
        embedder = VoyageEmbedder("m", api_key="k", base_url=base_url, dimensions=512, batch_size=8)
        with pytest.raises(UnsupportedModalityError):
            await embedder.embed_documents([EmbedInput(modality="audio", data=b"x", mime="audio/mp4")])
        await embedder.embed_query("q")
    assert seen[0][1]["output_dimension"] == 512 and embedder.modalities == {"text", "image", "video"}


# --- Cohere ----------------------------------------------------------------------------------------


def _cohere_reply(body: dict) -> dict:
    return {"embeddings": {"float": [[0.0, 2.0] for _ in body["inputs"]]}}


async def test_cohere_sends_texts_and_captioned_images_as_inputs():
    handler, seen = _recorder(_cohere_reply)
    async with stub_server(handler) as base_url:
        embedder = CohereEmbedder("embed-v4.0", api_key="k", base_url=base_url, dimensions=None, batch_size=2)
        texts = [EmbedInput(modality="text", text=f"t{n}") for n in range(100)]
        assert len(await embedder.embed_documents(texts)) == 100
        images = [EmbedInput(modality="image", data=PNG, mime="image/png", title="red") for _ in range(3)]
        assert await embedder.embed_documents(images) == [[0.0, 1.0]] * 3
        await embedder.embed_query("red")
        with pytest.raises(UnsupportedModalityError):
            await embedder.embed_documents([EmbedInput(modality="video", data=MP4, mime="video/mp4")])
    # Texts go 96 to a request; images, which make a request heavy, the configured batch size.
    assert [len(body["inputs"]) for _, body in seen[:4]] == [96, 4, 2, 1]
    path, body = seen[2]
    assert path == "/embed" and body["input_type"] == "search_document" and body["embedding_types"] == ["float"]
    assert body["inputs"][0]["content"][0] == {"type": "text", "text": "red"}
    assert body["inputs"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen[4][1]["input_type"] == "search_query"


async def test_a_short_answer_from_a_provider_fails_the_call():
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response({"embeddings": {"float": [[1.0]]}})

    async with stub_server(handler) as base_url:
        embedder = CohereEmbedder("m", api_key="k", base_url=base_url, dimensions=None, batch_size=8)
        with pytest.raises(RuntimeError, match="returned 1 embeddings for 2 inputs"):
            await embedder.embed_documents(
                [EmbedInput(modality="text", text="a"), EmbedInput(modality="text", text="b")]
            )


# --- choosing a provider ---------------------------------------------------------------------------


def _config(**changes) -> HindsightConfig:
    return replace(HindsightConfig.from_env(), **changes)


def test_each_provider_is_built_from_the_knowledge_settings(memory):
    voyage = create_knowledge_embedder(
        _config(knowledge_embeddings_provider="voyage", knowledge_embeddings_api_key="k"), memory.embeddings
    )
    assert isinstance(voyage, VoyageEmbedder) and voyage.model == "voyage-multimodal-3.5"
    cohere = create_knowledge_embedder(
        _config(knowledge_embeddings_provider="cohere", embeddings_cohere_api_key="memory-key"), memory.embeddings
    )
    assert isinstance(cohere, CohereEmbedder) and cohere.model == "embed-v4.0"  # the memory banks' Cohere key
    with pytest.raises(ValueError, match="needs HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_API_KEY"):
        create_knowledge_embedder(
            _config(knowledge_embeddings_provider="voyage", knowledge_embeddings_api_key=None), memory.embeddings
        )
    with pytest.raises(ValueError, match="needs HINDSIGHT_API_EMBEDDINGS_VERTEXAI_PROJECT_ID"):
        create_knowledge_embedder(
            _config(knowledge_embeddings_provider="vertex", embeddings_vertexai_project_id=None), memory.embeddings
        )
    with pytest.raises(ValueError, match="Expected local, gemini, vertex, voyage, cohere, openai or text"):
        create_knowledge_embedder(_config(knowledge_embeddings_provider="nope"), memory.embeddings)


async def test_the_text_provider_reuses_the_memory_banks_embedder_by_default(memory):
    embedder = create_knowledge_embedder(
        _config(knowledge_embeddings_provider="text", knowledge_text_embeddings=None), memory.embeddings
    )
    assert isinstance(embedder, TextEmbedder) and embedder.modalities == {"text"}
    assert embedder.dimension == memory.embeddings.dimension
    assert len(await embedder.embed_query("red")) == memory.embeddings.dimension
    with pytest.raises(UnsupportedModalityError):
        await embedder.embed_documents([EmbedInput(modality="image", data=PNG, mime="image/png")])


async def test_the_text_provider_can_use_any_memory_bank_provider_of_its_own(memory, monkeypatch):
    """HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_* reads every memory-bank embeddings setting again."""
    calls: list[str] = []

    async def tei(request: web.Request) -> web.StreamResponse:
        calls.append(request.path)
        if request.path == "/info":
            return web.json_response({"model_id": "stub-text"})
        inputs = (await request.json())["inputs"]
        return web.json_response([[1.0, 0.0, 0.0] for _ in (inputs if isinstance(inputs, list) else [inputs])])

    async with stub_server(tei) as base_url:
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_PROVIDER", "tei")
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_TEI_URL", base_url)
        config = HindsightConfig.from_env()
        assert config.knowledge_text_embeddings is not None
        assert config.knowledge_text_embeddings.embeddings_provider == "tei"
        embedder = create_knowledge_embedder(replace(config, knowledge_embeddings_provider="text"), memory.embeddings)
        assert isinstance(embedder, TextEmbedder) and embedder.dimension == 0  # started on first use
        vectors = await embedder.embed_documents([EmbedInput(modality="text", text="Keys expire.", title="keys")])
    assert vectors == [[1.0, 0.0, 0.0]] and embedder.dimension == 3 and "/embed" in calls

    monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_NOT_A_SETTING", "x")
    with pytest.raises(ValueError, match="Unknown knowledge_text embeddings variable"):
        HindsightConfig.from_env()


# --- what a server ingests -------------------------------------------------------------------------


@pytest.fixture
def disabled_media(monkeypatch):
    def disable(value: str) -> None:
        monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_DISABLED_MEDIA_TYPES", value)
        clear_config_cache()

    yield disable
    monkeypatch.delenv("HINDSIGHT_API_KNOWLEDGE_DISABLED_MEDIA_TYPES", raising=False)
    clear_config_cache()


async def test_a_disabled_media_type_is_refused_whatever_the_embedder(api_client, kb, disabled_media):  # noqa: F811
    disabled_media("image, VIDEO")
    r = await _upload(api_client, kb, [("p.png", PNG, "image/png")])
    assert r.status_code == 415 and "HINDSIGHT_API_KNOWLEDGE_DISABLED_MEDIA_TYPES" in r.text
    # The bank's settings report what the server refuses, as what the bank refuses too.
    assert (await api_client.get(f"{KB}/{kb}/config")).json()["config"]["disabled_media_types"] == ["image", "video"]
    r = await api_client.post(f"{KB}/{kb}/ingest", json={"items": [{"text": "still fine"}]})
    assert r.is_success, r.text


def test_an_unknown_disabled_media_type_is_a_startup_error(disabled_media):
    with pytest.raises(ValueError, match="unknown media types pictures"):
        disabled_media("pictures")
        HindsightConfig.from_env()


async def test_a_text_only_embedder_ingests_pdfs_from_their_text_layer(api_client, kb, fake_embedder):  # noqa: F811
    fake_embedder.modalities = frozenset({"text"})
    r = await _upload(
        api_client, kb, [("keys.pdf", _text_pdf("Rotate every key"), "application/pdf")], [{"document_id": "p"}]
    )
    assert r.is_success, r.text
    detail = (await api_client.get(f"{KB}/{kb}/documents/p")).json()
    assert detail["status"] == "ready" and detail["passages"][0]["page"] == 1
    # Every vector of the page was a text one: the embedder was never handed an image.
    assert all(i.modality == "text" for call in fake_embedder.calls for i in call)
    r = await _upload(api_client, kb, [("p.png", PNG, "image/png")])
    assert r.status_code == 415 and "only embeds text" in r.text


async def test_remote_embedder_requests_in_flight_are_capped_across_callers(monkeypatch):
    import asyncio

    from hindsight_api.config import clear_config_cache
    from hindsight_api.knowledge import embedders

    monkeypatch.setenv("HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_MAX_IN_FLIGHT", "3")
    clear_config_cache()
    try:
        now = peak = 0

        async def one() -> None:
            nonlocal now, peak
            async with embedders._in_flight():
                now += 1
                peak = max(peak, now)
                await asyncio.sleep(0.01)
                now -= 1

        await asyncio.gather(*(one() for _ in range(20)))
        assert peak == 3
    finally:
        monkeypatch.delenv("HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_MAX_IN_FLIGHT")
        clear_config_cache()


# --- mixed queries -----------------------------------------------------------------------------------


async def test_a_query_with_text_image_and_video_is_one_vector():
    handler, seen = _recorder(lambda body: {"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
    async with stub_server(handler) as base_url:
        openai = OpenAIEmbedder(
            "m",
            base_url=base_url + "/v1",
            api_key=None,
            dimensions=None,
            modalities=frozenset({"text", "image", "video"}),
            batch_size=8,
        )
        assert await openai.embed_query("red car", None, MP4) == [0.6, 0.8]
        # EmbeddingGemma 2 embeds one medium per input: an image and a video go apart, then are averaged.
        assert await openai.embed_query("red car", PNG, MP4) == [0.6, 0.8]
        voyage = VoyageEmbedder("m", api_key="k", base_url=base_url, dimensions=None, batch_size=8)
        await voyage.embed_query("red car", PNG, MP4)
    (video_only,) = (m["content"] for m in seen[0][1]["messages"])
    assert [c["type"] for c in video_only] == ["video_url", "text"]
    assert video_only[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    mixed = sorted(tuple(c["type"] for c in body["messages"][0]["content"]) for _, body in seen[1:3])
    assert mixed == [("image_url", "text"), ("video_url", "text")]
    assert [c["type"] for c in seen[3][1]["inputs"][0]["content"]] == ["text", "image_base64", "video_base64"]


async def test_a_video_query_is_refused_by_a_model_without_video():
    cohere = CohereEmbedder("m", api_key="k", base_url="http://unused", dimensions=None, batch_size=8)
    with pytest.raises(UnsupportedModalityError):
        await cohere.embed_query("q", None, MP4)
    images_only = OpenAIEmbedder(
        "m",
        base_url="http://unused/v1",
        api_key=None,
        dimensions=None,
        modalities=frozenset({"text", "image"}),
        batch_size=8,
    )
    with pytest.raises(UnsupportedModalityError):
        await images_only.embed_query(None, None, MP4)
