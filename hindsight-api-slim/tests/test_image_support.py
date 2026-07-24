from __future__ import annotations

import asyncio
import hashlib
import io
import json
import uuid
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest
from PIL import Image

from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.image import (
    AnalyzedImage,
    AnthropicImageAnalysisProvider,
    GeminiImageAnalysisProvider,
    ImageAnalysisInput,
    ImageAssetInUseError,
    ImageProviderConfig,
    ImageProviderConfigurationError,
    ImageRetainOperationMetadata,
    ImageRetainParameters,
    ImageSemanticAnalysis,
    MockImageAnalysisProvider,
    OpenAIImageAnalysisProvider,
    PendingImageAsset,
    create_image_provider,
    render_image_analysis_markdown,
    resolve_image_provider_configs,
    validate_image,
    validate_image_provider_config,
)
from hindsight_api.engine.schema import fq_table
from hindsight_api.engine.storage.azure import AzureFileStorage
from hindsight_api.engine.storage.base import FileStorage
from hindsight_api.engine.storage.gcs import GCSFileStorage
from hindsight_api.engine.storage import postgresql as postgresql_storage
from hindsight_api.engine.storage.s3 import S3FileStorage
from hindsight_api.engine.transfer.export import export_bank
from hindsight_api.engine.transfer.importer import parse_archive, validate_archive_file
from hindsight_api.engine.transfer.schema import TransferDocument, TransferImageAsset, TransferManifest
from hindsight_api.extensions import (
    BankReadContext,
    BankReadOperation,
    BankWriteContext,
    BankWriteOperation,
    ImageAnalyzeResult,
    ImageRetainContext,
    ImageRetainPhase,
    OperationValidatorExtension,
    PrecheckContext,
    PrecheckOperation,
    RecallContext,
    ReflectContext,
    RetainContext,
    ValidationResult,
)


class _ImageHookRecorder(OperationValidatorExtension):
    def __init__(self, *, reject_precheck: bool = False) -> None:
        self.reject_precheck = reject_precheck
        self.prechecks: list[PrecheckContext] = []
        self.image_retain: list[ImageRetainContext] = []
        self.image_results: list[ImageAnalyzeResult] = []
        self.reads: list[BankReadContext] = []
        self.writes: list[BankWriteContext] = []
        self.recalls: list[RecallContext] = []

    async def precheck(self, ctx: PrecheckContext) -> ValidationResult:
        self.prechecks.append(ctx)
        if self.reject_precheck:
            return ValidationResult.reject("blocked before body", status_code=429)
        return ValidationResult.accept()

    async def validate_image_retain(self, ctx: ImageRetainContext) -> ValidationResult:
        self.image_retain.append(ctx)
        return ValidationResult.accept()

    async def on_image_analyze_complete(self, result: ImageAnalyzeResult) -> None:
        self.image_results.append(result)

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        self.reads.append(ctx)
        return ValidationResult.accept()

    async def validate_bank_write(self, ctx: BankWriteContext) -> ValidationResult:
        self.writes.append(ctx)
        return ValidationResult.accept()

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        self.recalls.append(ctx)
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()


def _image_bytes(fmt: str, *, size: tuple[int, int] = (8, 6), exif_orientation: int | None = None) -> bytes:
    image = Image.new("RGB", size, (32, 96, 160))
    output = io.BytesIO()
    exif = Image.Exif()
    if exif_orientation is not None:
        exif[274] = exif_orientation
    image.save(output, format=fmt, exif=exif)
    return output.getvalue()


@pytest.mark.parametrize(
    ("fmt", "declared", "expected"),
    [
        ("JPEG", "image/jpeg", "image/jpeg"),
        ("PNG", "image/png", "image/png"),
        ("WEBP", "image/webp", "image/webp"),
    ],
)
def test_validate_supported_images_preserves_exact_bytes(fmt: str, declared: str, expected: str) -> None:
    uploaded = _image_bytes(fmt)
    result = validate_image(
        uploaded,
        declared,
        max_size_bytes=1024 * 1024,
    )
    assert result.mime_type == expected
    assert (result.width, result.height) == (8, 6)
    assert len(result.sha256) == 64
    assert result.content == uploaded
    assert result.size_bytes == len(uploaded)
    assert result.sha256 == hashlib.sha256(uploaded).hexdigest()


def test_jpg_and_jpeg_are_the_same_decoded_mime() -> None:
    result = validate_image(
        _image_bytes("JPEG"),
        "application/octet-stream",
        max_size_bytes=1024 * 1024,
    )
    assert result.mime_type == "image/jpeg"


def test_exif_orientation_and_metadata_are_preserved() -> None:
    uploaded = _image_bytes("JPEG", size=(9, 4), exif_orientation=6)
    result = validate_image(
        uploaded,
        "image/jpeg",
        max_size_bytes=1024 * 1024,
    )
    assert result.content == uploaded
    assert (result.width, result.height) == (9, 4)
    with Image.open(io.BytesIO(result.content)) as retained:
        assert retained.getexif().get(274) == 6


@pytest.mark.parametrize(
    ("data", "declared", "message"),
    [
        (b"not an image", "image/jpeg", "invalid or damaged"),
        (_image_bytes("GIF"), "image/gif", "only JPEG, PNG, and WebP"),
        (_image_bytes("PNG"), "image/jpeg", "does not match"),
    ],
)
def test_validate_rejects_invalid_or_unsupported_inputs(data: bytes, declared: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_image(data, declared, max_size_bytes=1024 * 1024)


def test_validate_enforces_byte_limit() -> None:
    data = _image_bytes("PNG", size=(20, 20))
    with pytest.raises(ValueError, match="byte limit"):
        validate_image(data, "image/png", max_size_bytes=len(data) - 1)


@pytest.mark.asyncio
async def test_mock_provider_and_semantic_rendering_are_deterministic() -> None:
    provider = MockImageAnalysisProvider(
        ImageProviderConfig(provider="mock", api_key=None, model="mock-vision", base_url=None)
    )
    analysis = await provider.analyze(
        ImageAnalysisInput(
            asset_id="asset-1",
            content=b"bytes",
            mime_type="image/jpeg",
            user_content=None,
            common_context=None,
            image_context=None,
        )
    )
    rendered = render_image_analysis_markdown(
        user_content="holiday",
        images=[AnalyzedImage(image_context="beach", analysis=analysis)],
        update_mode="replace",
    )
    assert "A managed image/jpeg image." in rendered
    assert (
        rendered
        == """# Image Memory

## Image Entry

### User-provided Content

holiday

### Image 1

#### Image Context

beach

#### Summary

A managed image/jpeg image.

#### Visual Details

- The image is associated with asset asset-1."""
    )

    appended = render_image_analysis_markdown(
        user_content="new scene",
        images=[
            AnalyzedImage(
                image_context=None,
                analysis=ImageSemanticAnalysis(
                    summary="A second image.",
                    visual_details=["A multiline visual detail\nwith supporting detail."],
                    visual_relations=["The object is beside the first image's subject."],
                ),
            ),
            AnalyzedImage(
                image_context=None,
                analysis=ImageSemanticAnalysis(summary="A third image.", visual_details=[], visual_relations=[]),
            ),
        ],
        update_mode="append",
    )
    # Retain itself contributes the first newline between the existing document
    # and the append fragment; the fragment contributes the blank separator.
    document = f"{rendered}\n{appended}"
    assert document.count("# Image Memory") == 1
    assert document.count("## Image Entry") == 2
    assert "\n\n---\n\n## Image Entry\n" in document
    assert "\n### Image 1\n" in appended
    assert "\n### Image 2\n" in appended
    assert "\n#### Visual Relationships\n" in document
    assert "- A multiline visual detail\n  with supporting detail." in document


def test_unsupported_provider_fails_closed() -> None:
    with pytest.raises(ValueError, match="has no image request adapter"):
        create_image_provider(ImageProviderConfig(provider="text-only", api_key=None, model="model", base_url=None))


def test_image_provider_validation_does_not_guess_capability_from_model_name() -> None:
    validate_image_provider_config(
        ImageProviderConfig(provider="openai", api_key="configured", model="deployment-alias", base_url=None)
    )


def test_image_model_overrides_follow_per_operation_inheritance_and_credential_boundaries() -> None:
    from hindsight_api.config import _get_raw_config

    base = replace(
        _get_raw_config(),
        retain_llm_provider="openai",
        retain_llm_api_key="retain-secret",
        retain_llm_model="gpt-4o",
        retain_llm_base_url="https://retain.example/v1",
        retain_llm_members=[],
        llm_members=[],
        image_llm_provider=None,
        image_llm_api_key=None,
        image_llm_model=None,
        image_llm_base_url=None,
    )
    model_only = resolve_image_provider_configs(base, replace(base, image_llm_model="gpt-5-mini"))
    assert len(model_only) == 1
    assert model_only[0] == ImageProviderConfig(
        provider="openai",
        api_key="retain-secret",
        model="gpt-5-mini",
        base_url="https://retain.example/v1",
        vertexai_project_id=base.llm_vertexai_project_id,
        vertexai_region=base.llm_vertexai_region,
        vertexai_service_account_key=base.llm_vertexai_service_account_key,
    )

    with pytest.raises(ImageProviderConfigurationError, match="IMAGE_LLM_MODEL is required"):
        resolve_image_provider_configs(base, replace(base, image_llm_provider="anthropic"))

    switched = resolve_image_provider_configs(
        base,
        replace(base, image_llm_provider="anthropic", image_llm_model="deployment-vision-alias"),
    )
    assert len(switched) == 1
    assert switched[0].provider == "anthropic"
    assert switched[0].model == "deployment-vision-alias"
    assert switched[0].api_key is None
    assert switched[0].base_url is None


@pytest.mark.asyncio
async def test_openai_image_provider_contract_uses_inline_exact_bytes() -> None:
    captured: dict[str, object] = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"summary":"scene","visual_details":[],"visual_relations":[]}')
                    )
                ]
            )

    provider = object.__new__(OpenAIImageAnalysisProvider)
    provider.provider = "openai"
    provider.model = "gpt-4.1-mini"
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    image_bytes = b"exact-image-bytes"
    result = await provider.analyze(
        ImageAnalysisInput(
            asset_id="asset",
            content=image_bytes,
            mime_type="image/jpeg",
            user_content=None,
            common_context=None,
            image_context=None,
        )
    )
    image_part = captured["messages"][0]["content"][1]
    prompt = captured["messages"][0]["content"][0]["text"]
    assert "unsupported by the image or supplied context" in prompt
    assert "Omit details that cannot be determined" in prompt
    assert "Include visible text when it contributes to the image's meaning" in prompt
    assert "For text-heavy images, summarize the main content and preserve its key information" in prompt
    assert "Do not repeat the same fact across fields" in prompt
    assert "OCR" not in prompt
    assert (
        image_part["image_url"]["url"]
        == "data:image/jpeg;base64," + __import__("base64").b64encode(image_bytes).decode()
    )
    assert set(image_part["image_url"]) == {"url"}
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "image_analysis"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"summary", "visual_details", "visual_relations"}
    assert result.summary == "scene"


@pytest.mark.asyncio
async def test_anthropic_image_provider_contract_uses_base64_source() -> None:
    captured: dict[str, object] = {}

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="image_analysis",
                        input={"summary": "scene", "visual_details": [], "visual_relations": []},
                    )
                ]
            )

    provider = object.__new__(AnthropicImageAnalysisProvider)
    provider.provider = "anthropic"
    provider.model = "claude-3-5-sonnet-latest"
    provider._client = SimpleNamespace(messages=_Messages())
    image_bytes = b"exact-image-bytes"
    result = await provider.analyze(
        ImageAnalysisInput(
            asset_id="asset",
            content=image_bytes,
            mime_type="image/png",
            user_content=None,
            common_context=None,
            image_context=None,
        )
    )
    source = captured["messages"][0]["content"][0]["source"]
    assert source == {
        "type": "base64",
        "media_type": "image/png",
        "data": __import__("base64").b64encode(image_bytes).decode(),
    }
    assert captured["tool_choice"] == {"type": "tool", "name": "image_analysis"}
    assert captured["tools"][0]["input_schema"]["additionalProperties"] is False
    assert result.summary == "scene"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["gemini", "vertexai"])
async def test_gemini_and_vertex_image_provider_contract_use_byte_parts(provider_name: str) -> None:
    captured: dict[str, object] = {}

    class _Models:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                parsed={"summary": "scene", "visual_details": [], "visual_relations": []},
                text=None,
            )

    provider = object.__new__(GeminiImageAnalysisProvider)
    provider.provider = provider_name
    provider.model = "gemini-2.5-flash"
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=_Models()))
    result = await provider.analyze(
        ImageAnalysisInput(
            asset_id="asset",
            content=b"exact-image-bytes",
            mime_type="image/webp",
            user_content=None,
            common_context=None,
            image_context=None,
        )
    )
    assert captured["model"] == "gemini-2.5-flash"
    assert len(captured["contents"]) == 2
    schema = captured["config"].response_schema.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"summary", "visual_details", "visual_relations"}
    assert result.summary == "scene"


class _MemoryStorage(FileStorage):
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def store(self, file_data: bytes, key: str, metadata: dict[str, str] | None = None) -> str:
        self.data = file_data
        return key

    async def retrieve(self, key: str) -> bytes:
        return self.data

    async def delete(self, key: str) -> None:
        self.data = b""

    async def exists(self, key: str) -> bool:
        return bool(self.data)

    async def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        raise AssertionError("image reads must not request download URLs")


@pytest.mark.asyncio
async def test_file_storage_compatibility_stream_is_chunked_without_url() -> None:
    storage = _MemoryStorage(b"abcdefgh")
    assert [chunk async for chunk in storage.iter_bytes("key", chunk_size=3)] == [b"abc", b"def", b"gh"]
    assert (await storage.stat("key")).size_bytes == 8


@pytest.mark.asyncio
async def test_postgresql_file_storage_stream_roundtrip(memory) -> None:
    key = f"stream-contract/{uuid.uuid4()}"

    async def chunks():
        yield b"abc"
        yield b"defgh"

    try:
        await memory._file_storage.store_stream(chunks(), key)
        assert await memory._file_storage.exists(key)
        assert (await memory._file_storage.stat(key)).size_bytes == 8
        assert b"".join([chunk async for chunk in memory._file_storage.iter_bytes(key)]) == b"abcdefgh"
        assert await memory._file_storage.retrieve(key) == b"abcdefgh"
    finally:
        await memory._file_storage.delete(key)


@pytest.mark.asyncio
async def test_native_file_storage_uses_oracle_blob_length(monkeypatch) -> None:
    queries: list[str] = []

    class _Connection:
        async def fetchval(self, query: str, key: str) -> int:
            queries.append(query)
            assert key == "asset"
            return 8

    @asynccontextmanager
    async def fake_acquire(_pool: object) -> AsyncIterator[_Connection]:
        yield _Connection()

    monkeypatch.setattr(postgresql_storage, "_is_oracle", lambda: True)
    monkeypatch.setattr(postgresql_storage, "acquire_with_retry", fake_acquire)
    storage = postgresql_storage.PostgreSQLFileStorage(lambda: object())

    assert (await storage.stat("asset")).size_bytes == 8
    assert "DBMS_LOB.GETLENGTH(data)" in queries[0]
    assert "DBMS_LOB.GETLENGTH(f.data)" in queries[0]
    assert "octet_length" not in queries[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_type", [S3FileStorage, GCSFileStorage, AzureFileStorage])
async def test_object_file_storage_stream_upload_uses_disk_backed_multipart(storage_type, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def put_async(store, key, file, **kwargs):
        captured.update({"store": store, "key": key, "data": open(file, "rb").read(), **kwargs})

    monkeypatch.setattr("obstore.put_async", put_async)
    storage = object.__new__(storage_type)
    storage._store = object()

    async def chunks():
        yield b"abc"
        yield b"def"

    assert await storage.store_stream(chunks(), "stream-key") == "stream-key"
    assert captured["data"] == b"abcdef"
    assert captured["use_multipart"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_type", [S3FileStorage, GCSFileStorage, AzureFileStorage])
async def test_object_file_storage_stream_contract(storage_type, monkeypatch) -> None:
    """All managed-image object backends expose the same non-URL lifecycle."""
    calls: list[tuple[str, object]] = []

    class _Response:
        async def bytes_async(self) -> bytes:
            return b"abcdef"

        async def stream(self, *, min_chunk_size: int):
            calls.append(("chunk_size", min_chunk_size))
            yield b"abc"
            yield b"def"

    async def head_async(_store, key):
        calls.append(("head", key))
        return {"size": 6, "e_tag": "etag"}

    async def get_async(_store, key):
        calls.append(("get", key))
        return _Response()

    async def delete_async(_store, key):
        calls.append(("delete", key))

    monkeypatch.setattr("obstore.head_async", head_async)
    monkeypatch.setattr("obstore.get_async", get_async)
    monkeypatch.setattr("obstore.delete_async", delete_async)
    storage = object.__new__(storage_type)
    storage._store = object()
    storage.get_download_url = AsyncMock(side_effect=AssertionError("managed images never request URLs"))

    assert await storage.exists("asset")
    info = await storage.stat("asset")
    assert (info.size_bytes, info.etag) == (6, "etag")
    assert b"".join([chunk async for chunk in storage.iter_bytes("asset", chunk_size=3)]) == b"abcdef"
    await storage.delete("asset")
    storage.get_download_url.assert_not_awaited()
    assert ("delete", "asset") in calls


def _pending_image(asset_id: str) -> PendingImageAsset:
    return PendingImageAsset(
        asset_id=asset_id,
        image=validate_image(
            _image_bytes("PNG"),
            "image/png",
            max_size_bytes=1024 * 1024,
        ),
        context=None,
        asset_id_supplied=True,
    )


async def _insert_image_operation(
    memory,
    *,
    bank_id: str,
    document_id: str,
    asset_ids: list[str],
    batch_id: uuid.UUID,
    published: bool = False,
) -> str:
    operation_id = uuid.uuid4()
    metadata = ImageRetainOperationMetadata(
        document_id=document_id,
        asset_ids=asset_ids,
        published_batch_id=str(batch_id) if published else None,
    )
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        await conn.execute(
            f"INSERT INTO {fq_table('async_operations')} "
            "(operation_id, bank_id, operation_type, result_metadata, status, task_payload) "
            "VALUES ($1, $2, 'image_semantic_retain', $3, 'processing', $4::jsonb)",
            operation_id,
            bank_id,
            metadata.model_dump_json(),
            "{}",
        )
    return str(operation_id)


@pytest.mark.asyncio
async def test_image_admission_persists_assets_and_operation_atomically(memory, request_context, monkeypatch) -> None:
    bank_id = f"test-image-admission-{uuid.uuid4()}"
    document_id = "document-admission"
    submit_task = AsyncMock()
    monkeypatch.setattr(memory._task_backend, "submit_task", submit_task)
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed.", "document_id": "seed"}],
            request_context=request_context,
        )
        submit_task.reset_mock()
        result = await memory.submit_image_retain(
            bank_id,
            [_pending_image("asset-admission")],
            ImageRetainParameters(document_id=document_id),
            request_context=request_context,
            idempotency_key="admission-key",
        )
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            asset = await conn.fetchrow(
                f"SELECT storage_key FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                "asset-admission",
            )
            operation = await conn.fetchrow(
                f"SELECT task_payload FROM {fq_table('async_operations')} WHERE operation_id = $1",
                uuid.UUID(result.operation_id),
            )
        assert asset is not None
        assert operation is not None
        payload_text = str(operation["task_payload"])
        assert str(asset["storage_key"]) not in payload_text
        assert "data:image" not in payload_text
        assert "base64" not in payload_text
        submit_task.assert_awaited_once()
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_image_admission_rolls_back_assets_and_blobs_if_operation_transaction_fails(
    memory, request_context, monkeypatch
) -> None:
    bank_id = f"test-image-admission-rollback-{uuid.uuid4()}"
    monkeypatch.setattr(memory._task_backend, "submit_task", AsyncMock())
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed.", "document_id": "seed"}],
            request_context=request_context,
        )
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("image_assets")}
                    (bank_id, asset_id, storage_key, mime_type, size_bytes, sha256, width, height, status)
                VALUES ($1, 'asset-existing', $2, 'image/png', 1, $3, 1, 1, 'ready')
                """,
                bank_id,
                f"preexisting/{uuid.uuid4()}",
                "0" * 64,
            )
            blobs_before = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('file_storage')} WHERE storage_key LIKE 'image-assets/%'"
            )

        with pytest.raises(Exception):
            await memory.submit_image_retain(
                bank_id,
                [_pending_image("asset-new"), _pending_image("asset-existing")],
                ImageRetainParameters(document_id="document-rollback"),
                request_context=request_context,
            )

        async with acquire_with_retry(backend) as conn:
            new_asset = await conn.fetchval(
                f"SELECT 1 FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = 'asset-new'",
                bank_id,
            )
            operation_count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('async_operations')} "
                "WHERE bank_id = $1 AND operation_type = 'image_semantic_retain'",
                bank_id,
            )
            blobs_after = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('file_storage')} WHERE storage_key LIKE 'image-assets/%'"
            )
        assert new_asset is None
        assert operation_count == 0
        assert blobs_after == blobs_before
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_concurrent_idempotent_image_admission_returns_one_operation(
    memory, request_context, monkeypatch
) -> None:
    bank_id = f"test-image-idempotent-race-{uuid.uuid4()}"
    submit_task = AsyncMock()
    monkeypatch.setattr(memory._task_backend, "submit_task", submit_task)
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed.", "document_id": "seed"}],
            request_context=request_context,
        )
        submit_task.reset_mock()

        async def submit_once():
            return await memory.submit_image_retain(
                bank_id,
                [_pending_image("asset-idempotent")],
                ImageRetainParameters(document_id="document-idempotent"),
                request_context=request_context,
                idempotency_key="same-key",
            )

        first, second = await asyncio.gather(submit_once(), submit_once())
        assert first.operation_id == second.operation_id
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            assert (
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = $2",
                    bank_id,
                    "asset-idempotent",
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('async_operations')} "
                    "WHERE bank_id = $1 AND operation_type = 'image_semantic_retain'",
                    bank_id,
                )
                == 1
            )
        submit_task.assert_awaited_once()
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


def test_analysis_schema_requires_exact_closed_shape() -> None:
    with pytest.raises(ValueError):
        ImageSemanticAnalysis(summary="", visual_details=[], visual_relations=[])
    with pytest.raises(ValueError):
        ImageSemanticAnalysis.model_validate({"summary": "scene", "visual_details": []})
    with pytest.raises(ValueError):
        ImageSemanticAnalysis.model_validate(
            {
                "summary": "scene",
                "visual_details": [],
                "visual_relations": [],
                "caption": "unexpected",
            }
        )


def _image_transfer_archive(*, data: bytes, archive_entry: str = "assets/000000") -> bytes:
    validated = validate_image(
        data,
        "image/png",
        max_size_bytes=1024 * 1024,
    )
    document = TransferDocument(
        id="document-1",
        image_assets=[
            TransferImageAsset(
                asset_id="asset-1",
                mime_type=validated.mime_type,
                size_bytes=validated.size_bytes,
                sha256=validated.sha256,
                width=validated.width,
                height=validated.height,
                status="ready",
                archive_entry=archive_entry,
            )
        ],
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            TransferManifest(source_bank_id="source", document_count=1, image_asset_count=1).model_dump_json(),
        )
        archive.writestr("documents/000000.json", document.model_dump_json())
        archive.writestr(archive_entry, validated.content)
    return output.getvalue()


def test_transfer_v2_parses_and_validates_managed_image_bytes() -> None:
    archive_bytes = _image_transfer_archive(data=_image_bytes("PNG"))
    parsed = parse_archive(archive_bytes)
    assert parsed.manifest.schema_version == 2
    assert parsed.documents[0].image_assets[0].asset_id == "asset-1"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert hashlib.sha256(archive.read("assets/000000")).hexdigest() == parsed.documents[0].image_assets[0].sha256


def test_transfer_v2_rejects_unsafe_image_entry() -> None:
    archive = _image_transfer_archive(data=_image_bytes("PNG"), archive_entry="assets/../image")
    with pytest.raises(ValueError, match="Invalid (image|transfer) archive entry"):
        parse_archive(archive)


def test_transfer_v2_rejects_tampered_image_bytes(tmp_path: Path) -> None:
    archive_bytes = _image_transfer_archive(data=_image_bytes("PNG"))
    source = zipfile.ZipFile(io.BytesIO(archive_bytes))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            target.writestr(item, data + b"tampered" if item.filename == "assets/000000" else data)
    archive_path = tmp_path / "tampered.zip"
    archive_path.write_bytes(output.getvalue())
    with pytest.raises(ValueError, match="size mismatch"):
        validate_archive_file(str(archive_path), image_max_file_size_bytes=10 * 1024 * 1024)


@pytest.mark.asyncio
async def test_published_image_batch_is_worker_retry_marker(memory, request_context, monkeypatch) -> None:
    """A post-commit retry returns before reading assets or calling the provider."""
    bank_id = "test-image-batch-retry-marker"
    document_id = "document-retry-marker"
    batch_id = uuid.uuid4()
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Already published image semantics.", "document_id": document_id}],
            request_context=request_context,
        )
        operation_id = await _insert_image_operation(
            memory,
            bank_id=bank_id,
            document_id=document_id,
            asset_ids=["asset-does-not-need-to-be-read"],
            batch_id=batch_id,
            published=True,
        )

        def fail_if_provider_is_created(_config: object) -> None:
            raise AssertionError("published image retries must not call the vision provider")

        monkeypatch.setattr("hindsight_api.engine.memory_engine.create_image_provider", fail_if_provider_is_created)
        await memory._handle_image_semantic_retain(
            {
                "bank_id": bank_id,
                "document_id": document_id,
                "operation_id": operation_id,
                "batch_id": str(batch_id),
                "asset_ids": ["asset-does-not-need-to-be-read"],
            }
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_image_worker_publishes_document_and_links_in_retain_callback(
    memory, request_context, monkeypatch
) -> None:
    """The real image Worker path persists its batch/link through Retain's callback."""
    bank_id = f"test-image-worker-publish-{uuid.uuid4()}"
    document_id = "document-image-worker"
    asset_id = "asset-image-worker"
    batch_id = uuid.uuid4()
    storage_key = f"image-assets/{uuid.uuid4()}"
    validated = validate_image(
        _image_bytes("PNG"),
        "image/png",
        max_size_bytes=1024 * 1024,
    )
    mock_provider = MockImageAnalysisProvider(
        ImageProviderConfig(provider="mock", api_key=None, model="mock-vision", base_url=None)
    )
    monkeypatch.setattr(
        "hindsight_api.engine.memory_engine.create_image_provider",
        lambda _config: mock_provider,
    )
    monkeypatch.setattr("hindsight_api.engine.memory_engine.validate_image_provider_config", lambda _config: None)
    try:
        # Create the bank before inserting its FK-owned asset. The image Worker
        # then replaces this seed Document through the same public Retain path.
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed text.", "document_id": document_id}],
            request_context=request_context,
        )
        await memory._file_storage.store(validated.content, storage_key)
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("image_assets")}
                    (bank_id, asset_id, storage_key, mime_type, size_bytes, sha256,
                     width, height, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'ready')
                """,
                bank_id,
                asset_id,
                storage_key,
                validated.mime_type,
                validated.size_bytes,
                validated.sha256,
                validated.width,
                validated.height,
            )
        operation_id = await _insert_image_operation(
            memory,
            bank_id=bank_id,
            document_id=document_id,
            asset_ids=[asset_id],
            batch_id=batch_id,
        )

        await memory._handle_image_semantic_retain(
            {
                "bank_id": bank_id,
                "document_id": document_id,
                "operation_id": operation_id,
                "batch_id": str(batch_id),
                "asset_ids": [asset_id],
                "images": [{"context": "test image"}],
                "retain": {"document_id": document_id, "update_mode": "replace"},
            }
        )

        async with acquire_with_retry(backend) as conn:
            link = await conn.fetchrow(
                f"SELECT ordinal, image_context FROM {fq_table('document_image_links')} "
                "WHERE bank_id = $1 AND document_id = $2 AND asset_id = $3",
                bank_id,
                document_id,
                asset_id,
            )
            document_text = await conn.fetchval(
                f"SELECT original_text FROM {fq_table('documents')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                document_id,
            )
        assert link is not None
        assert int(link["ordinal"]) == 0
        assert link["image_context"] == "test image"
        assert document_text.startswith("# Image Memory\n\n## Image Entry")
        assert "\n#### Summary\n" in document_text
        assert "hindsight:image-semantic" not in document_text

        with pytest.raises(ImageAssetInUseError, match="delete the document"):
            await memory.delete_image_asset(bank_id, asset_id, request_context=request_context)
        async with acquire_with_retry(backend) as conn:
            asset_exists = await conn.fetchval(
                f"SELECT 1 FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                asset_id,
            )
            link_exists = await conn.fetchval(
                f"SELECT 1 FROM {fq_table('document_image_links')} WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                asset_id,
            )
            retained_text = await conn.fetchval(
                f"SELECT original_text FROM {fq_table('documents')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                document_id,
            )
        assert asset_exists == 1
        assert link_exists == 1
        assert "\n#### Summary\n" in retained_text

        await memory.delete_document(document_id, bank_id, request_context=request_context)
        assert await memory.delete_image_asset(bank_id, asset_id, request_context=request_context)
        async with acquire_with_retry(backend) as conn:
            assert (
                await conn.fetchval(
                    f"SELECT 1 FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = $2",
                    bank_id,
                    asset_id,
                )
                is None
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        try:
            await memory._file_storage.delete(storage_key)
        except FileNotFoundError:
            pass


@pytest.mark.asyncio
async def test_delete_during_image_analysis_prevents_final_link_publication(
    memory, request_context, monkeypatch
) -> None:
    bank_id = f"test-image-delete-publish-race-{uuid.uuid4()}"
    document_id = "document-race"
    asset_id = "asset-race"
    storage_key = f"image-assets/{uuid.uuid4()}"
    analysis_started = asyncio.Event()
    release_analysis = asyncio.Event()

    class _BlockingProvider(MockImageAnalysisProvider):
        async def analyze(self, request: ImageAnalysisInput) -> ImageSemanticAnalysis:
            analysis_started.set()
            await release_analysis.wait()
            return await super().analyze(request)

    provider = _BlockingProvider(ImageProviderConfig(provider="mock", api_key=None, model="mock-vision", base_url=None))
    monkeypatch.setattr("hindsight_api.engine.memory_engine.create_image_provider", lambda _config: provider)
    monkeypatch.setattr("hindsight_api.engine.memory_engine.validate_image_provider_config", lambda _config: None)
    monkeypatch.setattr(memory._task_backend, "submit_task", AsyncMock())
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed.", "document_id": document_id}],
            request_context=request_context,
        )
        validated = _pending_image(asset_id).image
        await memory._file_storage.store(validated.content, storage_key)
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("image_assets")}
                    (bank_id, asset_id, storage_key, mime_type, size_bytes, sha256, width, height, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'ready')
                """,
                bank_id,
                asset_id,
                storage_key,
                validated.mime_type,
                validated.size_bytes,
                validated.sha256,
                validated.width,
                validated.height,
            )
        batch_id = uuid.uuid4()
        operation_id = await _insert_image_operation(
            memory,
            bank_id=bank_id,
            document_id=document_id,
            asset_ids=[asset_id],
            batch_id=batch_id,
        )
        worker = asyncio.create_task(
            memory._handle_image_semantic_retain(
                {
                    "bank_id": bank_id,
                    "document_id": document_id,
                    "operation_id": operation_id,
                    "batch_id": str(batch_id),
                    "asset_ids": [asset_id],
                    "images": [{"context": None}],
                    "retain": {"document_id": document_id, "update_mode": "append"},
                }
            )
        )
        await analysis_started.wait()
        assert await memory.delete_image_asset(bank_id, asset_id, request_context=request_context)
        release_analysis.set()
        with pytest.raises(ValueError, match="became unavailable"):
            await worker
        async with acquire_with_retry(backend) as conn:
            link = await conn.fetchval(
                f"SELECT 1 FROM {fq_table('document_image_links')} WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                asset_id,
            )
        assert link is None
    finally:
        release_analysis.set()
        await memory.delete_bank(bank_id, request_context=request_context)
        try:
            await memory._file_storage.delete(storage_key)
        except FileNotFoundError:
            pass


@pytest.mark.asyncio
async def test_concurrent_append_enforces_image_cap_inside_document_lock(memory, request_context, monkeypatch) -> None:
    from hindsight_api.engine.image import MAX_IMAGES_PER_DOCUMENT

    bank_id = f"test-image-cap-race-{uuid.uuid4()}"
    document_id = "document-cap"
    new_asset_ids = ["asset-cap-a", "asset-cap-b"]
    new_storage_keys = [f"image-assets/{uuid.uuid4()}" for _ in new_asset_ids]
    provider = MockImageAnalysisProvider(
        ImageProviderConfig(provider="mock", api_key=None, model="mock-vision", base_url=None)
    )
    monkeypatch.setattr("hindsight_api.engine.memory_engine.create_image_provider", lambda _config: provider)
    monkeypatch.setattr("hindsight_api.engine.memory_engine.validate_image_provider_config", lambda _config: None)
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed.", "document_id": document_id}],
            request_context=request_context,
        )
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                for ordinal in range(MAX_IMAGES_PER_DOCUMENT - 1):
                    old_asset_id = f"old-{ordinal}"
                    await conn.execute(
                        f"INSERT INTO {fq_table('image_assets')} "
                        "(bank_id, asset_id, storage_key, mime_type, size_bytes, sha256, width, height, status) "
                        "VALUES ($1, $2, $3, 'image/png', 1, $4, 1, 1, 'ready')",
                        bank_id,
                        old_asset_id,
                        f"old-storage/{uuid.uuid4()}",
                        f"{ordinal:064x}",
                    )
                    await conn.execute(
                        f"INSERT INTO {fq_table('document_image_links')} "
                        "(bank_id, document_id, asset_id, ordinal) "
                        "VALUES ($1, $2, $3, $4)",
                        bank_id,
                        document_id,
                        old_asset_id,
                        ordinal,
                    )
        for asset_id, storage_key in zip(new_asset_ids, new_storage_keys, strict=True):
            validated = _pending_image(asset_id).image
            await memory._file_storage.store(validated.content, storage_key)
            async with acquire_with_retry(backend) as conn:
                await conn.execute(
                    f"INSERT INTO {fq_table('image_assets')} "
                    "(bank_id, asset_id, storage_key, mime_type, size_bytes, sha256, width, height, status) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'ready')",
                    bank_id,
                    asset_id,
                    storage_key,
                    validated.mime_type,
                    validated.size_bytes,
                    validated.sha256,
                    validated.width,
                    validated.height,
                )

        async def retain_with_document_lock(*_args, outbox_callback=None, **_kwargs):
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    await conn.fetchval(
                        f"SELECT 1 FROM {fq_table('documents')} WHERE bank_id = $1 AND id = $2 FOR UPDATE",
                        bank_id,
                        document_id,
                    )
                    await outbox_callback(conn)

        monkeypatch.setattr(memory, "retain_batch_async", retain_with_document_lock)

        async def append_one(asset_id: str):
            batch_id = uuid.uuid4()
            operation_id = await _insert_image_operation(
                memory,
                bank_id=bank_id,
                document_id=document_id,
                asset_ids=[asset_id],
                batch_id=batch_id,
            )
            return await memory._handle_image_semantic_retain(
                {
                    "bank_id": bank_id,
                    "document_id": document_id,
                    "operation_id": operation_id,
                    "batch_id": str(batch_id),
                    "asset_ids": [asset_id],
                    "images": [{"context": None}],
                    "retain": {"document_id": document_id, "update_mode": "append"},
                }
            )

        outcomes = await asyncio.gather(*(append_one(asset_id) for asset_id in new_asset_ids), return_exceptions=True)
        assert sum(outcome is None for outcome in outcomes) == 1
        assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
        async with acquire_with_retry(backend) as conn:
            link_count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('document_image_links')} WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        assert link_count == MAX_IMAGES_PER_DOCUMENT
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        for storage_key in new_storage_keys:
            try:
                await memory._file_storage.delete(storage_key)
            except FileNotFoundError:
                pass


@pytest.mark.asyncio
async def test_concurrent_image_append_and_replace_publish_a_consistent_link_set(
    memory, request_context, monkeypatch
) -> None:
    """The Document lock serializes append/replace without mixed old batches."""
    bank_id = f"test-image-append-replace-race-{uuid.uuid4()}"
    document_id = "document-append-replace"
    asset_ids = ["asset-append", "asset-replace"]
    storage_keys = [f"image-assets/{uuid.uuid4()}" for _ in asset_ids]
    provider = MockImageAnalysisProvider(
        ImageProviderConfig(provider="mock", api_key=None, model="mock-vision", base_url=None)
    )
    monkeypatch.setattr("hindsight_api.engine.memory_engine.create_image_provider", lambda _config: provider)
    monkeypatch.setattr("hindsight_api.engine.memory_engine.validate_image_provider_config", lambda _config: None)
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed.", "document_id": document_id}],
            request_context=request_context,
        )
        backend = await memory._get_backend()
        for asset_id, storage_key in zip(asset_ids, storage_keys, strict=True):
            validated = _pending_image(asset_id).image
            await memory._file_storage.store(validated.content, storage_key)
            async with acquire_with_retry(backend) as conn:
                await conn.execute(
                    f"INSERT INTO {fq_table('image_assets')} "
                    "(bank_id, asset_id, storage_key, mime_type, size_bytes, sha256, width, height, status) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'ready')",
                    bank_id,
                    asset_id,
                    storage_key,
                    validated.mime_type,
                    validated.size_bytes,
                    validated.sha256,
                    validated.width,
                    validated.height,
                )

        publish_barrier = asyncio.Event()
        arrived = 0

        async def synchronized_retain(*_args, outbox_callback=None, **_kwargs):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                publish_barrier.set()
            await publish_barrier.wait()
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    await conn.fetchval(
                        f"SELECT 1 FROM {fq_table('documents')} WHERE bank_id = $1 AND id = $2 FOR UPDATE",
                        bank_id,
                        document_id,
                    )
                    await outbox_callback(conn)

        monkeypatch.setattr(memory, "retain_batch_async", synchronized_retain)

        async def publish(asset_id: str, update_mode: str) -> None:
            batch_id = uuid.uuid4()
            operation_id = await _insert_image_operation(
                memory,
                bank_id=bank_id,
                document_id=document_id,
                asset_ids=[asset_id],
                batch_id=batch_id,
            )
            await memory._handle_image_semantic_retain(
                {
                    "bank_id": bank_id,
                    "document_id": document_id,
                    "operation_id": operation_id,
                    "batch_id": str(batch_id),
                    "asset_ids": [asset_id],
                    "images": [{"context": None}],
                    "retain": {"document_id": document_id, "update_mode": update_mode},
                }
            )

        await asyncio.gather(publish(asset_ids[0], "append"), publish(asset_ids[1], "replace"))
        async with acquire_with_retry(backend) as conn:
            links = await conn.fetch(
                f"SELECT asset_id, ordinal FROM {fq_table('document_image_links')} "
                "WHERE bank_id = $1 AND document_id = $2 ORDER BY ordinal",
                bank_id,
                document_id,
            )
        published = [str(row["asset_id"]) for row in links]
        # replace after append yields only replacement; replace before append
        # yields replacement followed by the append. No interleaved old batch is
        # a valid outcome.
        assert published in [[asset_ids[1]], [asset_ids[1], asset_ids[0]]]
        assert [int(row["ordinal"]) for row in links] == list(range(len(links)))
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        for storage_key in storage_keys:
            try:
                await memory._file_storage.delete(storage_key)
            except FileNotFoundError:
                pass


@pytest.mark.asyncio
async def test_image_document_transfer_restores_exact_bytes_and_links(memory, request_context, monkeypatch) -> None:
    source_bank = f"test-image-transfer-source-{uuid.uuid4()}"
    target_bank = f"test-image-transfer-target-{uuid.uuid4()}"
    document_id = "document-transfer-image"
    asset_id = "asset-transfer-image"
    storage_key = f"image-assets/{uuid.uuid4()}"
    uploaded = _image_bytes("JPEG", exif_orientation=6)
    validated = validate_image(uploaded, "image/jpeg", max_size_bytes=1024 * 1024)
    provider = MockImageAnalysisProvider(
        ImageProviderConfig(provider="mock", api_key=None, model="mock-vision", base_url=None)
    )
    monkeypatch.setattr("hindsight_api.engine.memory_engine.create_image_provider", lambda _config: provider)
    monkeypatch.setattr("hindsight_api.engine.memory_engine.validate_image_provider_config", lambda _config: None)
    try:
        await memory.retain_batch_async(
            bank_id=source_bank,
            contents=[{"content": "Seed.", "document_id": document_id}],
            request_context=request_context,
        )
        await memory._file_storage.store(uploaded, storage_key)
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"INSERT INTO {fq_table('image_assets')} "
                "(bank_id, asset_id, storage_key, mime_type, size_bytes, sha256, width, height, status) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'ready')",
                source_bank,
                asset_id,
                storage_key,
                validated.mime_type,
                validated.size_bytes,
                validated.sha256,
                validated.width,
                validated.height,
            )
        batch_id = uuid.uuid4()
        operation_id = await _insert_image_operation(
            memory,
            bank_id=source_bank,
            document_id=document_id,
            asset_ids=[asset_id],
            batch_id=batch_id,
        )
        await memory._handle_image_semantic_retain(
            {
                "bank_id": source_bank,
                "document_id": document_id,
                "operation_id": operation_id,
                "batch_id": str(batch_id),
                "asset_ids": [asset_id],
                "images": [{"context": "transfer"}],
                "retain": {"document_id": document_id, "update_mode": "replace"},
            }
        )

        async with acquire_with_retry(backend) as conn:
            bank_archive = await export_bank(conn, source_bank, file_storage=memory._file_storage)
        try:
            with zipfile.ZipFile(bank_archive.path) as archive_file:
                exported_document = json.loads(archive_file.read("documents/000000.json"))
                exported_entry = exported_document["image_assets"][0]["archive_entry"]
                assert archive_file.read(exported_entry) == uploaded
                assert exported_document["image_links"][0]["asset_id"] == asset_id
        finally:
            bank_archive.cleanup()

        archive = await memory.export_documents_async(source_bank, request_context)
        archive_bytes = Path(archive.path).read_bytes()
        archive.cleanup()
        submission = await memory.import_documents_async(target_bank, archive_bytes, request_context)
        operation = await memory.get_operation_status(
            target_bank, submission["operation_id"], request_context=request_context
        )
        assert operation["status"] == "completed", operation
        imported = await memory.get_image_asset(target_bank, asset_id, request_context=request_context)
        assert imported is not None
        assert b"".join([chunk async for chunk in memory.iter_image_asset(imported)]) == uploaded
        async with acquire_with_retry(backend) as conn:
            link = await conn.fetchrow(
                f"SELECT ordinal, image_context FROM {fq_table('document_image_links')} "
                "WHERE bank_id = $1 AND document_id = $2 AND asset_id = $3",
                target_bank,
                document_id,
                asset_id,
            )
        assert link is not None
        assert link["ordinal"] == 0
        assert link["image_context"] == "transfer"

        skipped = await memory.import_documents_async(target_bank, archive_bytes, request_context, "skip")
        skipped_status = await memory.get_operation_status(
            target_bank, skipped["operation_id"], request_context=request_context
        )
        assert skipped_status["result_metadata"]["documents_skipped"] == 1
        replaced = await memory.import_documents_async(target_bank, archive_bytes, request_context, "replace")
        assert (
            await memory.get_operation_status(target_bank, replaced["operation_id"], request_context=request_context)
        )["status"] == "completed"
        remapped = await memory.import_documents_async(target_bank, archive_bytes, request_context, "new-id")
        remapped_status = await memory.get_operation_status(
            target_bank, remapped["operation_id"], request_context=request_context
        )
        assert document_id in remapped_status["result_metadata"]["remapped_document_ids"]
        async with acquire_with_retry(backend) as conn:
            assert (
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('image_assets')} WHERE bank_id = $1",
                    target_bank,
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('document_image_links')} WHERE bank_id = $1",
                    target_bank,
                )
                == 2
            )
    finally:
        await memory.delete_bank(source_bank, request_context=request_context)
        await memory.delete_bank(target_bank, request_context=request_context)
        try:
            await memory._file_storage.delete(storage_key)
        except FileNotFoundError:
            pass


@pytest.mark.asyncio
async def test_terminal_image_failure_marks_unlinked_asset_failed(memory, request_context) -> None:
    bank_id = f"test-image-failed-state-{uuid.uuid4()}"
    document_id = "document-failed-state"
    asset_id = "asset-failed-state"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed text.", "document_id": document_id}],
            request_context=request_context,
        )
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("image_assets")}
                    (bank_id, asset_id, storage_key, mime_type, size_bytes, sha256,
                     width, height, status)
                VALUES ($1, $2, $3, 'image/jpeg', 3, $4, 1, 1, 'ready')
                """,
                bank_id,
                asset_id,
                f"image-assets/{uuid.uuid4()}",
                "2" * 64,
            )

        await memory._finalize_image_task_failure(
            "image_semantic_retain",
            {"bank_id": bank_id, "asset_ids": [asset_id]},
        )

        assets = await memory.list_image_assets(bank_id, request_context=request_context)
        failed = next(item for item in assets.items if item.asset_id == asset_id)
        assert failed.status == "failed"
        assert failed.document_ids == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_image_delete_is_synchronous_and_retryable(memory, request_context, monkeypatch) -> None:
    bank_id = f"test-image-delete-{uuid.uuid4()}"
    asset_id = "asset-delete"
    storage_key = f"image-assets/{uuid.uuid4()}"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Seed.", "document_id": "seed"}],
            request_context=request_context,
        )
        validated = _pending_image(asset_id).image
        await memory._file_storage.store(validated.content, storage_key)
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("image_assets")}
                    (bank_id, asset_id, storage_key, mime_type, size_bytes, sha256, width, height, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'ready')
                """,
                bank_id,
                asset_id,
                storage_key,
                validated.mime_type,
                validated.size_bytes,
                validated.sha256,
                validated.width,
                validated.height,
            )

        original_delete = memory._file_storage.delete
        failed_delete = AsyncMock(side_effect=RuntimeError("storage unavailable"))
        monkeypatch.setattr(memory._file_storage, "delete", failed_delete)
        with pytest.raises(RuntimeError, match="storage unavailable"):
            await memory.delete_image_asset(bank_id, asset_id, request_context=request_context)
        async with acquire_with_retry(backend) as conn:
            status = await conn.fetchval(
                f"SELECT status FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                asset_id,
            )
            delete_operations = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('async_operations')} "
                "WHERE bank_id = $1 AND operation_type = 'image_asset_delete'",
                bank_id,
            )
        assert status == "deleting"
        assert int(delete_operations) == 0

        monkeypatch.setattr(memory._file_storage, "delete", original_delete)
        assert await memory.delete_image_asset(bank_id, asset_id, request_context=request_context)
        async with acquire_with_retry(backend) as conn:
            assert (
                await conn.fetchval(
                    f"SELECT 1 FROM {fq_table('image_assets')} WHERE bank_id = $1 AND asset_id = $2",
                    bank_id,
                    asset_id,
                )
                is None
            )
        assert not await memory._file_storage.exists(storage_key)
        assert not await memory.delete_image_asset(bank_id, asset_id, request_context=request_context)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        try:
            await memory._file_storage.delete(storage_key)
        except FileNotFoundError:
            pass


@pytest.mark.asyncio
async def test_image_http_multipart_management_and_recall_contract(api_client, memory, monkeypatch) -> None:
    from hindsight_api.config import clear_config_cache

    bank_id = f"test-image-http-{uuid.uuid4()}"
    other_bank = f"test-image-http-other-{uuid.uuid4()}"
    asset_id = "folder/photo.jpg"
    uploaded = _image_bytes("JPEG", exif_orientation=6)
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_IMAGE_RETAIN_API", "true")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_MODEL", "mock-vision")
    clear_config_cache()
    download_url = AsyncMock(side_effect=AssertionError("managed images never request download URLs"))
    storage_stat = AsyncMock(side_effect=AssertionError("image downloads do not issue a storage HEAD request"))
    monkeypatch.setattr(memory._file_storage, "get_download_url", download_url)
    monkeypatch.setattr(memory._file_storage, "stat", storage_stat)
    try:
        request_payload = {
            "document_id": "document-http-image",
            "content": "A preserved photo",
            "metadata": {"source": "http-test"},
            "tags": ["photos"],
            "images": [{"asset_id": asset_id, "context": "the first image"}],
        }
        retained = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/image-retain",
            files=[
                ("files", ("photo.jpg", uploaded, "image/jpeg")),
                (
                    "request",
                    (
                        None,
                        json.dumps(request_payload),
                        "application/json",
                    ),
                ),
            ],
            headers={"Idempotency-Key": "http-image-key"},
        )
        assert retained.status_code == 202, retained.text
        assert retained.json()["image_assets"][0]["asset_id"] == asset_id
        replayed = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/image-retain",
            files=[
                ("files", ("photo.jpg", uploaded, "image/jpeg")),
                ("request", (None, json.dumps(request_payload), "application/json")),
            ],
            headers={"Idempotency-Key": "http-image-key"},
        )
        assert replayed.status_code == 202
        assert replayed.json()["operation_id"] == retained.json()["operation_id"]
        conflicting_payload = {**request_payload, "content": "different request"}
        conflicted = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/image-retain",
            files=[
                ("files", ("photo.jpg", uploaded, "image/jpeg")),
                ("request", (None, json.dumps(conflicting_payload), "application/json")),
            ],
            headers={"Idempotency-Key": "http-image-key"},
        )
        assert conflicted.status_code == 409

        # Disabling image input only prevents new semantic image Retain calls. Existing
        # managed assets remain available for Recall and lifecycle management.
        monkeypatch.setenv("HINDSIGHT_API_ENABLE_IMAGE_RETAIN_API", "false")
        clear_config_cache()
        disabled_retain = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/image-retain",
            files={"files": ("photo.jpg", uploaded, "image/jpeg")},
        )
        assert disabled_retain.status_code == 404

        invalid_status = await api_client.get(
            f"/v1/default/banks/{bank_id}/image-assets",
            params={"status": "unknown"},
        )
        assert invalid_status.status_code == 422
        listed = await api_client.get(f"/v1/default/banks/{bank_id}/image-assets", params={"status": "ready"})
        assert listed.status_code == 200
        assert [item["asset_id"] for item in listed.json()["items"]] == [asset_id]

        content = await api_client.get(f"/v1/default/banks/{bank_id}/image-assets/{quote(asset_id, safe='')}")
        assert content.status_code == 200
        assert content.content == uploaded
        assert content.headers["content-type"].startswith("image/jpeg")
        assert content.headers["x-hindsight-image-sha256"] == hashlib.sha256(uploaded).hexdigest()
        assert "link" not in content.headers

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('image_assets')} SET status = 'failed' WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                asset_id,
            )
        failed_content = await api_client.get(f"/v1/default/banks/{bank_id}/image-assets/{quote(asset_id, safe='')}")
        assert failed_content.status_code == 409
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('image_assets')} SET status = 'deleting' WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                asset_id,
            )
        deleting_content = await api_client.get(f"/v1/default/banks/{bank_id}/image-assets/{quote(asset_id, safe='')}")
        assert deleting_content.status_code == 410
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('image_assets')} SET status = 'ready' WHERE bank_id = $1 AND asset_id = $2",
                bank_id,
                asset_id,
            )

        cross_bank = await api_client.get(f"/v1/default/banks/{other_bank}/image-assets/{quote(asset_id, safe='')}")
        assert cross_bank.status_code == 404

        default_recall = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/recall", json={"query": "preserved photo"}
        )
        assert default_recall.status_code == 200
        assert "image_assets" not in default_recall.json()
        included_recall = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/recall",
            json={"query": "preserved photo", "include": {"image_assets": True}},
        )
        assert included_recall.status_code == 200
        assert included_recall.json()["image_assets"]["document-http-image"][0]["asset_id"] == asset_id

        linked_delete = await api_client.delete(f"/v1/default/banks/{bank_id}/image-assets/{quote(asset_id, safe='')}")
        assert linked_delete.status_code == 409
        assert "delete the document" in linked_delete.json()["detail"]
        still_present = await api_client.get(f"/v1/default/banks/{bank_id}/image-assets/{quote(asset_id, safe='')}")
        assert still_present.status_code == 200

        deleted_document = await api_client.delete(
            f"/v1/default/banks/{bank_id}/documents/{quote(request_payload['document_id'], safe='')}"
        )
        assert deleted_document.status_code == 200
        deleted = await api_client.delete(f"/v1/default/banks/{bank_id}/image-assets/{quote(asset_id, safe='')}")
        assert deleted.status_code == 204
        missing = await api_client.get(f"/v1/default/banks/{bank_id}/image-assets/{quote(asset_id, safe='')}")
        assert missing.status_code == 404
        monkeypatch.setenv("HINDSIGHT_API_ENABLE_IMAGE_RETAIN_API", "true")
        clear_config_cache()
        replay_after_delete = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories/image-retain",
            files=[
                ("files", ("photo.jpg", uploaded, "image/jpeg")),
                ("request", (None, json.dumps(request_payload), "application/json")),
            ],
            headers={"Idempotency-Key": "http-image-key"},
        )
        assert replay_after_delete.status_code == 409
        download_url.assert_not_awaited()
        storage_stat.assert_not_awaited()
    finally:
        clear_config_cache()
        from hindsight_api.models import RequestContext

        context = RequestContext()
        await memory.delete_bank(bank_id, request_context=context)
        await memory.delete_bank(other_bank, request_context=context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "operation"),
    [
        ("/v1/default/banks/hook-bank/memories/image-retain", PrecheckOperation.IMAGE_RETAIN),
        ("/v1/default/banks/hook-bank/document-transfer", PrecheckOperation.IMPORT_DOCUMENTS),
    ],
)
async def test_large_body_prechecks_reject_before_multipart_parsing(
    api_client, memory, monkeypatch, path: str, operation: PrecheckOperation
) -> None:
    from hindsight_api.config import clear_config_cache

    recorder = _ImageHookRecorder(reject_precheck=True)
    original = memory._operation_validator
    memory._operation_validator = recorder
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_IMAGE_RETAIN_API", "true")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_MODEL", "mock-vision")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_DOCUMENT_IMPORT_API", "true")
    clear_config_cache()
    try:
        response = await api_client.post(
            path,
            content=b"this is deliberately not a valid multipart body",
            headers={"content-type": "multipart/form-data; boundary=missing"},
        )
        assert response.status_code == 429
        assert recorder.prechecks[-1].operation == operation
        assert recorder.prechecks[-1].content_length == len(b"this is deliberately not a valid multipart body")
    finally:
        memory._operation_validator = original
        clear_config_cache()


@pytest.mark.asyncio
async def test_image_http_rejects_damaged_mime_mismatch_and_configured_file_limit(api_client, monkeypatch) -> None:
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_ENABLE_IMAGE_RETAIN_API", "true")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_MODEL", "mock-vision")
    clear_config_cache()
    try:
        for filename, content, mime in [
            ("broken.jpg", b"not-an-image", "image/jpeg"),
            ("wrong.jpg", _image_bytes("PNG"), "image/jpeg"),
        ]:
            response = await api_client.post(
                "/v1/default/banks/image-validation/memories/image-retain",
                files={"files": (filename, content, mime)},
            )
            assert response.status_code == 400

        monkeypatch.setenv("HINDSIGHT_API_IMAGE_MAX_FILE_SIZE_MB", "1")
        clear_config_cache()
        oversized = await api_client.post(
            "/v1/default/banks/image-validation/memories/image-retain",
            files={"files": ("large.jpg", b"x" * (1024 * 1024 + 1), "image/jpeg")},
        )
        assert oversized.status_code == 400 and "limit" in oversized.text.lower()
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_version_image_flag_reflects_provider_configuration(api_client, monkeypatch) -> None:
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_ENABLE_IMAGE_RETAIN_API", "true")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_IMAGE_LLM_MODEL", "mock-vision")
    clear_config_cache()
    try:
        features = (await api_client.get("/version")).json()["features"]
        assert features["multimodal_image_input"] is True
        assert "image_asset_management" not in features
        assert "image_asset_recall" not in features
        assert "image_asset_transfer" not in features
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_image_and_transfer_extension_hooks_cover_all_new_operations(
    memory, request_context, monkeypatch
) -> None:
    bank_id = f"test-image-hooks-{uuid.uuid4()}"
    imported_bank = f"test-image-hooks-import-{uuid.uuid4()}"
    document_id = "hook-document"
    asset_id = "hook/asset.jpg"
    recorder = _ImageHookRecorder()
    original = memory._operation_validator
    memory._operation_validator = recorder
    validated = validate_image(_image_bytes("JPEG"), "image/jpeg", max_size_bytes=1024 * 1024)
    monkeypatch.setattr(
        "hindsight_api.engine.memory_engine.create_image_provider",
        lambda _config: MockImageAnalysisProvider(
            ImageProviderConfig(provider="mock", api_key=None, model="mock-vision", base_url=None)
        ),
    )
    monkeypatch.setattr("hindsight_api.engine.memory_engine.validate_image_provider_config", lambda _config: None)
    try:
        accepted = await memory.submit_image_retain(
            bank_id,
            [
                PendingImageAsset(
                    asset_id=asset_id,
                    asset_id_supplied=True,
                    image=validated,
                    context="hook image",
                )
            ],
            ImageRetainParameters(document_id=document_id, content="hook text"),
            request_context=request_context,
        )
        assert accepted.image_assets[0].asset_id == asset_id
        assert [ctx.phase for ctx in recorder.image_retain] == [
            ImageRetainPhase.ADMISSION,
            ImageRetainPhase.EXECUTION,
        ]
        assert len(recorder.image_results) == 1 and recorder.image_results[0].success
        assert recorder.image_results[0].asset_id == asset_id

        await memory.list_image_assets(bank_id, request_context=request_context)
        assert await memory.get_image_asset(bank_id, asset_id, request_context=request_context) is not None
        await memory.recall_async(
            bank_id=bank_id,
            query="hook image",
            request_context=request_context,
            include_image_assets=True,
        )
        await memory.reprocess_document(bank_id, document_id, request_context=request_context)
        archive = await memory.export_documents_async(bank_id, request_context)
        archive_bytes = Path(archive.path).read_bytes()
        archive.cleanup()
        await memory.import_documents_async(imported_bank, archive_bytes, request_context)
        await memory.delete_document(document_id, bank_id, request_context=request_context)
        await memory.delete_image_asset(bank_id, asset_id, request_context=request_context)

        assert {ctx.operation for ctx in recorder.reads} >= {
            BankReadOperation.LIST_IMAGE_ASSETS,
            BankReadOperation.GET_IMAGE_ASSET,
            BankReadOperation.EXPORT_DOCUMENTS,
        }
        assert {ctx.operation for ctx in recorder.writes} >= {
            BankWriteOperation.SUBMIT_IMAGE_RETAIN,
            BankWriteOperation.REPROCESS_DOCUMENT,
            BankWriteOperation.IMPORT_DOCUMENTS,
            BankWriteOperation.DELETE_IMAGE_ASSET,
        }
        assert any(ctx.include_image_assets for ctx in recorder.recalls)
    finally:
        memory._operation_validator = original
        await memory.delete_bank(bank_id, request_context=request_context)
        await memory.delete_bank(imported_bank, request_context=request_context)
