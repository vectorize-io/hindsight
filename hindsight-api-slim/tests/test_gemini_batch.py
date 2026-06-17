"""Tests for the Gemini Batch API provider path (``GeminiLLM`` batch overrides).

Google's Gemini Batch API gives a flat 50% input+output discount with a 24h SLA
(https://ai.google.dev/gemini-api/docs/batch-api). ``GeminiLLM`` extends
``LLMInterface`` directly (not the OpenAI-compatible base), so it overrides the
four batch members and translates Gemini's file-upload -> ``batches.create`` ->
``batches.get`` -> download flow back onto the OpenAI-batch interface contract
that the retain orchestrator + ``fact_extraction`` consumer depend on.

The interface contract that MUST be preserved (see fact_extraction.py)::
    result["response"]["body"]["choices"][0]["message"]["content"]

The pure translation/normalization helpers are unit-tested directly; the async
submit/status/retrieve flow is tested against a fake genai client. A live key is
only needed for the (separate) end-to-end path.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from hindsight_api.engine.providers.gemini_llm import GeminiLLM


def _make_gemini(model: str = "gemini-2.5-flash") -> GeminiLLM:
    # Patch the genai client constructor so no real credentials/network are used;
    # the batch tests swap in a fake aio client below.
    with patch("hindsight_api.engine.providers.gemini_llm.genai.Client", MagicMock()):
        return GeminiLLM(provider="gemini", api_key="test-key", base_url="", model=model)


def _openai_request(custom_id: str, *, strict: bool = True) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "Extract facts."},
                {"role": "user", "content": "Paris is the capital of France."},
            ],
            "temperature": 0.1,
            "max_completion_tokens": 2048,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "facts",
                    "schema": {"type": "object", "properties": {"facts": {"type": "array"}}},
                    "strict": strict,
                },
            },
        },
    }


# --------------------------------------------------------------------------
# Structural: capability flag (gemini yes, vertexai no)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_supports_batch_api_is_true():
    llm = _make_gemini()
    assert await llm.supports_batch_api() is True


@pytest.mark.asyncio
async def test_vertexai_does_not_support_batch_api():
    # Vertex AI's batch path is GCS/BigQuery-backed (no file upload), so it stays
    # unsupported — the startup validation then raises a clear error.
    llm = _make_gemini()
    llm.provider = "vertexai"
    assert await llm.supports_batch_api() is False


# --------------------------------------------------------------------------
# Pure translation: OpenAI body -> Gemini GenerateContentRequest
# --------------------------------------------------------------------------


def test_translate_requests_builds_keyed_jsonl():
    jsonl = GeminiLLM._translate_requests([_openai_request("chunk_0"), _openai_request("chunk_1")])
    lines = jsonl.split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["key"] == "chunk_0"
    assert set(first["request"].keys()) == {"contents", "systemInstruction", "generationConfig"}


def test_body_translation_maps_roles_and_generation_config():
    req = GeminiLLM._openai_body_to_gemini_request(_openai_request("c")["body"])

    # system -> systemInstruction; user -> contents(role=user)
    assert req["contents"] == [{"role": "user", "parts": [{"text": "Paris is the capital of France."}]}]
    assert req["systemInstruction"]["parts"][0]["text"].startswith("Extract facts.")

    gc = req["generationConfig"]
    assert gc["temperature"] == 0.1
    assert gc["maxOutputTokens"] == 2048
    assert gc["responseMimeType"] == "application/json"
    # strict=True -> grammar-enforced via responseJsonSchema
    assert gc["responseJsonSchema"] == {"type": "object", "properties": {"facts": {"type": "array"}}}
    # schema is also appended as a textual hint (mirrors the sync call path)
    assert "valid JSON matching this schema" in req["systemInstruction"]["parts"][0]["text"]


def test_body_translation_omits_response_json_schema_when_not_strict():
    req = GeminiLLM._openai_body_to_gemini_request(_openai_request("c", strict=False)["body"])
    gc = req["generationConfig"]
    # Non-strict still forces JSON output, but does not grammar-enforce the schema
    assert gc["responseMimeType"] == "application/json"
    assert "responseJsonSchema" not in gc


def test_assistant_role_maps_to_model():
    body = {"messages": [{"role": "assistant", "content": "prior turn"}]}
    req = GeminiLLM._openai_body_to_gemini_request(body)
    assert req["contents"] == [{"role": "model", "parts": [{"text": "prior turn"}]}]
    assert "systemInstruction" not in req


# --------------------------------------------------------------------------
# Pure normalization: Gemini output line -> OpenAI-batch-output shape
# --------------------------------------------------------------------------


def test_normalize_output_line_success_preserves_contract():
    line = {
        "key": "chunk_0",
        "response": {"candidates": [{"content": {"parts": [{"text": '{"facts": []}'}]}}]},
    }
    out = GeminiLLM._normalize_output_line(line)
    assert out["custom_id"] == "chunk_0"
    assert out["error"] is None
    # The exact path fact_extraction.py reads:
    assert out["response"]["body"]["choices"][0]["message"]["content"] == '{"facts": []}'


def test_normalize_output_line_concatenates_multiple_text_parts():
    line = {
        "key": "chunk_0",
        "response": {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]},
    }
    out = GeminiLLM._normalize_output_line(line)
    assert out["response"]["body"]["choices"][0]["message"]["content"] == "ab"


def test_normalize_output_line_includes_translated_usage():
    line = {
        "key": "chunk_0",
        "response": {
            "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30, "totalTokenCount": 150},
        },
    }
    out = GeminiLLM._normalize_output_line(line)
    # OpenAI-shaped usage block the batch consumer accumulates (was missing -> usage=0).
    assert out["response"]["body"]["usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }


def test_normalize_output_line_omits_usage_when_absent():
    line = {"key": "chunk_0", "response": {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}}
    out = GeminiLLM._normalize_output_line(line)
    assert "usage" not in out["response"]["body"]


def test_normalize_output_line_error_surfaces_per_key():
    line = {"key": "chunk_1", "error": {"code": 400, "message": "bad request"}}
    out = GeminiLLM._normalize_output_line(line)
    assert out["custom_id"] == "chunk_1"
    assert out["response"] is None
    assert out["error"] == {"code": 400, "message": "bad request"}


def test_extract_text_from_empty_response_is_empty_string():
    assert GeminiLLM._extract_text_from_response({}) == ""
    assert GeminiLLM._extract_text_from_response({"candidates": []}) == ""


# --------------------------------------------------------------------------
# Pure status mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [
        ("JOB_STATE_SUCCEEDED", "completed"),
        ("JOB_STATE_PARTIALLY_SUCCEEDED", "completed"),
        ("JOB_STATE_FAILED", "failed"),
        ("JOB_STATE_CANCELLED", "cancelled"),
        ("JOB_STATE_CANCELLING", "cancelled"),
        ("JOB_STATE_EXPIRED", "expired"),
        ("JOB_STATE_RUNNING", "in_progress"),
        ("JOB_STATE_PENDING", "in_progress"),
        ("JOB_STATE_QUEUED", "in_progress"),
        ("JOB_STATE_UNSPECIFIED", "in_progress"),
    ],
)
def test_normalize_state(state, expected):
    assert GeminiLLM._normalize_state(state) == expected


def test_normalize_state_handles_enum_and_qualified_string():
    assert GeminiLLM._normalize_state(SimpleNamespace(name="JOB_STATE_SUCCEEDED")) == "completed"
    assert GeminiLLM._normalize_state("JobState.JOB_STATE_RUNNING") == "in_progress"
    assert GeminiLLM._normalize_state(None) == "in_progress"


# --------------------------------------------------------------------------
# Async flow: submit -> status -> retrieve against a fake genai client
# --------------------------------------------------------------------------


def _fake_client(*, get_batch, download_text: bytes | None = None) -> MagicMock:
    """Build a fake ``client.aio`` namespace covering the batch flow."""
    aio = MagicMock()
    aio.files.upload = AsyncMock(return_value=SimpleNamespace(name="files/uploaded-123"))
    aio.batches.create = AsyncMock(return_value=SimpleNamespace(name="batches/abc", state="JOB_STATE_PENDING"))
    aio.batches.get = AsyncMock(return_value=get_batch)
    if download_text is not None:
        aio.files.download = AsyncMock(return_value=download_text)
    client = MagicMock()
    client.aio = aio
    return client


@pytest.mark.asyncio
async def test_submit_batch_uploads_and_creates_job():
    llm = _make_gemini()
    llm._client = _fake_client(get_batch=None)

    meta = await llm.submit_batch([_openai_request("chunk_0"), _openai_request("chunk_1")])

    assert meta["batch_id"] == "batches/abc"
    assert meta["status"] == "in_progress"  # JOB_STATE_PENDING
    assert meta["request_count"] == 2

    # Uploaded JSONL had one line per request, and the job used the model + file.
    upload_kwargs = llm._client.aio.files.upload.call_args.kwargs
    uploaded_bytes = upload_kwargs["file"].getvalue().decode("utf-8")
    assert len(uploaded_bytes.strip().split("\n")) == 2
    assert upload_kwargs["config"].mime_type == "jsonl"

    create_kwargs = llm._client.aio.batches.create.call_args.kwargs
    assert create_kwargs["model"] == "gemini-2.5-flash"
    assert create_kwargs["src"] == "files/uploaded-123"


@pytest.mark.asyncio
async def test_get_batch_status_maps_counts_and_output_file():
    batch = SimpleNamespace(
        name="batches/abc",
        state="JOB_STATE_SUCCEEDED",
        completion_stats=SimpleNamespace(successful_count=3, failed_count=1, incomplete_count=0),
        dest=SimpleNamespace(file_name="files/output-999"),
        error=None,
    )
    llm = _make_gemini()
    llm._client = _fake_client(get_batch=batch)

    status = await llm.get_batch_status("batches/abc")
    assert status["status"] == "completed"
    assert status["request_counts"] == {"total": 4, "completed": 3, "failed": 1}
    assert status["output_file_id"] == "files/output-999"


@pytest.mark.asyncio
async def test_retrieve_batch_results_downloads_and_normalizes():
    output_lines = [
        {"key": "chunk_0", "response": {"candidates": [{"content": {"parts": [{"text": '{"facts": []}'}]}}]}},
        {"key": "chunk_1", "error": {"code": 500, "message": "boom"}},
    ]
    download_text = ("\n".join(json.dumps(line) for line in output_lines)).encode("utf-8")
    batch = SimpleNamespace(
        name="batches/abc",
        state="JOB_STATE_SUCCEEDED",
        completion_stats=None,
        dest=SimpleNamespace(file_name="files/output-999"),
        error=None,
    )
    llm = _make_gemini()
    llm._client = _fake_client(get_batch=batch, download_text=download_text)

    results = await llm.retrieve_batch_results("batches/abc")
    by_id = {r["custom_id"]: r for r in results}
    assert by_id["chunk_0"]["response"]["body"]["choices"][0]["message"]["content"] == '{"facts": []}'
    assert by_id["chunk_1"]["error"] == {"code": 500, "message": "boom"}

    llm._client.aio.files.download.assert_awaited_once()
    assert llm._client.aio.files.download.call_args.kwargs["file"] == "files/output-999"


@pytest.mark.asyncio
async def test_retrieve_batch_results_raises_when_not_completed():
    batch = SimpleNamespace(name="batches/abc", state="JOB_STATE_RUNNING", completion_stats=None, dest=None, error=None)
    llm = _make_gemini()
    llm._client = _fake_client(get_batch=batch)
    with pytest.raises(ValueError, match="not completed"):
        await llm.retrieve_batch_results("batches/abc")
