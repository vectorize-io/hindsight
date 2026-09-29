"""Tests for the Fireworks AI batch-inference provider (``FireworksLLM``).

Fireworks' batch API is NOT OpenAI ``/v1/batches``-compatible — it is a
proprietary dataset -> job -> download REST workflow on a control-plane host.
``FireworksLLM`` subclasses ``OpenAICompatibleLLM`` (so online inference reuses
the OpenAI-compatible path) and overrides only the four batch members, mapping
the Fireworks workflow back onto the OpenAI-batch interface contract that the
retain orchestrator + ``fact_extraction`` consumer depend on.

The interface contract that MUST be preserved (see fact_extraction.py:1843):
    result["response"]["body"]["choices"][0]["message"]["content"]

API shapes (endpoints, ``state`` enum, ``jobProgress`` counts, download
endpoint) are grounded in the Fireworks docs. The exact *output JSONL line*
nesting is not verbatim-documented, so the normalizer is defensive and these
tests pin the behavior we rely on; the real shape is confirmed by the
integration/manual path with a live key.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from aiohttp import web

from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.providers.fireworks_llm import FireworksLLM
from tests.aiohttp_stub import stub_server


def _make_fireworks(
    *,
    account_id: str | None = "acct-test",
    max_wait_seconds: int = 86_400,
    batch_base_url: str = "https://api.fireworks.ai",
    model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
) -> FireworksLLM:
    return FireworksLLM(
        provider="fireworks",
        api_key="fw-test-key",
        base_url="",
        model=model,
        reasoning_effort="low",
        account_id=account_id,
        batch_base_url=batch_base_url,
        max_wait_seconds=max_wait_seconds,
    )


# --------------------------------------------------------------------------
# Structural: capability, routing, key requirement, default model
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fireworks_supports_batch_api_is_true():
    llm = _make_fireworks()
    assert await llm.supports_batch_api() is True


def test_wrapper_routes_fireworks_to_fireworks_llm_with_inference_base_url():
    from hindsight_api.engine.llm_wrapper import LLMProvider

    llm = LLMProvider(
        provider="fireworks",
        api_key="fw-test-key",
        base_url="",
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    )

    assert isinstance(llm._provider_impl, FireworksLLM)
    # Online inference uses the OpenAI-compatible Fireworks inference host,
    # which is distinct from the batch control-plane host.
    assert llm._provider_impl.base_url == "https://api.fireworks.ai/inference/v1"


def test_openai_compatible_accepts_fireworks_provider():
    """Subclassing requires the parent's ``valid_providers`` to accept fireworks."""
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        provider="fireworks",
        api_key="fw-test-key",
        base_url="",
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    )
    assert llm.base_url == "https://api.fireworks.ai/inference/v1"


def test_fireworks_requires_api_key():
    from hindsight_api.engine.llm_wrapper import requires_api_key

    assert requires_api_key("fireworks") is True


def test_fireworks_has_default_model():
    from hindsight_api.config import PROVIDER_DEFAULT_MODELS

    assert PROVIDER_DEFAULT_MODELS["fireworks"] == "accounts/fireworks/models/llama-v3p1-8b-instruct"


# --------------------------------------------------------------------------
# §6.5 input translation: OpenAI request -> Fireworks JSONL (drop method/url)
# --------------------------------------------------------------------------


def test_translate_requests_drops_method_and_url_keeps_custom_id_and_body():
    requests = [
        {
            "custom_id": "chunk_0",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        },
        {
            "custom_id": "chunk_1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "m", "messages": [{"role": "user", "content": "bye"}]},
        },
    ]

    jsonl = FireworksLLM._translate_requests(requests)
    lines = [json.loads(line) for line in jsonl.strip().split("\n")]

    assert len(lines) == 2
    for line, original in zip(lines, requests):
        assert set(line.keys()) == {"custom_id", "body"}
        assert "method" not in line
        assert "url" not in line
        assert line["custom_id"] == original["custom_id"]
        assert line["body"] == original["body"]


# --------------------------------------------------------------------------
# §6.4 status normalization: Fireworks state -> orchestrator status string
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fw_state, expected",
    [
        ("JOB_STATE_COMPLETED", "completed"),
        ("COMPLETED", "completed"),
        ("JOB_STATE_FAILED", "failed"),
        ("FAILED", "failed"),
        ("JOB_STATE_CANCELLED", "cancelled"),
        ("CANCELED", "cancelled"),
        ("EXPIRED", "expired"),
        ("JOB_STATE_RUNNING", "in_progress"),
        ("RUNNING", "in_progress"),
        ("JOB_STATE_PENDING", "in_progress"),
        ("VALIDATING", "in_progress"),
        ("JOB_STATE_CREATING", "in_progress"),
        ("JOB_STATE_UNSPECIFIED", "in_progress"),
        ("", "in_progress"),
    ],
)
def test_normalize_state_table(fw_state, expected):
    assert FireworksLLM._normalize_state(fw_state) == expected


# --------------------------------------------------------------------------
# §6.6 output normalization: Fireworks output -> OpenAI-batch-output shape
# --------------------------------------------------------------------------


def test_normalize_output_line_wraps_response_under_body():
    """Fireworks puts the chat completion at ``response`` directly; the consumer
    reads ``result["response"]["body"]["choices"]`` so we must re-nest it."""
    fw_line = {
        "custom_id": "chunk_0",
        "response": {
            "id": "chatcmpl-abc",
            "choices": [{"message": {"role": "assistant", "content": '{"facts": []}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        "error": None,
    }

    out = FireworksLLM._normalize_output_line(fw_line)

    assert out["custom_id"] == "chunk_0"
    assert out["error"] is None
    # The exact accessor the retain consumer uses:
    assert out["response"]["body"]["choices"][0]["message"]["content"] == '{"facts": []}'
    assert out["response"]["body"]["usage"]["total_tokens"] == 15


def test_normalize_output_line_handles_already_body_nested():
    """Defensive: if Fireworks ever nests under response.body, don't double-wrap."""
    fw_line = {
        "custom_id": "chunk_1",
        "response": {"body": {"choices": [{"message": {"content": "ok"}}]}},
    }

    out = FireworksLLM._normalize_output_line(fw_line)

    assert out["response"]["body"]["choices"][0]["message"]["content"] == "ok"


def test_normalize_output_line_surfaces_errors():
    """A failed request (error-file line) must yield a truthy ``error`` so the
    consumer's ``result.get("error")`` branch fires instead of vanishing."""
    fw_line = {
        "custom_id": "chunk_2",
        "response": None,
        "error": {"code": "model_error", "message": "boom"},
    }

    out = FireworksLLM._normalize_output_line(fw_line)

    assert out["custom_id"] == "chunk_2"
    assert out["error"] == {"code": "model_error", "message": "boom"}


# --------------------------------------------------------------------------
# submit_batch: dataset create -> upload -> job create (stub control plane)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_batch_runs_dataset_upload_job_workflow():
    paths: list[str] = []
    upload_body: list[str] = []
    upload_content_type: list[str] = []
    job_body: list[dict] = []
    dataset_body: list[dict] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        path = request.path
        paths.append(f"{request.method} {path}")
        assert request.headers["authorization"] == "Bearer fw-test-key"

        if request.method == "POST" and path.endswith("/datasets"):
            body = await request.json()
            dataset_body.append(body)
            return web.json_response({"name": f"accounts/acct-test/datasets/{body['datasetId']}"})
        if request.method == "POST" and path.endswith(":upload"):
            upload_content_type.append(request.content_type)
            upload_body.append((await request.read()).decode("utf-8", errors="replace"))
            return web.json_response({})
        if request.method == "POST" and path.endswith("/batchInferenceJobs"):
            job_body.append(await request.json())
            return web.json_response(
                {
                    "name": "accounts/acct-test/batchInferenceJobs/job-xyz",
                    "state": "JOB_STATE_CREATING",
                    "createTime": "2026-05-28T00:00:00Z",
                }
            )
        raise AssertionError(f"unexpected request: {request.method} {path}")

    requests = [
        {
            "custom_id": "chunk_0",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "m", "messages": []},
        },
    ]
    async with stub_server(handler) as base_url:
        llm = _make_fireworks(batch_base_url=base_url)
        result = await llm.submit_batch(requests)
        await llm.cleanup()

    # batch_id is the Fireworks jobId, used as the polling handle.
    assert result["batch_id"]
    assert result["request_count"] == 1

    # All control-plane calls are account-scoped.
    assert all("/v1/accounts/acct-test/" in p for p in paths)
    # Workflow order: create dataset, upload, create job.
    assert paths[0].endswith("/datasets")
    assert ":upload" in paths[1]
    assert paths[2].endswith("/batchInferenceJobs")

    # The input file is a multipart upload.
    assert upload_content_type[0] == "multipart/form-data"
    # Uploaded JSONL is Fireworks-shaped (no method/url leaked through).
    assert "custom_id" in upload_body[0]
    assert '"method"' not in upload_body[0]
    assert '"url"' not in upload_body[0]

    # Job references the configured model.
    assert job_body[0]["model"] == "accounts/fireworks/models/llama-v3p1-8b-instruct"

    # Dataset create declares format + exampleCount (Fireworks rejects uploaded
    # datasets without example_count; it's the JSONL line count, as a string).
    assert dataset_body[0]["dataset"]["format"] == "CHAT"
    assert dataset_body[0]["dataset"]["exampleCount"] == str(len(requests))


@pytest.mark.asyncio
async def test_submit_batch_without_account_id_fails_fast():
    llm = _make_fireworks(account_id=None)
    with pytest.raises(ValueError, match="account.id"):
        await llm.submit_batch([{"custom_id": "c0", "method": "POST", "url": "/v1/chat/completions", "body": {}}])


@pytest.mark.asyncio
async def test_api_errors_surface_the_response_body():
    """A 4xx must include Fireworks' error body in the raised error — otherwise a
    malformed dataset/job request is undebuggable (raise_for_status drops it)."""

    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response({"error": "invalid field 'userUploaded' in dataset"}, status=400)

    async with stub_server(handler) as base_url:
        llm = _make_fireworks(batch_base_url=base_url)
        with pytest.raises(UpstreamHTTPError, match="Fireworks API 400 for POST .*invalid field 'userUploaded'") as exc:
            await llm.submit_batch([{"custom_id": "c0", "method": "POST", "url": "/v1/chat/completions", "body": {}}])
        await llm.cleanup()

    # status_code is what remote_retry classifies on.
    assert exc.value.status_code == 400


# --------------------------------------------------------------------------
# get_batch_status: state + counts mapping, and the PENDING-forever timeout
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_batch_status_maps_state_and_counts():
    async def handler(request: web.Request) -> web.StreamResponse:
        assert request.method == "GET"
        assert request.path == "/v1/accounts/acct-test/batchInferenceJobs/job-xyz"
        return web.json_response(
            {
                "name": "accounts/acct-test/batchInferenceJobs/job-xyz",
                "state": "JOB_STATE_COMPLETED",
                "createTime": "2026-05-28T00:00:00Z",
                "outputDatasetId": "accounts/acct-test/datasets/out-1",
                "jobProgress": {
                    "totalInputRequests": "2",
                    "successfullyProcessedRequests": "2",
                    "failedRequests": "0",
                },
            }
        )

    async with stub_server(handler) as base_url:
        llm = _make_fireworks(batch_base_url=base_url)
        status = await llm.get_batch_status("job-xyz")
        await llm.cleanup()

    assert status["batch_id"] == "job-xyz"
    assert status["status"] == "completed"
    assert status["request_counts"]["total"] == 2
    assert status["request_counts"]["completed"] == 2
    assert status["request_counts"]["failed"] == 0


@pytest.mark.asyncio
async def test_get_batch_status_times_out_when_stuck_pending():
    """The Fireworks gotcha: a non-batch-eligible model leaves the job PENDING
    forever. The shared poll loop is ``while True`` with no max-wait, so the
    provider must surface a terminal status once createTime is too old."""
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response(
            {
                "name": "accounts/acct-test/batchInferenceJobs/job-stuck",
                "state": "JOB_STATE_PENDING",
                "createTime": stale,
                "jobProgress": {"totalInputRequests": "1"},
            }
        )

    async with stub_server(handler) as base_url:
        llm = _make_fireworks(batch_base_url=base_url, max_wait_seconds=60)
        status = await llm.get_batch_status("job-stuck")
        await llm.cleanup()

    # Driver treats expired/failed/cancelled as fatal and raises — no infinite poll.
    assert status["status"] in ("expired", "failed")
    assert status.get("errors")


# --------------------------------------------------------------------------
# retrieve_batch_results: download + normalize + merge separate error file
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_batch_results_normalizes_and_merges_error_file():
    results_jsonl = json.dumps(
        {
            "custom_id": "chunk_0",
            "response": {
                "choices": [{"message": {"content": '{"facts": [{"what": "x"}]}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            "error": None,
        }
    )
    errors_jsonl = json.dumps({"custom_id": "chunk_1", "response": None, "error": {"code": "oops", "message": "bad"}})
    # A pre-signed query string: percent-escapes must reach the server untouched.
    signature = "X-Goog-Signature=ab%2Fcd%3D%3D"
    download_auth: list[str | None] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        path = request.path
        if path.endswith("/batchInferenceJobs/job-done"):
            return web.json_response(
                {
                    "state": "JOB_STATE_COMPLETED",
                    "createTime": "2026-05-28T00:00:00Z",
                    "outputDatasetId": "accounts/acct-test/datasets/out-1",
                }
            )
        if path.endswith(":getDownloadEndpoint"):
            base = f"{request.scheme}://{request.host}"
            return web.json_response(
                {
                    "filenameToSignedUrls": {
                        "results.jsonl": f"{base}/signed/results.jsonl?{signature}",
                        "errors.jsonl": f"{base}/signed/errors.jsonl?{signature}",
                    }
                }
            )
        if path.startswith("/signed/"):
            assert request.raw_path.partition("?")[2] == signature
            download_auth.append(request.headers.get("Authorization"))
            return web.Response(text=results_jsonl if path.endswith("results.jsonl") else errors_jsonl)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async with stub_server(handler) as base_url:
        llm = _make_fireworks(batch_base_url=base_url)
        results = await llm.retrieve_batch_results("job-done")
        await llm.cleanup()
    by_id = {r["custom_id"]: r for r in results}

    assert set(by_id) == {"chunk_0", "chunk_1"}
    # Success line normalized to the consumer's accessor.
    assert by_id["chunk_0"]["response"]["body"]["choices"][0]["message"]["content"] == '{"facts": [{"what": "x"}]}'
    assert not by_id["chunk_0"].get("error")
    # Error-file line merged in as a per-custom_id error (not dropped).
    assert by_id["chunk_1"]["error"] == {"code": "oops", "message": "bad"}
    # Signed URLs are pre-authenticated: the bearer token is not sent to them.
    assert download_auth == [None, None]


@pytest.mark.asyncio
async def test_retrieve_batch_results_raises_when_not_completed():
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.json_response({"state": "JOB_STATE_RUNNING", "createTime": "2026-05-28T00:00:00Z"})

    async with stub_server(handler) as base_url:
        llm = _make_fireworks(batch_base_url=base_url)
        with pytest.raises(ValueError, match="not completed"):
            await llm.retrieve_batch_results("job-running")
        await llm.cleanup()


# --------------------------------------------------------------------------
# Config: account id / batch base url / max wait from env
# --------------------------------------------------------------------------


def test_config_reads_fireworks_settings_from_env(monkeypatch):
    from hindsight_api.config import HindsightConfig, clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_FIREWORKS_ACCOUNT_ID", "acct-from-env")
    monkeypatch.setenv("HINDSIGHT_API_FIREWORKS_BATCH_BASE_URL", "https://batch.example")
    monkeypatch.setenv("HINDSIGHT_API_FIREWORKS_BATCH_MAX_WAIT_SECONDS", "120")
    clear_config_cache()
    try:
        config = HindsightConfig.from_env()
        assert config.fireworks_account_id == "acct-from-env"
        assert config.fireworks_batch_base_url == "https://batch.example"
        assert config.fireworks_batch_max_wait_seconds == 120
    finally:
        clear_config_cache()


def test_config_fireworks_defaults(monkeypatch):
    from hindsight_api.config import HindsightConfig, clear_config_cache

    monkeypatch.delenv("HINDSIGHT_API_FIREWORKS_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_FIREWORKS_BATCH_BASE_URL", raising=False)
    clear_config_cache()
    try:
        config = HindsightConfig.from_env()
        assert config.fireworks_account_id is None
        assert config.fireworks_batch_base_url == "https://api.fireworks.ai"
    finally:
        clear_config_cache()
