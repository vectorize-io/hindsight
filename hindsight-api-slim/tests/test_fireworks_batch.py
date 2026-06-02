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

import httpx
import pytest

from hindsight_api.engine.providers.fireworks_llm import FireworksLLM


def _make_fireworks(
    *,
    account_id: str | None = "acct-test",
    http_client: httpx.AsyncClient | None = None,
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
        http_client=http_client,
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
# submit_batch: dataset create -> upload -> job create (httpx MockTransport)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_batch_runs_dataset_upload_job_workflow():
    paths: list[str] = []
    upload_body: list[str] = []
    job_body: list[dict] = []
    dataset_body: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        paths.append(f"{request.method} {path}")
        assert request.headers["authorization"] == "Bearer fw-test-key"

        if request.method == "POST" and path.endswith("/datasets"):
            body = json.loads(request.content)
            dataset_body.append(body)
            return httpx.Response(200, json={"name": f"accounts/acct-test/datasets/{body['datasetId']}"})
        if request.method == "POST" and path.endswith(":upload"):
            upload_body.append(request.content.decode("utf-8", errors="replace"))
            return httpx.Response(200, json={})
        if request.method == "POST" and path.endswith("/batchInferenceJobs"):
            job_body.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "name": "accounts/acct-test/batchInferenceJobs/job-xyz",
                    "state": "JOB_STATE_CREATING",
                    "createTime": "2026-05-28T00:00:00Z",
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = _make_fireworks(http_client=client)

    requests = [
        {
            "custom_id": "chunk_0",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "m", "messages": []},
        },
    ]
    result = await llm.submit_batch(requests)

    # batch_id is the Fireworks jobId, used as the polling handle.
    assert result["batch_id"]
    assert result["request_count"] == 1

    # All control-plane calls are account-scoped.
    assert all("/v1/accounts/acct-test/" in p for p in paths)
    # Workflow order: create dataset, upload, create job.
    assert paths[0].endswith("/datasets")
    assert ":upload" in paths[1]
    assert paths[2].endswith("/batchInferenceJobs")

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

    await client.aclose()


@pytest.mark.asyncio
async def test_submit_batch_without_account_id_fails_fast():
    llm = _make_fireworks(account_id=None)
    with pytest.raises(ValueError, match="account.id"):
        await llm.submit_batch([{"custom_id": "c0", "method": "POST", "url": "/v1/chat/completions", "body": {}}])


@pytest.mark.asyncio
async def test_api_errors_surface_the_response_body():
    """A 4xx must include Fireworks' error body in the raised error — otherwise a
    malformed dataset/job request is undebuggable (raise_for_status drops it)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid field 'userUploaded' in dataset"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = _make_fireworks(http_client=client)

    with pytest.raises(httpx.HTTPStatusError, match="invalid field 'userUploaded'"):
        await llm.submit_batch(
            [{"custom_id": "c0", "method": "POST", "url": "/v1/chat/completions", "body": {}}]
        )

    await client.aclose()


# --------------------------------------------------------------------------
# get_batch_status: state + counts mapping, and the PENDING-forever timeout
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_batch_status_maps_state_and_counts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/accounts/acct-test/batchInferenceJobs/job-xyz"
        return httpx.Response(
            200,
            json={
                "name": "accounts/acct-test/batchInferenceJobs/job-xyz",
                "state": "JOB_STATE_COMPLETED",
                "createTime": "2026-05-28T00:00:00Z",
                "outputDatasetId": "accounts/acct-test/datasets/out-1",
                "jobProgress": {
                    "totalInputRequests": "2",
                    "successfullyProcessedRequests": "2",
                    "failedRequests": "0",
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = _make_fireworks(http_client=client)

    status = await llm.get_batch_status("job-xyz")

    assert status["batch_id"] == "job-xyz"
    assert status["status"] == "completed"
    assert status["request_counts"]["total"] == 2
    assert status["request_counts"]["completed"] == 2
    assert status["request_counts"]["failed"] == 0

    await client.aclose()


@pytest.mark.asyncio
async def test_get_batch_status_times_out_when_stuck_pending():
    """The Fireworks gotcha: a non-batch-eligible model leaves the job PENDING
    forever. The shared poll loop is ``while True`` with no max-wait, so the
    provider must surface a terminal status once createTime is too old."""
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "name": "accounts/acct-test/batchInferenceJobs/job-stuck",
                "state": "JOB_STATE_PENDING",
                "createTime": stale,
                "jobProgress": {"totalInputRequests": "1"},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = _make_fireworks(http_client=client, max_wait_seconds=60)

    status = await llm.get_batch_status("job-stuck")

    # Driver treats expired/failed/cancelled as fatal and raises — no infinite poll.
    assert status["status"] in ("expired", "failed")
    assert status.get("errors")

    await client.aclose()


# --------------------------------------------------------------------------
# retrieve_batch_results: download + normalize + merge separate error file
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_batch_results_normalizes_and_merges_error_file():
    results_url = "https://signed.example/results.jsonl"
    errors_url = "https://signed.example/errors.jsonl"

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

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/batchInferenceJobs/job-done"):
            return httpx.Response(
                200,
                json={
                    "state": "JOB_STATE_COMPLETED",
                    "createTime": "2026-05-28T00:00:00Z",
                    "outputDatasetId": "accounts/acct-test/datasets/out-1",
                },
            )
        if path.endswith(":getDownloadEndpoint"):
            return httpx.Response(
                200,
                json={"filenameToSignedUrls": {"results.jsonl": results_url, "errors.jsonl": errors_url}},
            )
        if str(request.url) == results_url:
            return httpx.Response(200, text=results_jsonl)
        if str(request.url) == errors_url:
            return httpx.Response(200, text=errors_jsonl)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = _make_fireworks(http_client=client)

    results = await llm.retrieve_batch_results("job-done")
    by_id = {r["custom_id"]: r for r in results}

    assert set(by_id) == {"chunk_0", "chunk_1"}
    # Success line normalized to the consumer's accessor.
    assert by_id["chunk_0"]["response"]["body"]["choices"][0]["message"]["content"] == '{"facts": [{"what": "x"}]}'
    assert not by_id["chunk_0"].get("error")
    # Error-file line merged in as a per-custom_id error (not dropped).
    assert by_id["chunk_1"]["error"] == {"code": "oops", "message": "bad"}

    await client.aclose()


@pytest.mark.asyncio
async def test_retrieve_batch_results_raises_when_not_completed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"state": "JOB_STATE_RUNNING", "createTime": "2026-05-28T00:00:00Z"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = _make_fireworks(http_client=client)

    with pytest.raises(ValueError, match="not completed"):
        await llm.retrieve_batch_results("job-running")

    await client.aclose()


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
