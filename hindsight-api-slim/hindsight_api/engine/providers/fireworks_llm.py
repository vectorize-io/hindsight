"""Fireworks AI provider with batch-inference support.

Fireworks' *online* inference endpoint (``/inference/v1``) is OpenAI-compatible,
so ``FireworksLLM`` subclasses :class:`OpenAICompatibleLLM` and reuses its entire
chat path. Only the *batch* mechanism differs: Fireworks does NOT implement the
OpenAI ``/v1/batches`` API. Instead it exposes a proprietary, account-scoped
dataset -> job -> download REST workflow on a separate control-plane host. This
class overrides only the four batch members of the interface, translating that
workflow to/from the OpenAI-batch shapes the retain orchestrator and
``fact_extraction`` consumer expect — so nothing downstream changes.

Interface contract preserved (see ``fact_extraction.py`` result handling)::

    result["response"]["body"]["choices"][0]["message"]["content"]

Workflow (control-plane host, e.g. ``https://api.fireworks.ai``)::

    POST /v1/accounts/{acct}/datasets                      create input dataset
    POST /v1/accounts/{acct}/datasets/{id}:upload          upload input JSONL
    POST /v1/accounts/{acct}/batchInferenceJobs            create job
    GET  /v1/accounts/{acct}/batchInferenceJobs/{jobId}    poll status
    GET  /v1/accounts/{acct}/datasets/{out}:getDownloadEndpoint   signed URLs
    GET  <signed-url>                                      download output JSONL

NOTE: the exact *output JSONL line* nesting is not verbatim-documented by
Fireworks. ``_normalize_output_line`` handles both the observed shape
(``{custom_id, response: {...completion...}, error}``) and a ``response.body``
nesting defensively. Confirm against a live key via the integration path.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from .openai_compatible_llm import OpenAICompatibleLLM

logger = logging.getLogger(__name__)

# Normalized statuses the retain driver treats as fatal (it raises) vs. keeps
# polling on. "completed" ends the poll; anything else not in this set means
# "keep polling".
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})

# Default per-request timeout for control-plane HTTP calls (not the job wait).
_HTTP_TIMEOUT_SECONDS = 60.0

# Fallback max job wait if neither a constructor arg nor config supplies one
# (24h matches Fireworks' maximum job timeout).
_DEFAULT_MAX_WAIT_SECONDS = 86_400


class FireworksLLM(OpenAICompatibleLLM):
    """Fireworks provider: OpenAI-compatible online inference + native batch."""

    def __init__(
        self,
        provider: str = "fireworks",
        *,
        api_key: str,
        base_url: str = "",
        model: str,
        reasoning_effort: str = "low",
        account_id: str | None = None,
        batch_base_url: str | None = None,
        max_wait_seconds: int | None = None,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

        # Batch settings are static, server-level config. Resolve any unset
        # values from the global config lazily so the online inference path
        # works even when batch is never configured.
        if account_id is None or batch_base_url is None or max_wait_seconds is None:
            from ...config import get_config

            cfg = get_config()
            if account_id is None:
                account_id = cfg.fireworks_account_id
            if batch_base_url is None:
                batch_base_url = cfg.fireworks_batch_base_url
            if max_wait_seconds is None:
                max_wait_seconds = cfg.fireworks_batch_max_wait_seconds

        self._account_id = account_id
        self._batch_base_url = (batch_base_url or "https://api.fireworks.ai").rstrip("/")
        self._max_wait_seconds: int = (
            int(max_wait_seconds) if max_wait_seconds is not None else _DEFAULT_MAX_WAIT_SECONDS
        )
        self._http_client = http_client
        self._owns_http_client = http_client is None

    # ----- interface: batch members -------------------------------------

    async def supports_batch_api(self) -> bool:
        return True

    async def submit_batch(
        self,
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
    ) -> dict[str, Any]:
        # endpoint/completion_window are part of the LLMInterface batch contract
        # (used by the OpenAI path) but have no analogue in Fireworks' job API:
        # the request shape is fixed (chat) and the job timeout is server-side.
        # Kept for signature compatibility with the shared retain driver.
        self._require_account_id()
        logger.info(f"Submitting Fireworks batch with {len(requests)} requests")

        jsonl = self._translate_requests(requests)
        input_dataset_id = f"hs-batch-in-{uuid.uuid4().hex}"
        output_dataset_id = f"hs-batch-out-{uuid.uuid4().hex}"
        headers = self._auth_headers()

        # The `dataset` resource takes format + exampleCount on create. CHAT is
        # the format for chat-completion batch input; exampleCount is the JSONL
        # line count (Fireworks rejects uploaded datasets without it) and is an
        # int64 proto field, so it goes over the wire as a string.
        await self._request(
            "POST",
            self._datasets_url(),
            headers=headers,
            json={
                "datasetId": input_dataset_id,
                "dataset": {"format": "CHAT", "exampleCount": str(len(requests))},
            },
        )

        await self._request(
            "POST",
            f"{self._datasets_url()}/{input_dataset_id}:upload",
            headers=headers,
            files={"file": ("batch_input.jsonl", jsonl.encode("utf-8"), "application/jsonl")},
        )

        job_resp = await self._request(
            "POST",
            self._jobs_url(),
            headers=headers,
            json={
                "model": self.model,
                "inputDatasetId": self._dataset_resource(input_dataset_id),
                "outputDatasetId": self._dataset_resource(output_dataset_id),
            },
        )
        job = job_resp.json()
        job_id = self._last_segment(job.get("name")) or output_dataset_id

        logger.info(f"Fireworks batch job submitted: {job_id}, state={job.get('state')}")

        return {
            "batch_id": job_id,
            "status": self._normalize_state(job.get("state", "")),
            "input_dataset_id": input_dataset_id,
            "output_dataset_id": output_dataset_id,
            "created_at": job.get("createTime"),
            "request_count": len(requests),
        }

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        self._require_account_id()
        job = (await self._request("GET", self._job_url(batch_id), headers=self._auth_headers())).json()

        status = self._normalize_state(job.get("state", ""))
        progress = job.get("jobProgress") or {}
        result: dict[str, Any] = {
            "batch_id": batch_id,
            "status": status,
            "created_at": job.get("createTime"),
            "request_counts": {
                "total": _to_int(progress.get("totalInputRequests")),
                "completed": _to_int(progress.get("successfullyProcessedRequests")),
                "failed": _to_int(progress.get("failedRequests")),
            },
        }

        output_dataset_id = job.get("outputDatasetId")
        if output_dataset_id:
            result["output_dataset_id"] = output_dataset_id
        # Fireworks reports terminal failure detail in the `status` {code,message}.
        if job.get("status"):
            result["errors"] = job["status"]

        # PENDING-forever guard: the shared retain poll loop has no max-wait, so
        # if a (likely non-batch-eligible) job never reaches a terminal state we
        # surface "expired" once createTime is older than the cap. Derived from
        # the server's createTime so it survives crash-recovery polling resumes.
        if status not in _TERMINAL_STATUSES:
            elapsed = self._elapsed_seconds(job.get("createTime"))
            if elapsed is not None and elapsed > self._max_wait_seconds:
                result["status"] = "expired"
                result["errors"] = (
                    f"Fireworks batch {batch_id} exceeded max wait of {self._max_wait_seconds}s "
                    f"in state {job.get('state')!r}. The model may not be batch-eligible "
                    f"(such jobs stay PENDING indefinitely)."
                )
                logger.error(result["errors"])

        return result

    async def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        self._require_account_id()
        job = (await self._request("GET", self._job_url(batch_id), headers=self._auth_headers())).json()

        status = self._normalize_state(job.get("state", ""))
        if status != "completed":
            raise ValueError(f"Fireworks batch {batch_id} is not completed yet (state: {job.get('state')!r})")

        output_dataset_id = job.get("outputDatasetId")
        if not output_dataset_id:
            raise ValueError(f"Fireworks batch {batch_id} completed but reported no output dataset")

        output_short_id = self._last_segment(output_dataset_id)
        if not output_short_id:
            raise ValueError(
                f"Fireworks batch {batch_id} reported an unparseable output dataset: {output_dataset_id!r}"
            )
        download = (
            await self._request("GET", self._download_endpoint_url(output_short_id), headers=self._auth_headers())
        ).json()
        signed_urls = (download or {}).get("filenameToSignedUrls") or {}
        if not signed_urls:
            raise ValueError(f"Fireworks batch {batch_id} returned no downloadable output files")

        # The output dataset contains a results file plus a separate error file.
        # Download every file and normalize each line; error-file lines carry an
        # `error` so partial failures surface per custom_id instead of vanishing.
        results: list[dict[str, Any]] = []
        for url in signed_urls.values():
            # Signed URLs are pre-authenticated — do not attach the bearer token.
            file_resp = await self._request("GET", url)
            for line in file_resp.text.strip().split("\n"):
                if line.strip():
                    results.append(self._normalize_output_line(json.loads(line)))

        logger.info(f"Retrieved {len(results)} results for Fireworks batch {batch_id}")
        return results

    async def cleanup(self) -> None:
        await super().cleanup()
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()

    # ----- pure translation/normalization helpers (unit-tested) ----------

    @staticmethod
    def _translate_requests(requests: list[dict[str, Any]]) -> str:
        """OpenAI batch request -> Fireworks input JSONL.

        Fireworks lines are ``{"custom_id", "body"}`` — the OpenAI ``method`` and
        ``url`` keys are dropped; ``body`` is kept verbatim.
        """
        lines = [
            json.dumps({"custom_id": req.get("custom_id"), "body": req.get("body")}, ensure_ascii=False)
            for req in requests
        ]
        return "\n".join(lines)

    @staticmethod
    def _normalize_state(fw_state: str) -> str:
        """Fireworks job state -> the retain driver's expected status strings.

        Handles both the API enum (``JOB_STATE_*``) and the guide's bare names
        (``COMPLETED``/``VALIDATING``/``EXPIRED``). Unknown / in-flight states map
        to ``in_progress`` so the driver keeps polling.
        """
        state = (fw_state or "").upper()
        if state.startswith("JOB_STATE_"):
            state = state[len("JOB_STATE_") :]
        if state == "COMPLETED":
            return "completed"
        if state == "FAILED":
            return "failed"
        if state in ("CANCELLED", "CANCELED"):
            return "cancelled"
        if state == "EXPIRED":
            return "expired"
        return "in_progress"

    @staticmethod
    def _normalize_output_line(line: dict[str, Any]) -> dict[str, Any]:
        """Fireworks output JSONL line -> OpenAI-batch-output shape.

        Target: ``{"custom_id", "response": {"body": <chat-completion>}, "error"}``
        so the consumer's ``result["response"]["body"]["choices"][0]...`` works.
        """
        custom_id = line.get("custom_id")
        error = line.get("error")
        if error:
            return {"custom_id": custom_id, "response": None, "error": error}

        response = line.get("response")
        if response is None:
            response = line.get("body")
        # If Fireworks already nests the completion under `body`, unwrap it;
        # otherwise the `response` object *is* the completion.
        if isinstance(response, dict) and "body" in response:
            body = response["body"]
        else:
            body = response
        return {"custom_id": custom_id, "response": {"body": body}, "error": None}

    # ----- low-level HTTP + URL helpers ----------------------------------

    def _require_account_id(self) -> None:
        if not self._account_id:
            raise ValueError(
                "Fireworks batch inference requires an account id. "
                "Set HINDSIGHT_API_FIREWORKS_ACCOUNT_ID to your Fireworks account id."
            )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _http(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(_HTTP_TIMEOUT_SECONDS))
        return self._http_client

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> httpx.Response:
        resp = await self._http().request(method, url, headers=headers, json=json, files=files)
        if resp.is_error:
            # Surface the API's error body. Fireworks returns JSON describing why a
            # 4xx/5xx happened; raise_for_status() alone discards it, which makes
            # failures (e.g. a malformed dataset/job request) undebuggable.
            raise httpx.HTTPStatusError(
                f"Fireworks API {resp.status_code} for {method} {url}: {resp.text[:2000]}",
                request=resp.request,
                response=resp,
            )
        return resp

    def _accounts_base(self) -> str:
        return f"{self._batch_base_url}/v1/accounts/{self._account_id}"

    def _datasets_url(self) -> str:
        return f"{self._accounts_base()}/datasets"

    def _jobs_url(self) -> str:
        return f"{self._accounts_base()}/batchInferenceJobs"

    def _job_url(self, job_id: str) -> str:
        return f"{self._jobs_url()}/{job_id}"

    def _download_endpoint_url(self, dataset_short_id: str) -> str:
        return f"{self._datasets_url()}/{dataset_short_id}:getDownloadEndpoint"

    def _dataset_resource(self, dataset_id: str) -> str:
        return f"accounts/{self._account_id}/datasets/{dataset_id}"

    @staticmethod
    def _last_segment(resource_name: str | None) -> str | None:
        if not resource_name:
            return None
        return resource_name.rstrip("/").split("/")[-1]

    @staticmethod
    def _elapsed_seconds(create_time: str | None) -> float | None:
        if not create_time:
            return None
        try:
            normalized = create_time.replace("Z", "+00:00")
            created = datetime.fromisoformat(normalized)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - created).total_seconds()
        except (ValueError, TypeError):
            return None


def _to_int(value: Any) -> int:
    """Coerce Fireworks' string/int counts to int, defaulting to 0."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
