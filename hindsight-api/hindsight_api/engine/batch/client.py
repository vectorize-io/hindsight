"""
Groq Batch API client.

Handles file upload, batch creation, status polling, and result retrieval
via the Groq Batch API (OpenAI-compatible).

API reference: https://console.groq.com/docs/batch
"""

import json
import logging
from io import BytesIO
from typing import Any

import httpx

from .models import BatchJob, BatchRequest, BatchResponse, BatchStatus, FileUploadResponse

logger = logging.getLogger(__name__)

GROQ_API_BASE = "https://api.groq.com/openai/v1"


class GroqBatchClient:
    """Client for the Groq Batch API.

    Provides methods to:
    1. Build JSONL from a list of BatchRequest objects
    2. Upload JSONL files to the Groq Files API
    3. Create batch jobs from uploaded files
    4. Poll batch job status
    5. Download and parse batch results
    """

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 120.0):
        """Initialize the Groq Batch API client.

        Args:
            api_key: Groq API key
            base_url: Override for the Groq API base URL (for testing)
            timeout: HTTP request timeout in seconds
        """
        self._api_key = api_key
        self._base_url = (base_url or GROQ_API_BASE).rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        """Build authorization headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
        }

    def build_jsonl(self, requests: list[BatchRequest]) -> bytes:
        """Build a JSONL byte string from a list of batch requests.

        Args:
            requests: List of BatchRequest objects

        Returns:
            JSONL content as bytes, one JSON object per line
        """
        lines = []
        for req in requests:
            line = req.model_dump_json()
            lines.append(line)
        return b"\n".join(line.encode() for line in lines)

    async def upload_file(self, jsonl_content: bytes, filename: str = "batch_input.jsonl") -> FileUploadResponse:
        """Upload a JSONL file to the Groq Files API.

        Args:
            jsonl_content: JSONL content as bytes
            filename: Name for the uploaded file

        Returns:
            FileUploadResponse with the file ID

        Raises:
            httpx.HTTPStatusError: If the upload fails
        """
        url = f"{self._base_url}/files"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                headers=self._headers(),
                files={"file": (filename, BytesIO(jsonl_content), "application/jsonl")},
                data={"purpose": "batch"},
            )
            response.raise_for_status()

            data = response.json()
            result = FileUploadResponse.model_validate(data)
            logger.info(f"Uploaded batch file: id={result.id}, bytes={result.bytes}, filename={filename}")
            return result

    async def create_batch(
        self,
        input_file_id: str,
        completion_window: str = "24h",
        endpoint: str = "/v1/chat/completions",
    ) -> BatchJob:
        """Create a new batch job from an uploaded file.

        Args:
            input_file_id: File ID from upload_file()
            completion_window: Processing window ("24h" to "7d")
            endpoint: API endpoint path

        Returns:
            BatchJob with the batch ID and initial status

        Raises:
            httpx.HTTPStatusError: If batch creation fails
        """
        url = f"{self._base_url}/batches"

        payload = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                headers={**self._headers(), "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            result = BatchJob.model_validate(data)
            logger.info(
                f"Created batch job: id={result.id}, status={result.status}, "
                f"input_file={input_file_id}, window={completion_window}"
            )
            return result

    async def get_batch_status(self, batch_id: str) -> BatchJob:
        """Check the status of a batch job.

        Args:
            batch_id: Batch job ID from create_batch()

        Returns:
            BatchJob with current status and metadata

        Raises:
            httpx.HTTPStatusError: If the status check fails
        """
        url = f"{self._base_url}/batches/{batch_id}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()

            data = response.json()
            return BatchJob.model_validate(data)

    async def download_results(self, output_file_id: str) -> list[BatchResponse]:
        """Download and parse batch results from the output file.

        Args:
            output_file_id: Output file ID from the completed batch job

        Returns:
            List of BatchResponse objects, one per request in the batch

        Raises:
            httpx.HTTPStatusError: If the download fails
        """
        url = f"{self._base_url}/files/{output_file_id}/content"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()

            content = response.text
            results = []
            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                results.append(BatchResponse.model_validate(data))

            logger.info(f"Downloaded {len(results)} batch results from file {output_file_id}")
            return results

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        """Cancel a batch job.

        Args:
            batch_id: Batch job ID to cancel

        Returns:
            BatchJob with updated status

        Raises:
            httpx.HTTPStatusError: If cancellation fails
        """
        url = f"{self._base_url}/batches/{batch_id}/cancel"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=self._headers())
            response.raise_for_status()

            data = response.json()
            result = BatchJob.model_validate(data)
            logger.info(f"Cancelled batch job: id={result.id}, status={result.status}")
            return result

    async def submit_requests(
        self,
        requests: list[BatchRequest],
        completion_window: str = "24h",
    ) -> BatchJob:
        """Convenience method: build JSONL, upload, and create batch in one call.

        Args:
            requests: List of batch requests to submit
            completion_window: Processing window

        Returns:
            BatchJob tracking the submitted batch

        Raises:
            httpx.HTTPStatusError: If any step fails
        """
        if not requests:
            raise ValueError("Cannot submit empty batch")

        # Build JSONL
        jsonl = self.build_jsonl(requests)
        logger.info(f"Built batch JSONL: {len(requests)} requests, {len(jsonl)} bytes")

        # Upload file
        file_response = await self.upload_file(jsonl)

        # Create batch
        batch = await self.create_batch(
            input_file_id=file_response.id,
            completion_window=completion_window,
        )

        return batch
