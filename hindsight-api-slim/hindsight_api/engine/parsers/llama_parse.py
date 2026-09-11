"""LlamaParse parser implementation using the LlamaIndex Cloud parsing API."""

import asyncio
import json
import logging
import mimetypes
import time
from typing import Any

import aiohttp

from ..aiohttp_session import per_phase_timeout
from .base import FileParser, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

_LLAMA_PARSE_BASE_URL = "https://api.cloud.llamaindex.ai/api/parsing"
_DEFAULT_POLL_INTERVAL = 2.0  # seconds
_DEFAULT_TIMEOUT = 300.0  # seconds

# HTTP status codes that indicate the file type is not supported.
# Other 4xx codes (401, 403, 429, etc.) are operational errors, not file-type issues.
_UNSUPPORTED_FILE_STATUS_CODES = {400, 415, 422}


class LlamaParseParser(FileParser):
    """
    LlamaParse file parser using LlamaIndex's hosted parsing service.

    Uploads files to the LlamaParse API, polls until the parse job completes,
    and returns the resulting markdown. The API determines which file types
    are supported — UnsupportedFileTypeError is raised if the file is rejected.
    """

    def __init__(
        self,
        api_key: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        """
        Initialize llama_parse parser.

        Args:
            api_key: LlamaCloud API key (typically starts with "llx-")
            poll_interval: Seconds between status poll requests (default: 2)
            timeout: Maximum seconds to wait for parsing (default: 300)
        """
        self._api_key = api_key
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._auth_headers = {"Authorization": f"Bearer {api_key}"}

    async def convert(self, file_data: bytes, filename: str) -> str:
        """
        Parse file to markdown using the LlamaParse API.

        Raises:
            UnsupportedFileTypeError: If the LlamaParse API rejects the file type
            RuntimeError: If parsing fails for another reason
        """
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        # One session per conversion: parsers have no close hook to release a
        # long-lived one, and a conversion is a rare, multi-second polling job.
        async with aiohttp.ClientSession(timeout=per_phase_timeout(120.0, connect=30.0)) as session:
            return await self._convert(session, file_data, filename, content_type)

    async def _convert(self, session: aiohttp.ClientSession, file_data: bytes, filename: str, content_type: str) -> str:
        # Step 1: Upload file and start parse job
        form = aiohttp.FormData()
        # Ensure file_data is plain bytes (storage backends may return obstore.Bytes)
        form.add_field("file", bytes(file_data), filename=filename, content_type=content_type)
        async with session.post(f"{_LLAMA_PARSE_BASE_URL}/upload", headers=self._auth_headers, data=form) as resp:
            upload_data = await _json_or_raise(resp, filename, "upload")
        job_id: str = upload_data["id"]

        # Step 2: Poll job status until SUCCESS or ERROR
        deadline = time.monotonic() + self._timeout
        while True:
            async with session.get(f"{_LLAMA_PARSE_BASE_URL}/job/{job_id}", headers=self._auth_headers) as resp:
                status_data = await _json_or_raise(resp, filename, "poll job status")
            status = status_data.get("status")

            if status == "SUCCESS":
                break
            if status in ("ERROR", "CANCELLED"):
                error = status_data.get("error_code") or status_data.get("error") or "unknown error"
                raise RuntimeError(f"LlamaParse job failed for '{filename}': {error}")

            if time.monotonic() >= deadline:
                raise RuntimeError(f"LlamaParse job timed out after {self._timeout}s for '{filename}'")

            await asyncio.sleep(self._poll_interval)

        # Step 3: Fetch the markdown result
        async with session.get(
            f"{_LLAMA_PARSE_BASE_URL}/job/{job_id}/result/markdown", headers=self._auth_headers
        ) as resp:
            result_data = await _json_or_raise(resp, filename, "fetch markdown result")
        markdown = result_data.get("markdown")
        if not markdown:
            raise RuntimeError(f"No content extracted from '{filename}'")
        return markdown

    def name(self) -> str:
        """Get parser name."""
        return "llama_parse"


async def _json_or_raise(response: aiohttp.ClientResponse, filename: str, step: str) -> Any:
    """
    Return the decoded JSON body, or raise an appropriate error for HTTP errors.

    Raises UnsupportedFileTypeError for 400/415/422 (file rejected by the API).
    Raises RuntimeError for all other errors (auth, rate-limit, server errors).
    """
    text = await response.text()
    if response.status < 400:
        return json.loads(text)
    body = text or "<empty>"
    msg = f"LlamaParse API error during {step} for '{filename}': {response.status} {response.reason} — {body}"
    if response.status in _UNSUPPORTED_FILE_STATUS_CODES:
        raise UnsupportedFileTypeError(msg)
    raise RuntimeError(msg)
