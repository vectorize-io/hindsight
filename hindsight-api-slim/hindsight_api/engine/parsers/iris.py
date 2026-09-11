"""Iris parser implementation using the Vectorize Iris HTTP API."""

import asyncio
import json
import logging
import mimetypes
import time
from typing import Any

import aiohttp
from yarl import URL

from ..aiohttp_session import per_phase_timeout
from .base import FileParser, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

_IRIS_BASE_URL = "https://api.vectorize.io/v1"
_DEFAULT_POLL_INTERVAL = 2.0  # seconds
_DEFAULT_TIMEOUT = 300.0  # seconds


class IrisParser(FileParser):
    """
    Iris file parser using the Vectorize Iris cloud extraction service.

    Uploads files to the Vectorize Iris API, starts an extraction job,
    and polls until the text is ready. The API determines which file types
    are supported — UnsupportedFileTypeError is raised if the file is rejected.

    Authentication:
        Requires HINDSIGHT_API_FILE_PARSER_IRIS_TOKEN and
        HINDSIGHT_API_FILE_PARSER_IRIS_ORG_ID environment variables,
        or pass them explicitly via the constructor.
    """

    def __init__(
        self,
        token: str,
        org_id: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        """
        Initialize iris parser.

        Args:
            token: Vectorize API token
            org_id: Vectorize organization ID
            poll_interval: Seconds between status poll requests (default: 2)
            timeout: Maximum seconds to wait for extraction (default: 300)
        """
        self._token = token
        self._org_id = org_id
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._auth_headers = {"Authorization": f"Bearer {token}"}

    async def convert(self, file_data: bytes, filename: str) -> str:
        """
        Parse file to text using the Vectorize Iris API.

        Raises:
            UnsupportedFileTypeError: If the Iris API rejects the file type (4xx)
            RuntimeError: If extraction fails for another reason
        """
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        # One session per conversion: a rare call, and the parser is built at startup
        # possibly off the serving loop, so it does not hold a long-lived one.
        async with aiohttp.ClientSession(timeout=per_phase_timeout(120.0, connect=30.0)) as client:
            # Step 1: Request a presigned upload URL
            async with client.post(
                f"{_IRIS_BASE_URL}/org/{self._org_id}/files",
                headers=self._auth_headers,
                json={"name": filename, "contentType": content_type},
            ) as resp:
                init_data = await _json_or_raise(resp, filename, "file upload init")
            file_id: str = init_data["fileId"]
            upload_url: str = init_data["uploadUrl"]

            # Step 2: Upload the file bytes to the presigned URL (no auth header)
            # Ensure file_data is plain bytes (GCS storage may return obstore.Bytes)
            # encoded=True: re-quoting the presigned query string would break its signature.
            async with client.put(
                URL(upload_url, encoded=True),
                data=bytes(file_data),
                headers={"Content-Type": content_type},
            ) as resp:
                await _read_or_raise(resp, filename, "file upload")

            # Step 3: Start extraction
            async with client.post(
                f"{_IRIS_BASE_URL}/org/{self._org_id}/extraction",
                headers=self._auth_headers,
                json={"fileId": file_id},
            ) as resp:
                extraction_id: str = (await _json_or_raise(resp, filename, "start extraction"))["extractionId"]

            # Step 4: Poll until ready or timeout
            deadline = time.monotonic() + self._timeout
            while True:
                async with client.get(
                    f"{_IRIS_BASE_URL}/org/{self._org_id}/extraction/{extraction_id}",
                    headers=self._auth_headers,
                ) as resp:
                    status_data = await _json_or_raise(resp, filename, "poll extraction status")

                if status_data.get("ready"):
                    data = status_data.get("data", {})
                    if not data.get("success"):
                        error = data.get("error", "unknown error")
                        raise RuntimeError(f"Iris extraction failed for '{filename}': {error}")
                    text = data.get("text")
                    if not text:
                        raise RuntimeError(f"No content extracted from '{filename}'")
                    return text

                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Iris extraction timed out after {self._timeout}s for '{filename}'")

                await asyncio.sleep(self._poll_interval)

    def name(self) -> str:
        """Get parser name."""
        return "iris"


async def _json_or_raise(response: aiohttp.ClientResponse, filename: str, step: str) -> Any:
    """Return the decoded JSON body; raise like :func:`_read_or_raise` on HTTP errors."""
    return json.loads(await _read_or_raise(response, filename, step))


async def _read_or_raise(response: aiohttp.ClientResponse, filename: str, step: str) -> str:
    """
    Return the response body, raising an appropriate error including it on HTTP errors.

    Raises UnsupportedFileTypeError for 4xx responses (file rejected by the API),
    RuntimeError for other HTTP errors.
    """
    text = await response.text()
    if response.status < 400:
        return text
    body = text or "<empty>"
    msg = f"Iris API error during {step} for '{filename}': {response.status} {response.reason} — {body}"
    if response.status < 500:
        raise UnsupportedFileTypeError(msg)
    raise RuntimeError(msg)
