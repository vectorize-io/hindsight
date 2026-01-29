"""Vertex AI token refresher with background refresh and caching."""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class VertexAITokenRefresher:
    """
    Background token refresher for Vertex AI.

    Refreshes Google Cloud access tokens every 50 minutes to ensure they don't expire (60-min default).
    Thread-safe token caching for concurrent access from multiple async tasks.
    """

    def __init__(self, credentials: Any, project_id: str, region: str):
        """
        Initialize the token refresher.

        Args:
            credentials: Google Cloud credentials object (from google.auth.default or service_account)
            project_id: GCP project ID
            region: GCP region (e.g., "us-central1")
        """
        self._credentials = credentials
        self._project_id = project_id
        self._region = region

        # Thread-safe token cache
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        self._lock = threading.Lock()

        # Background refresh task
        self._refresh_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # Initial token fetch (synchronous, must complete before returning)
        self._refresh_token_sync()

    def _refresh_token_sync(self) -> None:
        """Synchronously refresh the token (thread-safe)."""
        try:
            import google.auth.transport.requests

            request = google.auth.transport.requests.Request()
            self._credentials.refresh(request)

            with self._lock:
                self._token = self._credentials.token
                self._token_expiry = self._credentials.expiry

            logger.debug(f"Vertex AI token refreshed, expires at {self._token_expiry}")
        except Exception as e:
            logger.error(f"Failed to refresh Vertex AI token: {e}")
            raise

    async def _refresh_loop(self) -> None:
        """Background refresh loop (runs every 50 minutes)."""
        while not self._stop_event.is_set():
            try:
                # Wait 50 minutes or until stop event
                await asyncio.wait_for(self._stop_event.wait(), timeout=50 * 60)
                # If we get here, stop was signaled
                break
            except asyncio.TimeoutError:
                # 50 minutes passed, refresh token
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._refresh_token_sync)
                except Exception as e:
                    logger.error(f"Background token refresh failed: {e}")
                    # Continue loop - next API call will fail with auth error

    def start_refresh_task(self) -> None:
        """Start the background refresh task."""
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_loop())
            logger.info("Vertex AI token refresh task started (refreshes every 50 minutes)")

    async def stop(self) -> None:
        """Stop the background refresh task."""
        if self._refresh_task is not None and not self._refresh_task.done():
            self._stop_event.set()
            try:
                await asyncio.wait_for(self._refresh_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Vertex AI token refresh task did not stop within 5 seconds")
            logger.info("Vertex AI token refresh task stopped")

    def get_token(self) -> str:
        """
        Get current access token (thread-safe).

        Returns:
            Current Google Cloud access token

        Raises:
            RuntimeError: If token is not available
        """
        with self._lock:
            if self._token is None:
                raise RuntimeError("Vertex AI token not available")
            return self._token

    def get_base_url(self) -> str:
        """
        Get the Vertex AI OpenAI-compatible endpoint URL.

        Returns:
            Base URL for Vertex AI OpenAI API
        """
        return (
            f"https://{self._region}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{self._project_id}/locations/{self._region}/endpoints/openapi"
        )
