"""HTTP client that enumerates observations from a Hindsight API."""

from collections.abc import Callable

import httpx

from .models import Observation

ProgressCallback = Callable[[int, int], None]


class ObservationClient:
    """Reads observation memory units from a running Hindsight API.

    The API has no bulk export, so observations are pulled page-by-page from
    the ``/memories/list`` endpoint filtered to ``type=observation``.
    """

    def __init__(
        self,
        api_url: str,
        *,
        api_key: str | None = None,
        tenant: str = "default",
        timeout: float = 60.0,
    ) -> None:
        self._base = api_url.rstrip("/")
        self._tenant = tenant
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def __enter__(self) -> "ObservationClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def check_health(self) -> None:
        """Raise if the API is unreachable or unhealthy."""
        resp = self._client.get(f"{self._base}/health", timeout=10.0)
        resp.raise_for_status()

    def fetch_observations(
        self,
        bank_id: str,
        *,
        page_size: int = 200,
        progress: ProgressCallback | None = None,
    ) -> list[Observation]:
        """Fetch every observation for ``bank_id``, following pagination."""
        url = f"{self._base}/v1/{self._tenant}/banks/{bank_id}/memories/list"
        observations: list[Observation] = []
        offset = 0
        while True:
            resp = self._client.get(
                url,
                params={"type": "observation", "limit": page_size, "offset": offset},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            for item in items:
                observations.append(
                    Observation(
                        id=item["id"],
                        text=item.get("text") or "",
                        entities=item.get("entities") or "",
                        tags=tuple(item.get("tags") or []),
                        mentioned_at=item.get("mentioned_at"),
                    )
                )
            total = data.get("total", len(observations))
            offset += len(items)
            if progress is not None:
                progress(len(observations), total)
            # Stop on an empty page or once we've collected the reported total.
            if not items or offset >= total:
                break
        return observations
