"""Drive client abstraction.

A ``DriveClient`` protocol with two implementations:
- ``FakeDriveClient`` — in-memory, used by dev and all tests (no network, no creds).
- ``GoogleDriveClient`` — real read-only client; google libs are imported lazily
  so the package is optional and tests never need it.

Only read methods exist. There is intentionally no write/delete/export-write
surface — the connector is read-only by policy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DriveClient(Protocol):
    async def list_files(self, folder_id: str) -> list[dict[str, Any]]: ...
    async def get_file(self, file_id: str) -> dict[str, Any]: ...
    async def list_permissions(self, file_id: str) -> list[dict[str, Any]]: ...


class FakeDriveClient:
    """Deterministic in-memory Drive for dev/tests. Seed with raw-shaped dicts."""

    def __init__(self, files: dict[str, list[dict]] | None = None,
                 permissions: dict[str, list[dict]] | None = None) -> None:
        # files: {folder_id: [file_resource, ...]}
        self._files = files or {}
        self._permissions = permissions or {}

    async def list_files(self, folder_id: str) -> list[dict[str, Any]]:
        return list(self._files.get(folder_id, []))

    async def get_file(self, file_id: str) -> dict[str, Any]:
        for files in self._files.values():
            for f in files:
                if f.get("id") == file_id:
                    return dict(f)
        raise KeyError(file_id)

    async def list_permissions(self, file_id: str) -> list[dict[str, Any]]:
        return list(self._permissions.get(file_id, []))


# Fields requested from Drive — metadata only, no content.
_FILE_FIELDS = (
    "files(id,name,mimeType,size,createdTime,modifiedTime,owners(emailAddress),"
    "webViewLink,parents,trashed,md5Checksum,version)"
)
_PERM_FIELDS = (
    "permissions(id,type,role,emailAddress,domain,allowFileDiscovery,expirationTime)"
)


class GoogleDriveClient:
    """Real read-only Drive client. Requires the optional ``gdrive`` extra.

    Constructed from an authorized credentials object; never holds raw secrets
    in attributes that get logged or serialized.
    """

    def __init__(self, credentials: Any) -> None:
        try:
            from googleapiclient.discovery import build  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised only when extra missing
            raise RuntimeError(
                "Google Drive client requires the 'gdrive' extra: pip install '.[gdrive]'"
            ) from exc
        # cache_discovery=False avoids file-cache warnings in server contexts.
        self._svc = build("drive", "v3", credentials=credentials, cache_discovery=False)

    async def list_files(self, folder_id: str) -> list[dict[str, Any]]:  # pragma: no cover
        q = f"'{folder_id}' in parents and trashed = false"
        out: list[dict] = []
        page_token = None
        while True:
            resp = self._svc.files().list(
                q=q, fields=f"nextPageToken,{_FILE_FIELDS}", pageSize=100,
                pageToken=page_token, supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            out.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                return out

    async def get_file(self, file_id: str) -> dict[str, Any]:  # pragma: no cover
        return self._svc.files().get(
            fileId=file_id, fields=_FILE_FIELDS.replace("files(", "").rstrip(")"),
            supportsAllDrives=True,
        ).execute()

    async def list_permissions(self, file_id: str) -> list[dict[str, Any]]:  # pragma: no cover
        resp = self._svc.permissions().list(
            fileId=file_id, fields=_PERM_FIELDS, supportsAllDrives=True,
        ).execute()
        return resp.get("permissions", [])
