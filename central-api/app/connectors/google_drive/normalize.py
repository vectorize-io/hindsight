"""Pure normalization: raw Drive API shapes → control-plane row dicts.

No I/O, no secrets — just shape mapping, so these are cheap to unit-test.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Google-native MIME types → the export format we request before ingestion.
GOOGLE_EXPORT_MAP: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# Binary/text types ingested as-is (no export step).
SUPPORTED_DIRECT_TYPES: frozenset[str] = frozenset({
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "text/plain",
    "text/markdown",
    "text/csv",
})


def is_supported(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    return mime_type in GOOGLE_EXPORT_MAP or mime_type in SUPPORTED_DIRECT_TYPES


def export_mime_for(mime_type: str | None) -> str | None:
    """Export format for a Google-native file, or None if no export needed."""
    if not mime_type:
        return None
    return GOOGLE_EXPORT_MAP.get(mime_type)


def _parse_time(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_file(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a Drive ``files.get/list`` resource to source_documents fields."""
    mime = raw.get("mimeType")
    return {
        "external_id": raw.get("id"),
        "name": raw.get("name"),
        "mime_type": mime,
        "size": _to_int(raw.get("size")),
        "web_view_link": raw.get("webViewLink"),
        "checksum": raw.get("md5Checksum") or raw.get("version"),
        "trashed": bool(raw.get("trashed", False)),
        "metadata": {
            "owners": [o.get("emailAddress") for o in raw.get("owners", []) if isinstance(o, dict)],
            "parents": raw.get("parents", []),
            "created_time": raw.get("createdTime"),
            "modified_time": raw.get("modifiedTime"),
            "supported": is_supported(mime),
            "export_mime": export_mime_for(mime),
        },
    }


def normalize_permission(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a Drive permission resource to source_document_permissions fields."""
    return {
        "external_permission_id": raw.get("id"),
        "ptype": raw.get("type"),  # user|group|domain|anyone
        "role": raw.get("role"),  # owner|organizer|writer|commenter|reader
        "email_address": raw.get("emailAddress"),
        "domain": raw.get("domain"),
        "allow_file_discovery": raw.get("allowFileDiscovery"),
        "expiration_time": _parse_time(raw.get("expirationTime")),
    }
