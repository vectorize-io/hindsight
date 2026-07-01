"""Pure normalization tests — no I/O, mock data only."""

from app.connectors.google_drive.normalize import (
    export_mime_for,
    is_supported,
    normalize_file,
    normalize_permission,
)

RAW_GDOC = {
    "id": "f1",
    "name": "Quarterly Plan",
    "mimeType": "application/vnd.google-apps.document",
    "size": "0",
    "createdTime": "2026-01-01T00:00:00.000Z",
    "modifiedTime": "2026-02-02T10:00:00.000Z",
    "owners": [{"emailAddress": "owner@x.com"}],
    "webViewLink": "https://drive/f1",
    "parents": ["fld"],
    "trashed": False,
    "md5Checksum": "abc123",
}


def test_normalize_file_maps_fields_and_export():
    out = normalize_file(RAW_GDOC)
    assert out["external_id"] == "f1"
    assert out["name"] == "Quarterly Plan"
    assert out["mime_type"] == "application/vnd.google-apps.document"
    assert out["size"] == 0
    assert out["web_view_link"] == "https://drive/f1"
    assert out["checksum"] == "abc123"
    assert out["trashed"] is False
    assert out["metadata"]["owners"] == ["owner@x.com"]
    assert out["metadata"]["parents"] == ["fld"]
    assert out["metadata"]["supported"] is True
    assert out["metadata"]["export_mime"] == "text/plain"


def test_normalize_file_handles_missing_size_and_owners():
    out = normalize_file({"id": "x", "mimeType": "application/pdf"})
    assert out["size"] is None
    assert out["metadata"]["owners"] == []
    assert out["metadata"]["export_mime"] is None  # PDFs ingest as-is


def test_is_supported_matrix():
    assert is_supported("application/pdf")
    assert is_supported("application/vnd.google-apps.spreadsheet")
    assert is_supported("text/markdown")
    assert not is_supported("application/octet-stream")
    assert not is_supported(None)


def test_export_mime_for():
    assert export_mime_for("application/vnd.google-apps.spreadsheet") == "text/csv"
    assert export_mime_for("application/vnd.google-apps.presentation") == "text/plain"
    assert export_mime_for("application/pdf") is None


def test_normalize_permission():
    out = normalize_permission({
        "id": "p1", "type": "user", "role": "reader", "emailAddress": "a@x.com",
        "allowFileDiscovery": True, "expirationTime": "2030-01-01T00:00:00Z",
    })
    assert out["external_permission_id"] == "p1"
    assert out["ptype"] == "user"
    assert out["role"] == "reader"
    assert out["email_address"] == "a@x.com"
    assert out["allow_file_discovery"] is True
    assert out["expiration_time"] is not None
