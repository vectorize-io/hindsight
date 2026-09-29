"""Binary download endpoints must publish only their binary media type (#4292).

A route that returns a raw ``Response`` without ``response_class`` gets FastAPI's
default ``application/json`` media type added next to the declared one. The
generated SDKs pick JSON first and decode the body as text, which corrupts the
bytes (a PNG attachment fails with ``UnicodeDecodeError`` on its 0x89 magic byte).
"""

from types import SimpleNamespace

from hindsight_api.api import create_app

_TEXT_MEDIA_PREFIXES = ("application/json", "text/")


def test_no_success_response_mixes_json_with_a_binary_media_type():
    spec = create_app(SimpleNamespace(audit_logger=None), initialize_memory=False).openapi()

    offenders = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            content = operation.get("responses", {}).get("200", {}).get("content", {})
            binary = [media for media in content if not media.startswith(_TEXT_MEDIA_PREFIXES)]
            if binary and "application/json" in content:
                offenders.append(f"{method.upper()} {path}: {sorted(content)}")

    assert not offenders, "binary endpoints also declare application/json:\n" + "\n".join(offenders)


def test_download_endpoints_declare_a_binary_schema():
    """``format: binary`` is what makes the generated clients return bytes, not ``object``."""
    spec = create_app(SimpleNamespace(audit_logger=None), initialize_memory=False).openapi()
    expected = {
        "/v1/default/banks/{bank_id}/attachments/{attachment_id}": "application/octet-stream",
        "/v1/default/files/download/{key}": "application/zip",
    }

    for path, media_type in expected.items():
        content = spec["paths"][path]["get"]["responses"]["200"]["content"]
        assert list(content) == [media_type]
        assert content[media_type]["schema"] == {"type": "string", "format": "binary"}
