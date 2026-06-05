"""Regression tests for OpenAPI spec post-processing in generate_openapi."""

from hindsight_dev.generate_openapi import _restore_binary_format


def test_restores_format_binary_for_octet_stream_string():
    """contentMediaType binary uploads are rewritten to the format:binary form.

    Guards the Files / document-transfer file-upload regression: FastAPI 0.136 /
    Pydantic 2.12 emit OpenAPI-3.1 contentMediaType, which openapi-generator
    v7.10.0 generates as a plain string instead of a multipart file upload.
    """
    schema = {
        "type": "string",
        "title": "File",
        "contentMediaType": "application/octet-stream",
    }

    _restore_binary_format(schema)

    assert schema == {"type": "string", "title": "File", "format": "binary"}
    assert "contentMediaType" not in schema


def test_rewrites_nested_and_array_item_schemas():
    """The walk reaches binary fields nested in properties and array items."""
    schema = {
        "components": {
            "schemas": {
                "Upload": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "contentMediaType": "application/octet-stream"},
                        }
                    },
                }
            }
        }
    }

    _restore_binary_format(schema)

    item = schema["components"]["schemas"]["Upload"]["properties"]["files"]["items"]
    assert item == {"type": "string", "format": "binary"}


def test_leaves_other_content_media_types_untouched():
    """Only application/octet-stream is rewritten; other media types are preserved."""
    schema = {"type": "string", "contentMediaType": "application/json"}

    _restore_binary_format(schema)

    assert schema == {"type": "string", "contentMediaType": "application/json"}
    assert "format" not in schema
