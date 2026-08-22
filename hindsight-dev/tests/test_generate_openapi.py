"""Regression tests for OpenAPI spec post-processing in generate_openapi."""

from hindsight_dev.generate_openapi import _publish_webhook_contract, _restore_binary_format


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


def test_publishes_webhook_components_and_discriminator():
    schema = {"components": {"schemas": {}}}

    _publish_webhook_contract(schema)

    components = schema["components"]["schemas"]
    assert {
        "WebhookEvent",
        "WebhookEventEnvelope",
        "ConsolidationCompletedWebhookEvent",
        "RetainCompletedWebhookEvent",
        "MemoryDefenseTriggeredWebhookEvent",
    } <= components.keys()
    event_schema = components["WebhookEvent"]
    assert event_schema["discriminator"]["propertyName"] == "event"
    assert len(event_schema["oneOf"]) == 3

    observations_created = components["ConsolidationEventData"]["properties"]["observations_created"]
    assert observations_created["type"] == "integer"
    assert observations_created["nullable"] is True
    assert "anyOf" not in observations_created

    event_name = components["RetainCompletedWebhookEvent"]["properties"]["event"]
    assert event_name["enum"] == ["retain.completed"]
    assert event_name["title"] == "RetainCompletedWebhookEventType"
    assert "const" not in event_name


def test_publishes_post_webhook_operation_referencing_event_component():
    schema = {"components": {"schemas": {}}}

    _publish_webhook_contract(schema)

    operation = schema["webhooks"]["hindsightEvent"]["post"]
    body_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert body_schema == {"$ref": "#/components/schemas/WebhookEvent"}
    assert {parameter["name"] for parameter in operation["parameters"]} == {
        "X-Hindsight-Event",
        "X-Hindsight-Signature",
    }
