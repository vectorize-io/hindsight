#!/usr/bin/env python3
"""
Generate OpenAPI specification from FastAPI app.

This script imports the FastAPI app and exports its OpenAPI schema to a JSON file.
"""

import json
import sys
from pathlib import Path
from typing import Any

from hindsight_api import MemoryEngine
from hindsight_api.api import create_app
from hindsight_api.webhooks.models import KnownWebhookEvent, WebhookEventEnvelope
from pydantic import TypeAdapter


def _restore_binary_format(node: object) -> None:
    """Rewrite OpenAPI-3.1 binary string fields back to the 3.0 ``format: binary`` form.

    FastAPI/Pydantic (>=0.136 / >=2.12) serialize binary upload fields as
    ``{"type": "string", "contentMediaType": "application/octet-stream"}`` — valid
    OpenAPI 3.1, but openapi-generator v7.10.0 (used by generate-clients.sh) does
    NOT recognize ``contentMediaType`` as a file upload. It then generates the
    ``files``/``file`` params as plain strings instead of binary multipart uploads,
    silently breaking the Go/Python/TypeScript clients of the Files and
    document-transfer endpoints. Earlier FastAPI emitted ``format: binary`` (still
    under ``openapi: 3.1.0``), which the generator handles correctly, so we restore
    that exact representation in-place. Scoped to ``application/octet-stream`` so it
    only touches binary uploads, not arbitrary content-typed strings.
    """
    if isinstance(node, dict):
        if node.get("contentMediaType") == "application/octet-stream":
            node.pop("contentMediaType", None)
            node.pop("contentEncoding", None)
            node["format"] = "binary"
        for value in node.values():
            _restore_binary_format(value)
    elif isinstance(node, list):
        for item in node:
            _restore_binary_format(item)


def _normalize_webhook_schema(node: object) -> None:
    """Keep standalone webhook components compatible with SDK generators.

    OpenAPI Generator 7.10 incorrectly promotes the ``null`` branch of an
    ``anyOf`` in components that are only referenced by a top-level webhook to
    a standalone ``none_type`` model. It also ignores JSON Schema ``const`` in
    generated Python models. The equivalent OpenAPI 3.0 spellings avoid both
    issues and are already understood by every generator used in this repo.
    """
    if isinstance(node, dict):
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            non_null = [item for item in any_of if not (isinstance(item, dict) and item.get("type") == "null")]
            if len(non_null) == 1:
                node.pop("anyOf")
                node.update(non_null[0])
                node["nullable"] = True

        if "const" in node:
            node["enum"] = [node.pop("const")]

        for value in node.values():
            _normalize_webhook_schema(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_webhook_schema(item)


def _publish_webhook_contract(openapi_schema: dict[str, Any]) -> None:
    """Publish outbound webhook models and delivery semantics in the main spec."""
    event_schema = TypeAdapter(KnownWebhookEvent).json_schema(
        ref_template="#/components/schemas/{model}",
    )
    event_definitions = event_schema.pop("$defs")
    event_schema["title"] = "WebhookEvent"

    envelope_schema = WebhookEventEnvelope.model_json_schema(
        ref_template="#/components/schemas/{model}",
    )
    envelope_definitions = envelope_schema.pop("$defs", {})

    for schema in [*event_definitions.values(), *envelope_definitions.values(), event_schema, envelope_schema]:
        _normalize_webhook_schema(schema)

    # Typify names inline enums from their title. Pydantic calls every
    # discriminator field simply "Event", which makes Rust reuse the first
    # enum for all three event models unless each title is made unique.
    for model_name in (
        "ConsolidationCompletedWebhookEvent",
        "RetainCompletedWebhookEvent",
        "MemoryDefenseTriggeredWebhookEvent",
    ):
        event_definitions[model_name]["properties"]["event"]["title"] = f"{model_name}Type"

    components = openapi_schema.setdefault("components", {}).setdefault("schemas", {})
    components.update(event_definitions)
    components.update(envelope_definitions)
    components["WebhookEventEnvelope"] = envelope_schema
    components["WebhookEvent"] = event_schema

    # OpenAPI 3.1 models outbound webhooks directly. Components remain the source
    # consumed by generators that do not implement the top-level webhooks keyword.
    openapi_schema["webhooks"] = {
        "hindsightEvent": {
            "post": {
                "summary": "Receive a Hindsight webhook event",
                "description": (
                    "Verify X-Hindsight-Signature against the exact raw request body before parsing the event."
                ),
                "parameters": [
                    {
                        "name": "X-Hindsight-Event",
                        "in": "header",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Event name. Dispatch using the signed body rather than this unsigned header.",
                    },
                    {
                        "name": "X-Hindsight-Signature",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string", "pattern": "^sha256=[0-9a-f]{64}$"},
                        "description": "HMAC-SHA256 of the exact raw body. Present when the webhook has a secret.",
                    },
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/WebhookEvent"},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Return any 2xx response after the event is durably accepted.",
                    }
                },
            }
        }
    }


def generate_openapi_spec(output_path: str = None):
    """Generate OpenAPI spec and save to file."""
    # Default to hindsight-docs/static/openapi.json (single source of truth)
    if output_path is None:
        # Get the root of the project (3 levels up from this file)
        root_dir = Path(__file__).parent.parent.parent
        output_path = str(root_dir / "hindsight-docs" / "static" / "openapi.json")

    # Create a temporary memory instance for OpenAPI generation
    _memory = MemoryEngine(
        db_url="mock",
        memory_llm_provider="ollama",
        memory_llm_api_key="mock",
        memory_llm_model="mock",
    )
    app = create_app(_memory)

    # Get the OpenAPI schema from the app
    openapi_schema = app.openapi()

    # Outbound events are not FastAPI routes, so publish their contract explicitly.
    _publish_webhook_contract(openapi_schema)

    # Keep binary upload fields generator-compatible (see helper docstring).
    _restore_binary_format(openapi_schema)

    # Write to file
    output_file = Path(output_path)
    with open(output_file, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✓ OpenAPI specification generated: {output_file.absolute()}")
    print(f"  - Title: {openapi_schema['info']['title']}")
    print(f"  - Version: {openapi_schema['info']['version']}")
    print(f"  - Endpoints: {len(openapi_schema['paths'])}")

    # List endpoints
    print("\n  Endpoints:")
    for path, methods in openapi_schema["paths"].items():
        for method in methods.keys():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                endpoint_info = methods[method]
                summary = endpoint_info.get("summary", "No summary")
                tags = ", ".join(endpoint_info.get("tags", ["untagged"]))
                print(f"    {method.upper():6} {path:30} [{tags}] - {summary}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    generate_openapi_spec(output)
