#!/usr/bin/env python3
"""Extract the webhook schemas used by the TypeScript runtime validator."""

import json
import sys
from pathlib import Path


def referenced_schemas(schema, names):
    pending = list(reversed(names))
    selected = {}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected[name] = schema[name]
        prefix = "#/components/schemas/"
        references = []
        def collect(value):
            if isinstance(value, dict):
                reference = value.get("$ref")
                if reference and reference.startswith(prefix):
                    references.append(reference[len(prefix) :])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(schema[name])
        pending.extend(reversed(references))
    return selected


def main(source_name, destination_name):
    source = Path(source_name)
    destination = Path(destination_name)
    document = json.loads(source.read_text())
    schemas = document.get("components", {}).get("schemas", {})
    required = (
        "WebhookEventEnvelope",
        "WebhookEvent",
        "ConsolidationCompletedWebhookEvent",
        "MemoryDefenseTriggeredWebhookEvent",
        "RetainCompletedWebhookEvent",
    )
    missing = set(required) - schemas.keys()
    assert not missing, f"missing webhook schemas: {sorted(missing)}"
    webhook_schemas = referenced_schemas(schemas, required)
    output = {
        "openapi": document.get("openapi"),
        "components": {"schemas": webhook_schemas},
    }
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: generate-webhook-schema.py SOURCE DESTINATION")
    main(sys.argv[1], sys.argv[2])
