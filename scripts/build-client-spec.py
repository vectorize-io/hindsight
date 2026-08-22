#!/usr/bin/env python3
"""Build the OpenAPI projection consumed by ordinary client generators."""

import copy
import json
import sys
from pathlib import Path


def iter_refs(value):
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference:
            yield reference
        for child in value.values():
            yield from iter_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_refs(child)


def main(source_name, destination_name):
    source = Path(source_name)
    destination = Path(destination_name)
    original = json.loads(source.read_text())
    projected = copy.deepcopy(original)

    assert "webhooks" in projected, "the source OpenAPI document has no top-level webhooks"
    projected.pop("webhooks")
    assert "webhooks" not in projected
    schemas = projected.get("components", {}).get("schemas", {})
    assert "WebhookEvent" in schemas, "WebhookEvent schema is missing from components"

    for reference in iter_refs(projected):
        prefix = "#/components/schemas/"
        if reference.startswith(prefix):
            assert reference[len(prefix) :] in schemas, f"unresolved schema reference: {reference}"

    unchanged = copy.deepcopy(original)
    unchanged.pop("webhooks", None)
    # Keep this guard when adding projection transforms: only webhooks may be removed.
    assert projected == unchanged, "the client projection changed content besides webhooks"
    destination.write_text(json.dumps(projected, separators=(",", ":"), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: build-client-spec.py SOURCE DESTINATION")
    main(sys.argv[1], sys.argv[2])
