#!/usr/bin/env python3
"""
Generate OpenAPI specification from FastAPI app.

This script imports the FastAPI app and exports its OpenAPI schema to a JSON file.
"""

import json
import sys
from pathlib import Path

from hindsight_api import MemoryEngine
from hindsight_api.api import create_app


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
