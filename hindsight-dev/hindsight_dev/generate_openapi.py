#!/usr/bin/env python3
"""
Generate OpenAPI specification from FastAPI app.

This script imports the FastAPI app and exports its OpenAPI schema to a JSON file.
"""
import json
import sys
import os
from pathlib import Path

from hindsight_api.api import create_app
from hindsight_api import MemoryEngine

def generate_openapi_spec(output_path: str = None):
    """Generate OpenAPI spec and save to file."""
    # Default to root openapi.json if no path specified
    if output_path is None:
        # Get the root of the project (3 levels up from this file)
        root_dir = Path(__file__).parent.parent.parent
        output_path = str(root_dir / "openapi.json")

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

    # Write to file
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✓ OpenAPI specification generated: {output_file.absolute()}")
    print(f"  - Title: {openapi_schema['info']['title']}")
    print(f"  - Version: {openapi_schema['info']['version']}")
    print(f"  - Endpoints: {len(openapi_schema['paths'])}")

    # List endpoints
    print("\n  Endpoints:")
    for path, methods in openapi_schema['paths'].items():
        for method in methods.keys():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                endpoint_info = methods[method]
                summary = endpoint_info.get('summary', 'No summary')
                tags = ', '.join(endpoint_info.get('tags', ['untagged']))
                print(f"    {method.upper():6} {path:30} [{tags}] - {summary}")

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    generate_openapi_spec(output)
