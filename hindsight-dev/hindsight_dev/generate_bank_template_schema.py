#!/usr/bin/env python3
"""
Generate bank template JSON Schema from the Pydantic model.

This script imports BankTemplateManifest and exports its JSON Schema to
hindsight-docs/static/bank-template-schema.json.
"""

import json
import sys
from pathlib import Path

from hindsight_api.api.http import BankTemplateManifest


def generate_bank_template_schema(output_path: str | None = None) -> None:
    """Generate bank template JSON Schema and save to file."""
    if output_path is None:
        # Default to hindsight-docs/static/bank-template-schema.json
        root_dir = Path(__file__).parent.parent.parent
        output_path = str(root_dir / "hindsight-docs" / "static" / "bank-template-schema.json")

    schema = BankTemplateManifest.model_json_schema()

    output_file = Path(output_path)
    with open(output_file, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    print(f"✓ Bank template schema generated: {output_file.absolute()}")
    print(f"  - Title: {schema['title']}")
    print(f"  - Definitions: {len(schema['$defs'])}")
    print(f"  - BankTemplateConfig fields: {len(schema['$defs']['BankTemplateConfig']['properties'])}")


def main() -> None:
    output = sys.argv[1] if len(sys.argv) > 1 else None
    generate_bank_template_schema(output)


if __name__ == "__main__":
    main()
