#!/usr/bin/env python3
"""
Check that the Hindsight CLI covers every operation AND every request-body
parameter in the OpenAPI spec.

Endpoint-level check
--------------------
For each ``operationId`` in ``hindsight-docs/static/openapi.json`` we require
that one of the following is true:
  1. A call of the form ``.<operation_id>(`` appears somewhere in
     ``hindsight-cli/src/**/*.rs``. The progenitor-generated Rust client
     methods are named identically to the OpenAPI ``operationId``, so this is
     a strong signal the CLI wires the endpoint.
  2. The ``operationId`` is listed in ``hindsight-cli/.openapi-coverage.toml``
     under ``[skip]`` with a reason.

Parameter-level check
---------------------
For each operation whose request body schema has named properties, we require
that every property is one of:
  1. Present in ``hindsight-cli/src/main.rs`` as a clap command variant field
     (``field_name: <type>``) or as a ``long = "..."`` clap attribute. main.rs
     is where the user-facing CLI args live, so absence there means the user
     has no way to set that field.
  2. Listed in ``.openapi-coverage.toml`` under
     ``[fields.<operation_id>]`` with a reason explaining why it is not
     exposed (for example, flattened into several CLI flags, or a complex
     nested struct).

Stale manifest entries (skips for things that no longer need skipping) also
fail the check.

Usage:
    cd hindsight-dev
    uv run cli-coverage-check
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}

# OpenAPI request body fields sometimes collide with Rust reserved keywords;
# the progenitor client serde-renames them (e.g. `async` → `async_`). When we
# look up these fields in main.rs, we also accept the aliased name.
RUST_KEYWORD_ALIASES: dict[str, set[str]] = {
    "async": {"async_", "async_mode", "r#async"},
    "type": {"type_", "r#type"},
}


def get_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "hindsight-cli").is_dir() and (parent / "hindsight-docs").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repo root from {here}")


def load_operations(spec_path: Path) -> tuple[set[str], dict[str, list[str]]]:
    """Return (operation_ids, op_id -> list of request-body property names).

    Operations without a request body are omitted from the property map.
    """
    spec = json.loads(spec_path.read_text())
    op_ids: set[str] = set()
    op_props: dict[str, list[str]] = {}
    schemas = spec.get("components", {}).get("schemas", {})

    for path_item in spec.get("paths", {}).values():
        for method, op in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
            op_ids.add(op_id)

            body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            if not body:
                continue
            if "$ref" in body:
                schema_name = body["$ref"].split("/")[-1]
                schema = schemas.get(schema_name, {})
            else:
                schema = body
            props = list((schema.get("properties") or {}).keys())
            if props:
                op_props[op_id] = props

    return op_ids, op_props


def load_manifest(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Return (operation-level skips, per-operation field skips)."""
    if not path.exists():
        return {}, {}
    data = tomllib.loads(path.read_text())

    skip_raw = data.get("skip", {}) or {}
    if not isinstance(skip_raw, dict):
        raise ValueError(f"{path}: [skip] must be a table")
    op_skips: dict[str, str] = {}
    for op_id, reason in skip_raw.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: skip.{op_id} must be a non-empty string reason")
        op_skips[op_id] = reason.strip()

    fields_raw = data.get("fields", {}) or {}
    if not isinstance(fields_raw, dict):
        raise ValueError(f"{path}: [fields] must be a table")
    field_skips: dict[str, dict[str, str]] = {}
    for op_id, table in fields_raw.items():
        if not isinstance(table, dict):
            raise ValueError(f"{path}: [fields.{op_id}] must be a table")
        per_op: dict[str, str] = {}
        for field, reason in table.items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{path}: fields.{op_id}.{field} must be a non-empty string reason")
            per_op[field] = reason.strip()
        field_skips[op_id] = per_op

    return op_skips, field_skips


def load_rust_source(cli_src: Path) -> str:
    chunks: list[str] = []
    for rs in cli_src.rglob("*.rs"):
        try:
            chunks.append(rs.read_text())
        except OSError:
            continue
    return "\n".join(chunks)


def find_implemented_ops(all_src: str, op_ids: set[str]) -> set[str]:
    """Subset of op_ids that appear as ``.<op_id>(`` in any Rust file."""
    implemented: set[str] = set()
    for op_id in op_ids:
        if re.search(r"\.\s*" + re.escape(op_id) + r"\s*\(", all_src):
            implemented.add(op_id)
    return implemented


def field_in_main_rs(prop: str, main_src: str) -> bool:
    """Is `prop` exposed as a CLI arg in main.rs?

    We look for either a struct variant field declaration
    (``field_name: <type>``) or an explicit clap ``long = "..."`` attribute
    matching the property name (snake_case or kebab-case).
    """
    kebab = prop.replace("_", "-")
    patterns = [
        # Struct variant field declaration. We require a type after the colon
        # to distinguish from URL path tokens like `/v1/default/banks/...`.
        r"\b" + re.escape(prop) + r"\s*:\s*(?:Option<|Vec<|bool|i\d+|u\d+|f\d+|String|PathBuf|Path)",
        # Explicit clap long attribute (either spelling).
        rf'long\s*=\s*"{re.escape(prop)}"',
        rf'long\s*=\s*"{re.escape(kebab)}"',
    ]
    return any(re.search(p, main_src) for p in patterns)


def field_covered(prop: str, main_src: str) -> bool:
    if field_in_main_rs(prop, main_src):
        return True
    for alias in RUST_KEYWORD_ALIASES.get(prop, set()):
        if field_in_main_rs(alias, main_src):
            return True
    return False


def main() -> None:
    root = get_repo_root()
    spec_path = root / "hindsight-docs" / "static" / "openapi.json"
    manifest_path = root / "hindsight-cli" / ".openapi-coverage.toml"
    cli_src_dir = root / "hindsight-cli" / "src"
    main_rs_path = cli_src_dir / "main.rs"

    if not spec_path.exists():
        print(f"ERROR: OpenAPI spec not found at {spec_path}", file=sys.stderr)
        print("       Run ./scripts/generate-openapi.sh first.", file=sys.stderr)
        sys.exit(1)
    if not cli_src_dir.is_dir():
        print(f"ERROR: CLI source dir not found at {cli_src_dir}", file=sys.stderr)
        sys.exit(1)
    if not main_rs_path.exists():
        print(f"ERROR: main.rs not found at {main_rs_path}", file=sys.stderr)
        sys.exit(1)

    spec_ops, op_props = load_operations(spec_path)
    op_skips, field_skips = load_manifest(manifest_path)
    all_src = load_rust_source(cli_src_dir)
    main_src = main_rs_path.read_text()
    implemented = find_implemented_ops(all_src, spec_ops)

    errors: list[str] = []

    # ----- endpoint-level -----
    unmapped_ops = sorted(spec_ops - implemented - op_skips.keys())
    for op_id in unmapped_ops:
        errors.append(
            f"MISSING OP   {op_id}: not called from hindsight-cli/src/ and not in .openapi-coverage.toml [skip]"
        )

    stale_op_skips = sorted(op_skips.keys() - spec_ops)
    for op_id in stale_op_skips:
        errors.append(
            f"STALE OP     {op_id}: listed in [skip] but not present in "
            f"openapi.json. Remove it from .openapi-coverage.toml."
        )

    redundant_op_skips = sorted(op_skips.keys() & implemented)
    for op_id in redundant_op_skips:
        errors.append(
            f"REDUNDANT OP {op_id}: listed in [skip] but is now called from "
            f"the CLI. Remove it from .openapi-coverage.toml."
        )

    # ----- parameter-level -----
    total_props = 0
    skipped_props = 0
    covered_props = 0
    for op_id, props in op_props.items():
        # If the whole operation is intentionally skipped, its params don't
        # need to be covered either.
        if op_id in op_skips:
            continue
        per_op_skips = field_skips.get(op_id, {})
        for prop in props:
            total_props += 1
            if prop in per_op_skips:
                skipped_props += 1
                continue
            if field_covered(prop, main_src):
                covered_props += 1
                continue
            errors.append(
                f"MISSING PARAM {op_id}.{prop}: request body field not exposed "
                f"as a CLI arg in main.rs and not listed in "
                f"[fields.{op_id}]"
            )

    # Stale field skips (unknown operations or unknown fields)
    for op_id, per_op in field_skips.items():
        if op_id not in op_props:
            for field in per_op:
                errors.append(
                    f"STALE PARAM  {op_id}.{field}: [fields.{op_id}] references "
                    f"an operation with no request body in the spec"
                )
            continue
        known = set(op_props[op_id])
        for field in per_op:
            if field not in known:
                errors.append(
                    f"STALE PARAM  {op_id}.{field}: field not present in the "
                    f"{op_id} request schema. Remove from [fields.{op_id}]."
                )

    # Note: we intentionally do NOT flag [fields.<op>] entries as "redundant"
    # when the field name also appears in main.rs. Field names like
    # `retain_mission` can be a struct variant field under one command (e.g.
    # `bank set-config`) while being legitimately skipped for another
    # (e.g. `bank create`, which the skip points to). A global redundancy
    # check can't tell those apart without a per-operation → command-variant
    # mapping, so we keep only the stricter checks above.

    total = len(spec_ops)
    impl_count = len(implemented & spec_ops)
    op_skip_count = len(op_skips.keys() & spec_ops)

    print("Hindsight CLI OpenAPI coverage check")
    print(f"  Spec:             {spec_path.relative_to(root)}")
    print(f"  Manifest:         {manifest_path.relative_to(root)}")
    print(f"  Operations:       {total}")
    print(f"    implemented:    {impl_count}")
    print(f"    skipped:        {op_skip_count}")
    print(f"  Request params:   {total_props}")
    print(f"    covered:        {covered_props}")
    print(f"    skipped:        {skipped_props}")
    print()

    if errors:
        print(f"FAILED: {len(errors)} issue(s):")
        for e in errors:
            print(f"  {e}")
        print()
        print(
            "Fix by either exposing the endpoint/field as a CLI arg, or adding\n"
            "an entry to hindsight-cli/.openapi-coverage.toml with a reason."
        )
        sys.exit(1)

    print(f"OK: all {total} operations and {total_props} request params covered.")


if __name__ == "__main__":
    main()
