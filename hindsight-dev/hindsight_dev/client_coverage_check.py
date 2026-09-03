#!/usr/bin/env python3
"""
Check that the Python and TypeScript high-level client wrappers expose every
request-body parameter from the OpenAPI spec.

The generated (low-level) clients are auto-generated from the spec and are
always in sync.  The *wrapper* clients (``hindsight_client.py`` for Python,
``src/index.ts`` for TypeScript) are hand-written convenience layers that
re-export a curated surface.  This script catches fields that are present in
the OpenAPI spec but missing from the wrapper.

For each client, the script:
  1. Discovers which OpenAPI operations the wrapper covers by scanning for
     SDK call sites (e.g. ``sdk.recallMemories(`` or
     ``self._memory_api.recall_memories(``).
  2. For each covered operation, checks that every request-body property
     appears somewhere in the wrapper source around that call site (with
     automatic snake_case ↔ camelCase conversion).
  3. Allows explicit skips via per-client ``.openapi-coverage.toml`` files.

Bank config is checked separately.  ``PATCH /banks/{bank_id}/config`` takes a
free-form ``updates: dict[str, Any]``, so the spec advertises a single ``updates``
property and every individual setting is invisible to the request-body walk above.
The wrappers enumerate those settings as keyword arguments, which is exactly where
they drift, so ``check_bank_config_fields`` reads the server's ``_CONFIGURABLE_FIELDS``
directly and asserts each one reaches both wrappers.  Skips live in the same manifest
under ``[bank_config]``.

Usage:
    cd hindsight-dev
    uv run client-coverage-check
    uv run client-coverage-check --client python
    uv run client-coverage-check --client typescript
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def get_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "hindsight-clients").is_dir() and (parent / "hindsight-docs").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repo root from {here}")


# ── helpers ──────────────────────────────────────────────────────────────


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def load_openapi_operations(spec_path: Path) -> dict[str, list[str]]:
    """Return {operation_id: [property_names]} for operations with request bodies."""
    spec = json.loads(spec_path.read_text())
    schemas = spec.get("components", {}).get("schemas", {})
    result: dict[str, list[str]] = {}

    for path_item in spec.get("paths", {}).values():
        for method, op in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
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
                result[op_id] = props

    return result


def load_manifest(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Return (operation-level skips, per-operation field skips) from a TOML manifest."""
    if not path.exists():
        return {}, {}
    data = tomllib.loads(path.read_text())

    skip_raw = data.get("skip", {}) or {}
    op_skips: dict[str, str] = {}
    for op_id, reason in skip_raw.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: skip.{op_id} must be a non-empty string reason")
        op_skips[op_id] = reason.strip()

    fields_raw = data.get("fields", {}) or {}
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


# ── per-client checking ─────────────────────────────────────────────────


def field_present_in_source(prop: str, source: str) -> bool:
    """Check if a field name (snake_case from OpenAPI) appears in wrapper source.

    Checks both snake_case and camelCase variants, looking for the field
    as an identifier (word boundary on at least one side).
    """
    variants = {prop, snake_to_camel(prop)}
    for v in variants:
        # Match as an identifier: preceded by word boundary or common delimiters
        # and followed by word boundary or common delimiters
        if re.search(r"(?:^|[\s{,.(:\[\"'])" + re.escape(v) + r"(?:$|[\s},:)?\[\"'=])", source):
            return True
    return False


def check_client(
    *,
    client_name: str,
    source_path: Path,
    manifest_path: Path,
    op_props: dict[str, list[str]],
    find_covered_ops: Callable[[str, set[str]], set[str]],
) -> list[str]:
    """Check a single client wrapper. Returns list of error strings."""
    source = source_path.read_text()
    op_skips, field_skips = load_manifest(manifest_path)
    errors: list[str] = []

    # Find which operations this wrapper covers
    covered_ops = find_covered_ops(source, set(op_props.keys()))

    # Check that skipped operations actually exist and aren't covered
    for op_id in sorted(op_skips.keys()):
        if op_id not in op_props:
            errors.append(f"STALE OP     {op_id}: listed in [skip] but has no request body in the spec")

    # Check fields for each covered operation
    total_props = 0
    covered_props = 0
    skipped_props = 0

    for op_id in sorted(covered_ops):
        props = op_props.get(op_id, [])
        if op_id in op_skips:
            continue
        per_op_skips = field_skips.get(op_id, {})
        for prop in props:
            total_props += 1
            if prop in per_op_skips:
                skipped_props += 1
                continue
            if field_present_in_source(prop, source):
                covered_props += 1
                continue
            errors.append(
                f"MISSING PARAM {op_id}.{prop}: request body field not found in "
                f"wrapper source (checked: {prop}, {snake_to_camel(prop)})"
            )

    # Stale field skips
    for op_id, per_op in field_skips.items():
        if op_id not in op_props:
            for field in per_op:
                errors.append(
                    f"STALE PARAM  {op_id}.{field}: [fields.{op_id}] references "
                    f"an operation with no request body in the spec"
                )
            continue
        known = set(op_props.get(op_id, []))
        for field in per_op:
            if field not in known:
                errors.append(
                    f"STALE PARAM  {op_id}.{field}: field not present in the "
                    f"{op_id} request schema. Remove from [fields.{op_id}]."
                )

    print(f"  [{client_name}]")
    print(f"    Source:          {source_path}")
    print(f"    Manifest:        {manifest_path}")
    print(f"    Operations:      {len(covered_ops)} covered")
    print(f"    Request params:  {total_props}")
    print(f"      covered:       {covered_props}")
    print(f"      skipped:       {skipped_props}")
    print(f"      missing:       {total_props - covered_props - skipped_props}")

    return errors


# ── bank config ──────────────────────────────────────────────────────────


def load_configurable_fields(config_path: Path) -> list[str]:
    """Return the server's per-bank overridable config field names.

    Parsed out of ``config.py`` with ast rather than imported: hindsight-dev does
    not depend on hindsight-api, and this only needs the literal set members.
    """
    tree = ast.parse(config_path.read_text())
    for node in ast.walk(tree):
        # Bind the value inside each branch: narrowing on `node` does not survive
        # the if/elif when only `targets` is assigned in them.
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "_CONFIGURABLE_FIELDS" for t in targets):
            continue
        if not isinstance(value, ast.Set | ast.List | ast.Tuple):
            raise ValueError(f"{config_path}: _CONFIGURABLE_FIELDS is not a literal set/list/tuple")
        return sorted(elt.value for elt in value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str))
    raise ValueError(f"{config_path}: no _CONFIGURABLE_FIELDS assignment found")


def load_bank_config_skips(path: Path) -> dict[str, str]:
    """Return the ``[bank_config]`` skip table from a client's coverage manifest."""
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text())
    raw = data.get("bank_config", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: [bank_config] must be a table")
    skips: dict[str, str] = {}
    for field, reason in raw.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: bank_config.{field} must be a non-empty string reason")
        skips[field] = reason.strip()
    return skips


def extract_config_method(source: str, client_name: str) -> str:
    """Return just the wrapper's bank-config update method body.

    Scoped rather than scanning the whole file: these field names also appear in
    other methods (``recall_max_tokens`` is a recall argument, ``retain_chunk_size``
    a retain one), so a whole-file search reports a field as reachable when the
    config method does not actually accept it.
    """
    if client_name == "python":
        start = re.search(r"^    def update_bank_config\(", source, re.M)
        end_pat = r"^    (?:async )?def "
    else:
        start = re.search(r"^  async updateBankConfig\(", source, re.M)
        end_pat = r"^  (?:async )?\w+\("
    if start is None:
        raise ValueError(f"{client_name}: could not locate the bank-config update method")
    rest = source[start.end() :]
    end = re.search(end_pat, rest, re.M)
    return rest[: end.start()] if end else rest


def forwarding_pattern(field: str, client_name: str) -> str:
    """Regex proving a field is written into the PATCH body, not just accepted."""
    if client_name == "python":
        # "field_name": field_name,  inside the updates dict comprehension
        return r'"' + re.escape(field) + r'"\s*:'
    # updates.field_name = options.fieldName;
    return r"updates\." + re.escape(field) + r"\s*="


def check_bank_config_fields(
    *,
    client_name: str,
    source_path: Path,
    manifest_path: Path,
    fields: list[str],
) -> list[str]:
    """Check that every server-configurable bank field reaches one client wrapper."""
    source = extract_config_method(source_path.read_text(), client_name)
    skips = load_bank_config_skips(manifest_path)
    errors: list[str] = []

    covered = skipped = 0
    for field in fields:
        if field in skips:
            skipped += 1
            continue
        if not field_present_in_source(field, source):
            errors.append(f"MISSING CONFIG {field}: not accepted by the wrapper's bank-config method")
            continue
        # Accepting the argument is not enough — both wrappers build the request
        # body by enumerating fields, so one that is declared but never written
        # into that map is accepted and then silently dropped.
        if not re.search(forwarding_pattern(field, client_name), source):
            errors.append(
                f"UNFORWARDED CONFIG {field}: accepted by the wrapper but never "
                f"written into the request body — it would be silently dropped"
            )
            continue
        covered += 1

    for field in sorted(skips):
        if field not in fields:
            errors.append(
                f"STALE CONFIG  {field}: listed in [bank_config] but not in "
                f"the server's _CONFIGURABLE_FIELDS. Remove it."
            )

    print(f"    Bank config:     {len(fields)}")
    print(f"      covered:       {covered}")
    print(f"      skipped:       {skipped}")
    print(f"      missing:       {len(fields) - covered - skipped}")

    return errors


# ── TypeScript ───────────────────────────────────────────────────────────


def find_ts_covered_ops(source: str, op_ids: set[str]) -> set[str]:
    """Find operations called via ``sdk.<operationId>(`` in TS wrapper.

    The TS generated SDK uses camelCase method names (e.g. ``retainMemories``)
    while OpenAPI operation IDs are snake_case (``retain_memories``).  We check
    both variants.
    """
    covered: set[str] = set()
    for op_id in op_ids:
        camel = snake_to_camel(op_id)
        for variant in (op_id, camel):
            if re.search(r"sdk\.\s*" + re.escape(variant) + r"\s*\(", source):
                covered.add(op_id)
                break
    return covered


# ── Python ───────────────────────────────────────────────────────────────


def find_py_covered_ops(source: str, op_ids: set[str]) -> set[str]:
    """Find operations called via ``self._*_api.<operation_id>(`` in Python wrapper."""
    covered: set[str] = set()
    for op_id in op_ids:
        if re.search(r"self\._\w+_api\.\s*" + re.escape(op_id) + r"\s*\(", source):
            covered.add(op_id)
    return covered


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Check client SDK wrapper coverage against OpenAPI spec")
    parser.add_argument(
        "--client",
        choices=["python", "typescript", "all"],
        default="all",
        help="Which client to check (default: all)",
    )
    args = parser.parse_args()

    root = get_repo_root()
    spec_path = root / "hindsight-docs" / "static" / "openapi.json"

    if not spec_path.exists():
        print(f"ERROR: OpenAPI spec not found at {spec_path}", file=sys.stderr)
        print("       Run ./scripts/generate-openapi.sh first.", file=sys.stderr)
        sys.exit(1)

    op_props = load_openapi_operations(spec_path)

    config_path = root / "hindsight-api-slim" / "hindsight_api" / "config.py"
    bank_config_fields = load_configurable_fields(config_path)

    clients: list[dict] = []

    if args.client in ("python", "all"):
        clients.append(
            {
                "client_name": "python",
                "source_path": root / "hindsight-clients" / "python" / "hindsight_client" / "hindsight_client.py",
                "manifest_path": root / "hindsight-clients" / "python" / ".openapi-coverage.toml",
                "find_covered_ops": find_py_covered_ops,
            }
        )

    if args.client in ("typescript", "all"):
        clients.append(
            {
                "client_name": "typescript",
                "source_path": root / "hindsight-clients" / "typescript" / "src" / "index.ts",
                "manifest_path": root / "hindsight-clients" / "typescript" / ".openapi-coverage.toml",
                "find_covered_ops": find_ts_covered_ops,
            }
        )

    all_errors: list[str] = []

    print("Client SDK OpenAPI coverage check")
    print(f"  Spec:        {spec_path.relative_to(root)}")
    print(f"  Bank config: {config_path.relative_to(root)} ({len(bank_config_fields)} fields)")
    print()

    for client_cfg in clients:
        if not client_cfg["source_path"].exists():
            print(f"  [{client_cfg['client_name']}] SKIPPED: source not found at {client_cfg['source_path']}")
            continue
        errs = check_client(op_props=op_props, **client_cfg)
        errs += check_bank_config_fields(
            client_name=client_cfg["client_name"],
            source_path=client_cfg["source_path"],
            manifest_path=client_cfg["manifest_path"],
            fields=bank_config_fields,
        )
        all_errors.extend(f"[{client_cfg['client_name']}] {e}" for e in errs)
        print()

    if all_errors:
        print(f"FAILED: {len(all_errors)} issue(s):")
        for e in all_errors:
            print(f"  {e}")
        print()
        print(
            "Fix by either exposing the field in the wrapper, or adding\n"
            "an entry to the client's .openapi-coverage.toml with a reason."
        )
        sys.exit(1)

    print("OK: all covered operations have their request params accounted for.")


if __name__ == "__main__":
    main()
