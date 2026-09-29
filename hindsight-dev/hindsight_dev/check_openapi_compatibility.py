#!/usr/bin/env python3
"""
Check OpenAPI specification compatibility between two versions.

This script compares two OpenAPI specs and reports breaking changes:
- Backwards compatibility: Can old clients work with new API?
- Forwards compatibility: Can new clients work with old API?
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue."""

    severity: str  # "error", "warning"
    category: str  # "endpoint", "request", "response", "schema"
    path: str  # e.g., "/v1/banks/{bank_id}/mental-models POST"
    message: str


class OpenAPICompatibilityChecker:
    """Checks OpenAPI spec compatibility."""

    def __init__(self, old_spec: dict, new_spec: dict):
        self.old_spec = old_spec
        self.new_spec = new_spec
        self.issues: list[CompatibilityIssue] = []

    def check_backwards_compatibility(self) -> list[CompatibilityIssue]:
        """
        Check backwards compatibility (can old clients work with new API?).

        Breaking changes:
        - Removing endpoints
        - Adding required request fields
        - Removing response fields (that were required)
        - Changing types
        """
        self.issues = []

        # Check endpoints
        self._check_removed_endpoints()

        # Check each endpoint's request/response
        for path, methods in self.old_spec.get("paths", {}).items():
            for method, old_endpoint in methods.items():
                if method.upper() not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    continue

                new_endpoint = self.new_spec.get("paths", {}).get(path, {}).get(method)
                if not new_endpoint:
                    continue  # Already reported in _check_removed_endpoints

                endpoint_path = f"{path} {method.upper()}"

                # Check request body
                self._check_request_compatibility(endpoint_path, old_endpoint, new_endpoint)

                # Check response
                self._check_response_compatibility(endpoint_path, old_endpoint, new_endpoint)

        return self.issues

    def _check_removed_endpoints(self):
        """Check if any endpoints were removed."""
        old_endpoints = set()
        new_endpoints = set()

        for path, methods in self.old_spec.get("paths", {}).items():
            for method in methods.keys():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    old_endpoints.add(f"{path} {method.upper()}")

        for path, methods in self.new_spec.get("paths", {}).items():
            for method in methods.keys():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    new_endpoints.add(f"{path} {method.upper()}")

        removed = old_endpoints - new_endpoints
        for endpoint in removed:
            self.issues.append(
                CompatibilityIssue(
                    severity="error",
                    category="endpoint",
                    path=endpoint,
                    message="Endpoint removed (breaks old clients)",
                )
            )

    def _check_request_compatibility(self, endpoint_path: str, old_endpoint: dict, new_endpoint: dict):
        """Check request body compatibility."""
        old_request = old_endpoint.get("requestBody", {})
        new_request = new_endpoint.get("requestBody", {})

        if not old_request and not new_request:
            return

        # Get schema references
        old_schema_ref = self._get_request_schema_ref(old_request)
        new_schema_ref = self._get_request_schema_ref(new_request)

        if not old_schema_ref or not new_schema_ref:
            return

        old_schema = self._resolve_schema_ref(self.old_spec, old_schema_ref)
        new_schema = self._resolve_schema_ref(self.new_spec, new_schema_ref)

        if not old_schema or not new_schema:
            return

        # Check if new required fields were added
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))

        added_required = new_required - old_required
        if added_required:
            self.issues.append(
                CompatibilityIssue(
                    severity="error",
                    category="request",
                    path=endpoint_path,
                    message=f"Added required request fields: {', '.join(sorted(added_required))} (breaks old clients)",
                )
            )

    def _check_response_compatibility(self, endpoint_path: str, old_endpoint: dict, new_endpoint: dict):
        """Check response compatibility."""
        old_responses = old_endpoint.get("responses", {})
        new_responses = new_endpoint.get("responses", {})

        # Check 200/201 responses (most common success responses)
        for status_code in ["200", "201"]:
            old_response = old_responses.get(status_code)
            new_response = new_responses.get(status_code)

            if not old_response or not new_response:
                continue

            old_schema_ref = self._get_response_schema_ref(old_response)
            new_schema_ref = self._get_response_schema_ref(new_response)

            if not old_schema_ref or not new_schema_ref:
                continue

            old_schema = self._resolve_schema_ref(self.old_spec, old_schema_ref)
            new_schema = self._resolve_schema_ref(self.new_spec, new_schema_ref)

            if not old_schema or not new_schema:
                continue

            # Check if fields were completely removed from properties
            old_properties = set(old_schema.get("properties", {}).keys())
            new_properties = set(new_schema.get("properties", {}).keys())

            removed_properties = old_properties - new_properties
            if removed_properties:
                self.issues.append(
                    CompatibilityIssue(
                        severity="error",
                        category="response",
                        path=f"{endpoint_path} [{status_code}]",
                        message=f"Removed response fields: {', '.join(sorted(removed_properties))} (breaks old clients)",
                    )
                )

            # Check if required response fields were made optional (this is OK for backwards compatibility)
            old_required = set(old_schema.get("required", []))
            new_required = set(new_schema.get("required", []))

            made_optional = old_required - new_required
            # Only report if the field wasn't completely removed (handled above)
            made_optional = made_optional - removed_properties
            if made_optional:
                # This is backwards compatible but worth noting
                self.issues.append(
                    CompatibilityIssue(
                        severity="warning",
                        category="response",
                        path=f"{endpoint_path} [{status_code}]",
                        message=f"Made response fields optional: {', '.join(sorted(made_optional))} (backwards compatible but may indicate API instability)",
                    )
                )

            # Adding required response fields is a warning (not breaking but unexpected for old clients)
            added_required = new_required - old_required
            if added_required:
                # This is actually OK for backwards compatibility - old clients can ignore new fields
                # But it's worth noting
                self.issues.append(
                    CompatibilityIssue(
                        severity="warning",
                        category="response",
                        path=f"{endpoint_path} [{status_code}]",
                        message=f"Added required response fields: {', '.join(sorted(added_required))} (old clients may not expect these)",
                    )
                )

    def _get_request_schema_ref(self, request_body: dict) -> str | None:
        """Extract schema $ref from request body."""
        content = request_body.get("content", {})
        json_content = content.get("application/json", {})
        return json_content.get("schema", {}).get("$ref")

    def _get_response_schema_ref(self, response: dict) -> str | None:
        """Extract schema $ref from response."""
        content = response.get("content", {})
        json_content = content.get("application/json", {})
        return json_content.get("schema", {}).get("$ref")

    def _resolve_schema_ref(self, spec: dict, ref: str) -> dict | None:
        """Resolve a $ref to its schema definition."""
        if not ref or not ref.startswith("#/"):
            return None

        parts = ref[2:].split("/")
        current = spec
        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None

        return current


def load_openapi_spec(path: Path) -> dict:
    """Load OpenAPI spec from JSON file."""
    with open(path) as f:
        return json.load(f)


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: check-openapi-compatibility <old-spec.json> <new-spec.json>")
        print()
        print("Examples:")
        print("  # Check against main branch")
        print("  git show main:hindsight-docs/static/openapi.json > /tmp/old-openapi.json")
        print("  check-openapi-compatibility /tmp/old-openapi.json hindsight-docs/static/openapi.json")
        print()
        print("  # Check against a specific commit")
        print("  git show abc123:hindsight-docs/static/openapi.json > /tmp/old-openapi.json")
        print("  check-openapi-compatibility /tmp/old-openapi.json hindsight-docs/static/openapi.json")
        sys.exit(1)

    old_spec_path = Path(sys.argv[1])
    new_spec_path = Path(sys.argv[2])

    if not old_spec_path.exists():
        print(f"Error: Old spec file not found: {old_spec_path}")
        sys.exit(1)

    if not new_spec_path.exists():
        print(f"Error: New spec file not found: {new_spec_path}")
        sys.exit(1)

    print("Checking OpenAPI compatibility...")
    print(f"  Old spec: {old_spec_path}")
    print(f"  New spec: {new_spec_path}")
    print()

    old_spec = load_openapi_spec(old_spec_path)
    new_spec = load_openapi_spec(new_spec_path)

    checker = OpenAPICompatibilityChecker(old_spec, new_spec)
    issues = checker.check_backwards_compatibility()

    if not issues:
        print("✓ No backwards compatibility issues found!")
        sys.exit(0)

    # Group issues by severity
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        print(f"✗ Found {len(errors)} backwards compatibility error(s):")
        print()
        for issue in errors:
            print(f"  [{issue.category.upper()}] {issue.path}")
            print(f"    {issue.message}")
            print()

    if warnings:
        print(f"⚠ Found {len(warnings)} warning(s):")
        print()
        for issue in warnings:
            print(f"  [{issue.category.upper()}] {issue.path}")
            print(f"    {issue.message}")
            print()

    if errors:
        print("❌ Backwards compatibility check failed!")
        sys.exit(1)
    else:
        print("✓ Backwards compatibility check passed (with warnings)")
        sys.exit(0)


if __name__ == "__main__":
    main()
