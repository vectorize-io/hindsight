"""Every per-bank HTTP route answers to the operation validator (issue #5448).

A validator that refuses every hook must stop every ``/banks/{bank_id}/...`` route.
The sweep calls each route from the OpenAPI spec with a minimal body: a route may
still fail earlier on that dummy input (400/404/410/422), but it must never succeed
(the dry-run extract used to, running an LLM call under the bank's config) and never
turn the refusal into a 500 (the prompt preview used to).
"""

import re
import uuid
from typing import Any

import httpx
import pytest

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.extensions import OperationValidatorExtension, ValidationResult


class _RefuseEverything(OperationValidatorExtension):
    """Every ``validate_*`` hook refuses with 403. ``precheck`` keeps accepting: it is a
    billing gate only some routes wire, and refusing there would hide a route that
    skips the bank check behind it."""

    def __init__(self) -> None:
        super().__init__({})

    # The three abstract hooks must be defined for the class to instantiate; the
    # __getattribute__ below answers every other validate_* the same way.
    async def validate_retain(self, ctx: Any) -> ValidationResult:
        return ValidationResult.reject("refused", 403)

    async def validate_recall(self, ctx: Any) -> ValidationResult:
        return ValidationResult.reject("refused", 403)

    async def validate_reflect(self, ctx: Any) -> ValidationResult:
        return ValidationResult.reject("refused", 403)

    def __getattribute__(self, name: str) -> Any:
        # Covers hooks added later without touching this test.
        if name.startswith("validate_"):

            async def refuse(ctx: Any) -> ValidationResult:
                return ValidationResult.reject("refused", 403)

            return refuse
        return super().__getattribute__(name)


def _sample(schema: dict, spec: dict) -> Any:
    """The smallest value the schema accepts: required fields only, first enum/variant."""
    if "$ref" in schema:
        return _sample(spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]], spec)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = [o for o in schema[key] if o.get("type") != "null"]
            return _sample(options[0], spec) if options else None
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if schema.get("type") == "object" or "properties" in schema:
        required = schema.get("required", [])
        return {k: _sample(v, spec) for k, v in schema.get("properties", {}).items() if k in required}
    return {"string": "x", "integer": 1, "number": 1, "boolean": False, "array": []}.get(schema.get("type"), "x")


@pytest.mark.asyncio
async def test_every_bank_route_refuses_when_the_validator_does(memory, monkeypatch):
    bank_id = f"refused-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    app = create_app(memory, initialize_memory=False)
    spec = app.openapi()
    monkeypatch.setattr(memory, "_operation_validator", _RefuseEverything())

    leaks = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for path, operations in spec["paths"].items():
            if "{bank_id}" not in path:
                continue
            url = re.sub(
                r"\{(\w+)\}",
                lambda m: bank_id if m.group(1) == "bank_id" else "00000000-0000-0000-0000-000000000001",
                path,
            )
            for method, operation in operations.items():
                params = {
                    p["name"]: _sample(p.get("schema", {}), spec)
                    for p in operation.get("parameters", [])
                    if p["in"] == "query" and p.get("required")
                }
                kwargs: dict[str, Any] = {}
                body = operation.get("requestBody", {}).get("content", {})
                if "application/json" in body:
                    kwargs["json"] = _sample(body["application/json"]["schema"], spec)
                elif "multipart/form-data" in body:
                    kwargs["files"] = {"file": ("a.txt", b"hi")}
                resp = await client.request(method.upper(), url, params=params, **kwargs)
                # The minimal retain body has no items: the route has nothing to hand
                # the engine and answers a no-op 200 — nothing stored, nothing spent.
                if path.endswith("/memories") and method == "post" and resp.status_code == 200:
                    continue
                if resp.status_code < 400 or resp.status_code >= 500:
                    leaks.append(f"{resp.status_code} {method.upper()} {path}: {resp.text[:120]}")

    assert not leaks, "routes that ignored the validator's refusal:\n" + "\n".join(leaks)
