"""Portable JSON Schemas for MCP tool inputs.

Pydantic renders a recursive model — the compound tag expression behind
``MentalModelTriggerInput.tag_groups`` is the live case — as a ``$ref`` that
ultimately points back at itself. That is valid JSON Schema, but not every
consumer accepts it: the Meta model API rejects the *whole* request with
``400 Recursive JSON schemas are not currently supported``, so an agent that
discovers one of these tools loses the turn that loaded it instead of getting a
tool it could not use. An MCP tool input schema is a wire contract whose other
end we do not control, so we hand out one every consumer can read.

``dereference_refs`` inlines each ``$ref`` against the schema's own ``$defs`` and,
where a reference would close a cycle, substitutes a permissive placeholder. The
result is acyclic by construction, still describes every finite instance the
original accepts, and drops the definitions that are no longer referenced.
"""

from __future__ import annotations

from typing import Any

# Bound the expansion so a pathological (non-cyclic) reference graph cannot grow
# the schema without limit. The recursive tag expression reaches four definitions
# deep, so this is generous headroom, not a tuning knob.
MAX_REF_DEPTH = 8

# What a reference becomes where inlining it would close a cycle (or exceed
# MAX_REF_DEPTH): an empty schema, which accepts any instance — the same contract
# the recursive branch had at the point the cycle was cut. A typed placeholder
# would be wrong for a node that is not that type, and stays valid here.
ANY_SCHEMA: dict[str, Any] = {}

# Keywords whose value is a map of *names* to subschemas rather than a subschema
# itself. Their keys are schema-author-controlled, so a property literally named
# "$ref" must not be mistaken for a reference while walking. Mirrors the guard in
# ``engine/structured_output.py::_strip_ref_siblings``.
_SUBSCHEMA_NAME_MAPS = frozenset({"$defs", "definitions", "properties", "patternProperties"})


def _resolve_ref(root: Any, ref: str) -> Any | None:
    """Resolve a local ``#/...`` JSON Pointer against ``root``; ``None`` if absent.

    Only local references are inlined. A remote or unresolvable ``$ref`` becomes
    the placeholder rather than being forwarded: it is exactly the kind of
    reference the consumers that motivated this module cannot follow.
    """
    if not ref.startswith("#/"):
        return None
    node = root
    for raw in ref[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return None
    return node


def dereference_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline every ``$ref`` in ``schema`` and cut recursion with ``ANY_SCHEMA``.

    Recursion is cut with a per-path guard: a ``$ref`` already expanded on the
    current path is replaced by ``ANY_SCHEMA``. That keeps the expansion faithful
    up to the first repeat — enough for an agent to see the shape of a nested tag
    expression — while guaranteeing the output is acyclic. The dropped ``$defs``
    were only there to hold the references that are now inlined.
    """

    # ``schema`` is a JSON Schema: a dynamic JSON document, the documented
    # exception to the "no raw dict for structured data" rule.
    def walk(node: Any, path: frozenset[str], depth: int) -> Any:
        if depth > MAX_REF_DEPTH:
            return dict(ANY_SCHEMA)
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref in path or depth >= MAX_REF_DEPTH:
                    return dict(ANY_SCHEMA)
                target = _resolve_ref(schema, ref)
                if target is None:
                    return dict(ANY_SCHEMA)
                inlined = walk(target, path | {ref}, depth + 1)
                siblings = {key: value for key, value in node.items() if key != "$ref"}
                if siblings and isinstance(inlined, dict):
                    # JSON Schema 2020-12 lets keywords sit beside $ref, and
                    # Pydantic emits `description` that way. Keep them: the
                    # sibling is the more specific statement.
                    return {**inlined, **walk(siblings, path, depth)}
                return inlined
            return {
                key: (
                    {name: walk(sub, path, depth) for name, sub in value.items()}
                    if key in _SUBSCHEMA_NAME_MAPS and isinstance(value, dict)
                    else walk(value, path, depth)
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item, path, depth) for item in node]
        return node

    inlined = walk(schema, frozenset(), 0)
    if isinstance(inlined, dict):
        inlined.pop("$defs", None)
        inlined.pop("definitions", None)
    return inlined
