"""Supabase organization authz operation definitions."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Literal, TypedDict

from pydantic import BaseModel

OperationSource = Literal["bank_read", "bank_write", "special_bank", "unscoped"]
OperationScope = Literal["bank", "unscoped"]


class OperationDefinition(TypedDict, total=False):
    name: str
    source: OperationSource
    action: Literal["read", "write"]
    scope: OperationScope


class NamedOperation(BaseModel):
    name: str
    action: Literal["read", "write"]


class OperationManifest(BaseModel):
    bank_read: list[str]
    bank_write: list[str]
    special_bank: list[NamedOperation]
    unscoped: list[NamedOperation]


@lru_cache(maxsize=1)
def load_operation_definitions() -> tuple[OperationDefinition, ...]:
    """Load the shared Python/TypeScript API-key operation manifest."""
    manifest_path = files(__package__).joinpath("supabase_authz_operations.json")
    manifest = OperationManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    definitions: list[OperationDefinition] = []
    for name in manifest.bank_read:
        definitions.append({"name": name, "source": "bank_read", "action": "read", "scope": "bank"})
    for name in manifest.bank_write:
        definitions.append({"name": name, "source": "bank_write", "action": "write", "scope": "bank"})
    definitions.extend(
        {"name": item.name, "source": "special_bank", "action": item.action, "scope": "bank"}
        for item in manifest.special_bank
    )
    definitions.extend(
        {"name": item.name, "source": "unscoped", "action": item.action, "scope": "unscoped"}
        for item in manifest.unscoped
    )
    return tuple(definitions)


def operation_names_for_source(source: OperationSource) -> frozenset[str]:
    return frozenset(operation["name"] for operation in load_operation_definitions() if operation["source"] == source)


def operation_names_for_scope(scope: OperationScope) -> frozenset[str]:
    return frozenset(operation["name"] for operation in load_operation_definitions() if operation["scope"] == scope)
