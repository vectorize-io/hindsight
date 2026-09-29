"""
Built-in extension implementations.

These are the extensions that ship with the server: they have no dependencies
beyond the Hindsight core and are useful to any deployment. Everything else —
integrations with a specific identity provider, vendor, or deployment style —
lives in its own package under ``hindsight-extensions/`` in the repository and
is installed alongside the server.

Available built-in extensions:
    - ApiKeyTenantExtension: Simple API key validation with public schema
    - MemoryDefenseRegexExtension: Regex-based memory defense policies

Concrete classes are deliberately not re-exported from
``hindsight_api.extensions``: extensions are resolved by import path at load
time, so re-exporting them would make every extension's imports (and therefore
its dependencies) part of the core import graph.

Example usage:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
"""

from hindsight_api.extensions.builtin.memory_defense_regex import MemoryDefenseRegexExtension
from hindsight_api.extensions.builtin.tenant import ApiKeyTenantExtension

__all__ = [
    "ApiKeyTenantExtension",
    "MemoryDefenseRegexExtension",
]
