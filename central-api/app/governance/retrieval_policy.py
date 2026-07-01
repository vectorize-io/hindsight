"""Retrieval governance — fail closed.

A document chunk may be retrieved only if the requesting CollabMind principal is
allowed to access the original source file. Every uncertainty denies:

    unknown permission   → deny
    missing metadata     → deny
    disabled document    → deny
    inactive connector   → deny
    not a workspace member → deny
    approval required but not approved → deny

Pure function over plain dicts so it is trivially unit-testable and reused by
both retrieval and the MCP search tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.db.ids import utcnow

# Drive roles that grant read access.
_READ_ROLES = frozenset({"owner", "organizer", "writer", "commenter", "reader"})


@dataclass(frozen=True)
class Principal:
    """Who is asking. ``email`` and ``domain`` are matched against Drive perms."""

    user_id: str
    email: str | None = None
    domain: str | None = None
    is_workspace_member: bool = False
    # Agents may be restricted to non-restricted sources; operators bypass less.
    agent_level: str = "standard"  # standard|restricted|operator


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def _deny(reason: str) -> Decision:
    return Decision(False, reason)


def _permission_grants(principal: Principal, perm: dict) -> bool:
    if perm.get("role") not in _READ_ROLES:
        return False
    exp = perm.get("expiration_time")
    if isinstance(exp, datetime) and exp < utcnow():
        return False
    ptype = perm.get("ptype")
    if ptype == "anyone":
        return True
    if ptype == "domain":
        return bool(principal.domain) and principal.domain == perm.get("domain")
    if ptype in ("user", "group"):
        return bool(principal.email) and principal.email == perm.get("email_address")
    return False  # unknown type → no grant


def evaluate_retrieval(*, principal: Principal, connector: dict | None,
                       document: dict | None, permissions: list[dict] | None) -> Decision:
    """Decide whether ``principal`` may retrieve from ``document``. Fail closed."""
    if not principal.is_workspace_member:
        return _deny("not_workspace_member")
    if connector is None:
        return _deny("missing_connector")
    if connector.get("status") != "connected":
        return _deny("inactive_connector")
    if document is None:
        return _deny("missing_source_metadata")
    if document.get("trashed"):
        return _deny("source_trashed")
    if not document.get("enabled", False):
        return _deny("document_disabled")
    if permissions is None:
        return _deny("unknown_permission")  # snapshot never taken → fail closed
    if not any(_permission_grants(principal, p) for p in permissions):
        return _deny("not_permitted_on_source")
    return Decision(True, "permitted")
