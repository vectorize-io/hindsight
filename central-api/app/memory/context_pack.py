"""Context-pack builder — serialization, versioning, caching."""

import json
import hashlib
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session


class ContextPack:
    """Serialized context snapshot with versioning."""

    def __init__(
        self,
        tenant_id: str,
        content: dict[str, Any],
        version: int = 1,
        tag: str = "default",
    ):
        self.tenant_id = tenant_id
        self.content = content
        self.version = version
        self.tag = tag
        self.created_at = datetime.utcnow()
        self.content_hash = self._hash_content()

    def _hash_content(self) -> str:
        """Compute SHA256 hash of content."""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "tenant_id": self.tenant_id,
            "version": self.version,
            "tag": self.tag,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "content": self.content,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


class ContextPackBuilder:
    """Build context-packs from execution/governance/memory state."""

    def __init__(self, session: AsyncSession, tenant_id: str):
        self.session = session
        self.tenant_id = tenant_id

    async def build(self, tag: str = "default", version: int = 1) -> ContextPack:
        """Build context-pack from current state.
        
        Collects:
        - Recent execution ledger entries
        - Governance quarantine status
        - Auth context
        - System metadata
        """
        content = {
            "tenant_id": self.tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_ledger": await self._fetch_ledger(),
            "governance_state": await self._fetch_governance(),
            "metadata": self._build_metadata(),
        }
        return ContextPack(self.tenant_id, content, version, tag)

    async def _fetch_ledger(self) -> list[dict]:
        """Fetch recent execution ledger entries (stub)."""
        return []

    async def _fetch_governance(self) -> dict:
        """Fetch governance state (stub)."""
        return {"quarantine_items": 0, "approval_decisions": 0}

    def _build_metadata(self) -> dict:
        """Build system metadata."""
        return {
            "builder_version": "1.0.0",
            "compression": "none",
            "format": "json",
        }


async def create_context_pack(
    tenant_id: str, tag: str = "default", version: int = 1, session: AsyncSession = None
) -> ContextPack:
    """Factory to create context-pack."""
    if session is None:
        # For standalone usage; in routes use dependency injection
        from app.db.engine import session_scope
        async with session_scope() as s:
            builder = ContextPackBuilder(s, tenant_id)
            return await builder.build(tag, version)
    else:
        builder = ContextPackBuilder(session, tenant_id)
        return await builder.build(tag, version)
