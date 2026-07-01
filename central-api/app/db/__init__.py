"""Control-plane persistence.

The control-plane database is the **source of truth** for ownership, permissions,
source metadata, sync status, audit, and operator governance — Google Drive state
must never live only inside vector-DB payloads.

SQLAlchemy Core (no ORM relationships), async engine. Portable column types so the
same schema runs on Postgres (prod, asyncpg) and SQLite (dev/test, aiosqlite).
"""

from app.db.engine import get_session, init_models, session_scope
from app.db.tables import metadata

__all__ = ["metadata", "get_session", "session_scope", "init_models"]
