"""Shared test fixtures.

Each test gets its own SQLite file (``tmp_path``) so the control-plane DB is
isolated. The global engine is cleared before/after so it rebuilds against the
per-test URL. DB-flow tests run their whole scenario inside one ``asyncio.run``
to avoid reusing an async pool across event loops.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.config import settings
from app.db import engine as engine_mod


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    original = settings.database_url
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path / (uuid.uuid4().hex + '.db')}"
    engine_mod.clear_engine()
    
    # Ensure tables exist for test
    loop = asyncio.new_event_loop()
    loop.run_until_complete(engine_mod.init_models())
    loop.close()
    
    yield
    engine_mod.clear_engine()
    settings.database_url = original
