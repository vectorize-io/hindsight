import pytest

from hindsight_api.engine.memory_engine import _HEALTH_REQUIRED_TABLES, MemoryEngine


class _AcquireContext:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeBackend:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _AcquireContext(self.conn)


class _FakeConn:
    def __init__(self, present_tables):
        self.present_tables = present_tables

    async def fetchval(self, query, *args, **kwargs):
        assert query == "SELECT 1"
        return 1

    async def fetch(self, query, *args, **kwargs):
        assert "information_schema.tables" in query
        return [{"table_name": name} for name in self.present_tables]


async def _engine_with_tables(present_tables):
    engine = object.__new__(MemoryEngine)
    engine._initialized = True
    backend = _FakeBackend(_FakeConn(present_tables))

    async def _get_backend():
        return backend

    engine._get_backend = _get_backend
    return engine


@pytest.mark.asyncio
async def test_health_check_requires_core_schema_tables():
    engine = await _engine_with_tables(_HEALTH_REQUIRED_TABLES - {"memory_units"})

    health = await engine.health_check()

    assert health["status"] == "unhealthy"
    assert health["database"] == "connected"
    assert health["reason"] == "missing_schema_tables"
    assert "memory_units" in health["missing_tables"]


@pytest.mark.asyncio
async def test_health_check_reports_healthy_when_core_schema_exists():
    engine = await _engine_with_tables(_HEALTH_REQUIRED_TABLES)

    health = await engine.health_check()

    assert health["status"] == "healthy"
    assert health["database"] == "connected"
    assert health["schema"]
