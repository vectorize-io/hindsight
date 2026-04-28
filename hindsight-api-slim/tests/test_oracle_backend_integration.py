"""
Integration tests for the OracleBackend + OracleDialect abstractions against a real Oracle 23ai instance.

Validates that our DatabaseBackend / DatabaseConnection / SQLDialect abstractions
produce correct results when talking to a real Oracle database.  Every test mirrors
a PostgreSQL pattern used in the Hindsight engine so that passing here means the
abstraction is ready to replace raw asyncpg usage.

Requires:
  - Oracle 23ai instance (local Docker or OCI)
  - pip install oracledb
  - Set ORACLE_TEST_DSN (URL or bare DSN format)

Run:
    docker run -d --name oracle-test -p 1521:1521 -e ORACLE_PWD=oracle \
      container-registry.oracle.com/database/free:latest

    ORACLE_TEST_DSN=oracle://SYSTEM:oracle@localhost:1521/FREEPDB1 \
    uv run pytest tests/test_oracle_backend_integration.py -v -m oracle -n0
"""

import array
import json
import os
import uuid

import pytest

try:
    import oracledb

    oracledb.defaults.fetch_lobs = False
    ORACLEDB_AVAILABLE = True
except ImportError:
    ORACLEDB_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not ORACLEDB_AVAILABLE, reason="oracledb not installed"),
    pytest.mark.skipif(not os.getenv("ORACLE_TEST_DSN"), reason="ORACLE_TEST_DSN not set"),
]


def to_vector32(floats: list[float]) -> array.array:
    """Convert a list of floats to array.array('f') for Oracle VECTOR binding."""
    return array.array("f", floats)


SCHEMA_PREFIX = "hs_be"


# ---------------------------------------------------------------------------
# Fixtures — schema lifecycle + backend/dialect creation
# ---------------------------------------------------------------------------


def _parse_oracle_test_dsn() -> dict:
    """Parse ORACLE_TEST_DSN into (user, password, bare_dsn).

    Accepts URL format ``oracle://user:pass@host:port/service`` or bare DSN
    ``host:port/service`` with separate ORACLE_TEST_USER / ORACLE_TEST_PASSWORD
    env vars.
    """
    from urllib.parse import urlparse

    dsn = os.environ["ORACLE_TEST_DSN"]
    parsed = urlparse(dsn)
    if parsed.scheme in ("oracle", "oracle+oracledb"):
        host = parsed.hostname or "localhost"
        port = parsed.port or 1521
        service = parsed.path.lstrip("/") if parsed.path else "FREEPDB1"
        return {
            "user": parsed.username or "SYSTEM",
            "password": parsed.password or "oracle",
            "dsn": f"{host}:{port}/{service}",
        }
    return {
        "user": os.environ.get("ORACLE_TEST_USER", "SYSTEM"),
        "password": os.environ.get("ORACLE_TEST_PASSWORD", "oracle"),
        "dsn": dsn,
    }


@pytest.fixture(scope="session")
def oracle_dsn():
    return _parse_oracle_test_dsn()["dsn"]


@pytest.fixture(scope="session")
def oracle_user():
    return _parse_oracle_test_dsn()["user"]


@pytest.fixture(scope="session")
def oracle_password():
    return _parse_oracle_test_dsn()["password"]


@pytest.fixture(scope="session")
def sync_pool(oracle_dsn, oracle_user, oracle_password):
    """Session-scoped synchronous pool for DDL setup/teardown."""
    pool = oracledb.create_pool(user=oracle_user, password=oracle_password, dsn=oracle_dsn, min=1, max=4)
    yield pool
    pool.close()


@pytest.fixture(scope="session")
def test_schema(sync_pool):
    """Create an isolated test schema (Oracle user) for the session."""
    schema = f"{SCHEMA_PREFIX}_{uuid.uuid4().hex[:8]}".upper()
    with sync_pool.acquire() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f'CREATE USER {schema} IDENTIFIED BY "testpass" DEFAULT TABLESPACE USERS QUOTA UNLIMITED ON USERS'
        )
        cursor.execute(f"GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE TO {schema}")
        try:
            cursor.execute(f"GRANT EXECUTE ON UTL_MATCH TO {schema}")
        except Exception:
            pass
        conn.commit()
    yield schema
    with sync_pool.acquire() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sid, serial# FROM v$session WHERE username = :1", [schema])
        for sid, serial in cursor.fetchall():
            try:
                cursor.execute(f"ALTER SYSTEM KILL SESSION '{sid},{serial}' IMMEDIATE")
            except Exception:
                pass
        try:
            cursor.execute(f"DROP USER {schema} CASCADE")
            conn.commit()
        except Exception:
            pass


@pytest.fixture(scope="session")
def setup_tables(sync_pool, test_schema):
    """Create tables in the test schema matching Hindsight's schema."""
    # Connect as admin to create tables in the test user's schema
    with sync_pool.acquire() as conn:
        cursor = conn.cursor()
        cursor.execute(f'ALTER SESSION SET CURRENT_SCHEMA = "{test_schema}"')

        cursor.execute("""
            CREATE TABLE banks (
                bank_id       VARCHAR2(255) PRIMARY KEY,
                name          VARCHAR2(500),
                disposition   CLOB CHECK (disposition IS JSON),
                background    CLOB,
                internal_id   RAW(16) DEFAULT SYS_GUID(),
                created_at    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
                updated_at    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE memory_units (
                id              RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
                bank_id         VARCHAR2(255),
                text            CLOB,
                embedding       VECTOR(384, FLOAT32),
                context         CLOB,
                event_date      TIMESTAMP WITH TIME ZONE,
                fact_type       VARCHAR2(50),
                metadata        CLOB CHECK (metadata IS JSON),
                tags            CLOB CHECK (tags IS JSON),
                created_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE entities (
                id              RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
                canonical_name  VARCHAR2(500),
                bank_id         VARCHAR2(255),
                first_seen      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
                last_seen       TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
                mention_count   NUMBER DEFAULT 0,
                CONSTRAINT uq_entity_bank_name UNIQUE (bank_id, canonical_name)
            )
        """)

        cursor.execute("""
            CREATE TABLE memory_links (
                from_unit_id RAW(16),
                to_unit_id   RAW(16),
                link_type    VARCHAR2(50),
                entity_id    RAW(16),
                weight       BINARY_FLOAT,
                bank_id      VARCHAR2(255),
                created_at   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
                PRIMARY KEY (from_unit_id, to_unit_id, link_type)
            )
        """)

        cursor.execute("""
            CREATE TABLE async_operations (
                operation_id    VARCHAR2(255) PRIMARY KEY,
                bank_id         VARCHAR2(255),
                operation_type  VARCHAR2(100),
                status          VARCHAR2(50) DEFAULT 'pending',
                result_metadata CLOB CHECK (result_metadata IS JSON),
                task_payload    CLOB CHECK (task_payload IS JSON),
                created_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
                updated_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP
            )
        """)

        conn.commit()
    yield


@pytest.fixture(scope="session")
def backend_dsn(oracle_dsn, test_schema):
    """DSN string for connecting as the test schema user."""
    return f"{test_schema}/testpass@{oracle_dsn}"


@pytest.fixture()
def dialect():
    """Create an OracleDialect instance."""
    from hindsight_api.engine.sql.oracle import OracleDialect

    return OracleDialect()


# ---------------------------------------------------------------------------
# 1. OracleBackend — pool lifecycle and connection wrappers
# ---------------------------------------------------------------------------


class TestOracleBackendLifecycle:
    """Validate OracleBackend.initialize / acquire / shutdown against a real Oracle."""

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, oracle_dsn, test_schema, setup_tables):
        """Backend can create a pool and shut it down cleanly."""
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(
            f"{test_schema}/testpass@{oracle_dsn}",
            min_size=1,
            max_size=2,
        )
        assert backend.get_pool() is not None
        await backend.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_and_select(self, oracle_dsn, test_schema, setup_tables):
        """Can acquire a connection and run a basic SELECT."""
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                rows = await conn.fetch("SELECT 1 AS val FROM DUAL")
                assert len(rows) == 1
                assert rows[0]["val"] == 1
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_transaction_commit(self, oracle_dsn, test_schema, setup_tables):
        """Transaction commits on clean exit."""
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"txn-commit-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                await conn.execute(
                    "INSERT INTO banks (bank_id, name) VALUES (:1, :2)",
                    bank_id,
                    "test",
                )
            # Verify committed
            async with backend.acquire() as conn:
                row = await conn.fetchrow("SELECT bank_id FROM banks WHERE bank_id = :1", bank_id)
                assert row is not None
                assert row["bank_id"] == bank_id
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, oracle_dsn, test_schema, setup_tables):
        """Transaction rolls back on exception."""
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"txn-rollback-{uuid.uuid4().hex[:6]}"
        try:
            with pytest.raises(RuntimeError):
                async with backend.transaction() as conn:
                    await conn.execute(
                        "INSERT INTO banks (bank_id, name) VALUES (:1, :2)",
                        bank_id,
                        "test",
                    )
                    raise RuntimeError("Force rollback")
            # Verify NOT committed
            async with backend.acquire() as conn:
                row = await conn.fetchrow("SELECT bank_id FROM banks WHERE bank_id = :1", bank_id)
                assert row is None
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 2. DatabaseConnection — execute, fetch, fetchrow, fetchval, executemany
# ---------------------------------------------------------------------------


class TestOracleConnectionMethods:
    """Validate each DatabaseConnection method returns correct types."""

    @pytest.mark.asyncio
    async def test_execute_returns_status(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                status = await conn.execute(
                    "INSERT INTO banks (bank_id, name) VALUES (:1, :2)",
                    f"exec-{uuid.uuid4().hex[:6]}",
                    "test",
                )
                assert isinstance(status, str)
                assert "1" in status  # "OK 1"
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_fetch_returns_result_rows(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend
        from hindsight_api.engine.db.result import ResultRow

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                rows = await conn.fetch("SELECT 1 AS a, 2 AS b FROM DUAL")
                assert len(rows) == 1
                assert isinstance(rows[0], ResultRow)
                assert rows[0]["a"] == 1
                assert rows[0]["b"] == 2
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_fetchrow_returns_single_row_or_none(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                row = await conn.fetchrow("SELECT 42 AS val FROM DUAL")
                assert row is not None
                assert row["val"] == 42

                # No match → None
                row = await conn.fetchrow(
                    "SELECT 1 FROM banks WHERE bank_id = :1",
                    "nonexistent-bank-id-xyz",
                )
                assert row is None
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_fetchval_returns_scalar(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                val = await conn.fetchval("SELECT 99 FROM DUAL")
                assert val == 99

                val = await conn.fetchval("SELECT COUNT(*) FROM banks WHERE bank_id = :1", "nope")
                assert val == 0
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_executemany(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        prefix = f"em-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                await conn.executemany(
                    "INSERT INTO banks (bank_id, name) VALUES (:1, :2)",
                    [
                        (f"{prefix}-1", "bank1"),
                        (f"{prefix}-2", "bank2"),
                        (f"{prefix}-3", "bank3"),
                    ],
                )
            async with backend.acquire() as conn:
                val = await conn.fetchval(
                    "SELECT COUNT(*) FROM banks WHERE bank_id LIKE :1",
                    f"{prefix}%",
                )
                assert val == 3
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 3. ResultRow — verify dict-like access on real Oracle rows
# ---------------------------------------------------------------------------


class TestResultRowWithOracle:
    """Verify ResultRow wraps real Oracle cursor results properly."""

    @pytest.mark.asyncio
    async def test_column_access_by_name(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                rows = await conn.fetch("SELECT 'hello' AS greeting, 42 AS answer FROM DUAL")
                row = rows[0]
                assert row["greeting"] == "hello"
                assert row["answer"] == 42
                assert row.greeting == "hello"
                assert row.answer == 42
                assert row.get("greeting") == "hello"
                assert row.get("missing", "default") == "default"
                assert "greeting" in row
                assert "missing" not in row
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 4. Vector operations via dialect
# ---------------------------------------------------------------------------


class TestOracleVectorOps:
    """Validate vector insert + cosine distance search through the backend."""

    @pytest.mark.asyncio
    async def test_vector_insert_and_similarity_search(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"vec-{uuid.uuid4().hex[:6]}"
        try:
            # Insert vectors with different directions
            async with backend.acquire() as conn:
                for i in range(3):
                    emb = [0.0] * 384
                    emb[i] = 1.0
                    raw_conn = conn._conn
                    cursor = raw_conn.cursor()
                    await cursor.execute(
                        """
                        INSERT INTO memory_units (id, bank_id, text, embedding, fact_type)
                        VALUES (:1, :2, :3, :4, :5)
                        """,
                        [uuid.uuid4().bytes, bank_id, f"fact-{i}", to_vector32(emb), "world"],
                    )
                    await raw_conn.commit()

            # Search for vector closest to [1,0,0,...]
            async with backend.acquire() as conn:
                query_vec = [0.0] * 384
                query_vec[0] = 1.0
                raw_conn = conn._conn
                cursor = raw_conn.cursor()
                await cursor.execute(
                    """
                    SELECT text, VECTOR_DISTANCE(embedding, :qvec, COSINE) AS dist
                    FROM memory_units
                    WHERE bank_id = :bank
                    ORDER BY VECTOR_DISTANCE(embedding, :qvec, COSINE)
                    FETCH FIRST 1 ROWS ONLY
                    """,
                    {"qvec": to_vector32(query_vec), "bank": bank_id},
                )
                row = await cursor.fetchone()
                assert row is not None
                assert row[0] == "fact-0"
                assert row[1] < 0.01
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 5. JSON operations via dialect
# ---------------------------------------------------------------------------


class TestOracleJsonOps:
    """Validate JSON insert/extract/merge through the backend."""

    @pytest.mark.asyncio
    async def test_json_insert_and_extract(self, oracle_dsn, test_schema, setup_tables):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"json-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                await conn.execute(
                    "INSERT INTO banks (bank_id, disposition) VALUES (:1, :2)",
                    bank_id,
                    json.dumps({"skepticism": 3, "empathy": 4}),
                )

            async with backend.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT JSON_VALUE(disposition, '$.skepticism' RETURNING NUMBER) AS skepticism
                    FROM banks WHERE bank_id = :1
                    """,
                    bank_id,
                )
                assert row is not None
                assert row["skepticism"] == 3
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_json_merge_patch(self, oracle_dsn, test_schema, setup_tables):
        """JSON_MERGEPATCH — Oracle equivalent of PG's || for JSONB concatenation."""
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"merge-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                await conn.execute(
                    "INSERT INTO banks (bank_id, disposition) VALUES (:1, :2)",
                    bank_id,
                    json.dumps({"skepticism": 3}),
                )
            async with backend.transaction() as conn:
                await conn.execute(
                    """
                    UPDATE banks SET disposition = JSON_MERGEPATCH(disposition, :1)
                    WHERE bank_id = :2
                    """,
                    json.dumps({"empathy": 5}),
                    bank_id,
                )
            async with backend.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT JSON_VALUE(disposition, '$.empathy' RETURNING NUMBER) AS empathy,
                           JSON_VALUE(disposition, '$.skepticism' RETURNING NUMBER) AS skepticism
                    FROM banks WHERE bank_id = :1
                    """,
                    bank_id,
                )
                assert row["empathy"] == 5
                assert row["skepticism"] == 3  # original preserved
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 6. Upsert (MERGE INTO) via dialect
# ---------------------------------------------------------------------------


class TestOracleUpsert:
    """Validate MERGE INTO through OracleDialect.upsert()."""

    @pytest.mark.asyncio
    async def test_upsert_insert_then_update(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"ups-{uuid.uuid4().hex[:6]}"
        try:
            # Use the dialect to generate the MERGE statement
            sql = dialect.upsert(
                "entities",
                ["bank_id", "canonical_name", "mention_count"],
                ["bank_id", "canonical_name"],
                ["mention_count"],
            )

            async with backend.transaction() as conn:
                # First: insert
                await conn.execute(sql, bank_id, "Alice", 1)
            async with backend.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT mention_count FROM entities WHERE bank_id = :1 AND canonical_name = :2",
                    bank_id,
                    "Alice",
                )
                assert row["mention_count"] == 1

            async with backend.transaction() as conn:
                # Second: update
                await conn.execute(sql, bank_id, "Alice", 99)
            async with backend.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT mention_count FROM entities WHERE bank_id = :1 AND canonical_name = :2",
                    bank_id,
                    "Alice",
                )
                assert row["mention_count"] == 99
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 7. ILIKE via dialect
# ---------------------------------------------------------------------------


class TestOracleIlike:
    """Validate case-insensitive matching through OracleDialect.ilike()."""

    @pytest.mark.asyncio
    async def test_case_insensitive_search(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"ilike-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                await conn.execute(
                    "INSERT INTO entities (bank_id, canonical_name, mention_count) VALUES (:1, :2, :3)",
                    bank_id,
                    "Alice Johnson",
                    1,
                )

            ilike_expr = dialect.ilike("canonical_name", ":2")
            async with backend.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT canonical_name FROM entities WHERE bank_id = :1 AND {ilike_expr}",
                    bank_id,
                    "%alice%",
                )
                assert row is not None
                assert row["canonical_name"] == "Alice Johnson"
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 8. Fuzzy matching via dialect
# ---------------------------------------------------------------------------


class TestOracleFuzzyMatching:
    """Validate UTL_MATCH through OracleDialect.similarity()."""

    @pytest.mark.asyncio
    async def test_similarity_ranking(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"fuzz-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                for name in ["Alice Johnson", "Alicia Jonson", "Bob Smith"]:
                    await conn.execute(
                        "INSERT INTO entities (bank_id, canonical_name, mention_count) VALUES (:1, :2, :3)",
                        bank_id,
                        name,
                        1,
                    )

            # Use named params since similarity expression references :search_name multiple times
            sim_expr = dialect.similarity("canonical_name", ":search_name")
            async with backend.acquire() as conn:
                raw_conn = conn._conn
                cursor = raw_conn.cursor()
                await cursor.execute(
                    f"""
                    SELECT canonical_name, {sim_expr} AS sim
                    FROM entities
                    WHERE bank_id = :bank_id AND {sim_expr} > 0.5
                    ORDER BY {sim_expr} DESC
                    """,
                    {"search_name": "Alice Jonson", "bank_id": bank_id},
                )
                rows_raw = await cursor.fetchall()
                cursor.close()
                names = [r[0] for r in rows_raw]
                # Should match Alice/Alicia but not Bob
                assert any("Alice" in n or "Alicia" in n for n in names)
                assert not any("Bob" in n for n in names)
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 9. FOR UPDATE SKIP LOCKED via dialect
# ---------------------------------------------------------------------------


class TestOracleLocking:
    """Validate row-level locking through the dialect."""

    @pytest.mark.asyncio
    async def test_for_update_skip_locked(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=4)
        op_id = f"lock-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                await conn.execute(
                    "INSERT INTO async_operations (operation_id, bank_id, operation_type, status) "
                    "VALUES (:1, :2, :3, :4)",
                    op_id,
                    "lock-bank",
                    "retain",
                    "pending",
                )

            fuskl = dialect.for_update_skip_locked()
            async with backend.transaction() as conn:
                # Lock the row
                row = await conn.fetchrow(
                    f"SELECT operation_id FROM async_operations WHERE operation_id = :1 {fuskl}",
                    op_id,
                )
                assert row is not None
                assert row["operation_id"] == op_id
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 10. Pagination via dialect
# ---------------------------------------------------------------------------


class TestOraclePagination:
    """Validate FETCH FIRST N ROWS ONLY / OFFSET through the dialect."""

    @pytest.mark.asyncio
    async def test_limit_offset(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        bank_id = f"page-{uuid.uuid4().hex[:6]}"
        try:
            async with backend.transaction() as conn:
                for i in range(10):
                    await conn.execute(
                        "INSERT INTO memory_units (id, bank_id, text, fact_type) VALUES (:1, :2, :3, :4)",
                        uuid.uuid4().bytes,
                        bank_id,
                        f"fact-{i:02d}",
                        "world",
                    )

            limit_clause = dialect.limit_offset(":lim", ":off")
            async with backend.acquire() as conn:
                raw_conn = conn._conn
                cursor = raw_conn.cursor()
                await cursor.execute(
                    f"""
                    SELECT text FROM memory_units
                    WHERE bank_id = :bank_id
                    ORDER BY TO_CHAR(text)
                    {limit_clause}
                    """,
                    {"bank_id": bank_id, "lim": 3, "off": 2},
                )
                columns = [col[0].lower() for col in cursor.description or []]
                raw_rows = await cursor.fetchall()
                cursor.close()
                rows = [dict(zip(columns, r)) for r in raw_rows]
                assert len(rows) == 3
                # Should be fact-02, fact-03, fact-04 (offset 2 from sorted list)
                assert rows[0]["text"] == "fact-02"
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 11. Concurrent connections — matches PG pool behavior
# ---------------------------------------------------------------------------


class TestOracleConcurrency:
    """Validate that the pool supports concurrent async operations."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, oracle_dsn, test_schema, setup_tables):
        import asyncio

        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=2, max_size=4)
        try:

            async def query(n):
                async with backend.acquire() as conn:
                    val = await conn.fetchval(f"SELECT {n} FROM DUAL")
                    return val

            results = await asyncio.gather(*[query(i) for i in range(10)])
            assert sorted(results) == list(range(10))
        finally:
            await backend.shutdown()


# ---------------------------------------------------------------------------
# 12. SQLDialect string generation sanity (no DB needed, but validates
#     that generated SQL is accepted by Oracle when executed)
# ---------------------------------------------------------------------------


class TestOracleDialectSqlAccepted:
    """Run dialect-generated SQL fragments against real Oracle to verify syntax."""

    @pytest.mark.asyncio
    async def test_generate_uuid_accepted(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                val = await conn.fetchval(f"SELECT {dialect.generate_uuid()} FROM DUAL")
                assert val is not None
                # The abstraction normalizes RAW(16) to Python uuid.UUID
                import uuid as _uuid

                assert isinstance(val, _uuid.UUID), f"Expected uuid.UUID, got {type(val).__name__}"
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_current_timestamp_accepted(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                val = await conn.fetchval(f"SELECT {dialect.current_timestamp()} FROM DUAL")
                assert val is not None
        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_greatest_accepted(self, oracle_dsn, test_schema, setup_tables, dialect):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        await backend.initialize(f"{test_schema}/testpass@{oracle_dsn}", min_size=1, max_size=2)
        try:
            async with backend.acquire() as conn:
                val = await conn.fetchval(f"SELECT {dialect.greatest('3', '7', '1')} FROM DUAL")
                assert val == 7
        finally:
            await backend.shutdown()
