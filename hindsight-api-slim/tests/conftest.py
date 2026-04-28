"""
Pytest configuration and shared fixtures.
"""
import asyncio
import os
from pathlib import Path

import filelock
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from hindsight_api import LLMConfig, LocalSTEmbeddings, MemoryEngine, RequestContext
from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.pg0 import EmbeddedPostgres

# Default pg0 instance configuration for tests
DEFAULT_PG0_INSTANCE_NAME = "hindsight-test"
DEFAULT_PG0_PORT = int(os.environ.get("HINDSIGHT_TEST_PG_PORT", "5556"))


# Load environment variables from .env at the start of test session
def pytest_configure(config):
    """Load environment variables before running tests."""
    # Look for .env in the workspace root (two levels up from tests dir)
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        print(f"Warning: {env_file} not found, tests may fail without proper configuration")


@pytest.fixture(scope="session")
def db_url():
    """
    Provide a PostgreSQL connection URL for tests.

    If HINDSIGHT_API_DATABASE_URL is set, use it directly.
    Otherwise, return None to indicate pg0 should be used (managed by pg0_instance fixture).
    """
    return os.getenv("HINDSIGHT_API_DATABASE_URL")


@pytest.fixture(scope="session")
def pg0_db_url(db_url, tmp_path_factory, worker_id):
    """
    Session-scoped fixture that ensures pg0 is running, migrations are applied,
    and returns the database URL.

    If HINDSIGHT_API_DATABASE_URL is a plain postgresql:// URL, uses it directly.
    If HINDSIGHT_API_DATABASE_URL is a pg0:// URL, resolves it to a real URL first.
    Otherwise, starts pg0 once for the entire test session.

    Uses filelock to ensure only one pytest-xdist worker starts pg0.
    Migrations use PostgreSQL advisory locks internally, so they're safe to call
    from multiple workers - only one will actually run migrations.

    Note: We don't stop pg0 at the end because pytest-xdist runs workers in separate
    processes that share the same pg0 instance. pg0 will persist for the next test run.
    """
    from hindsight_api.pg0 import parse_pg0_url as _parse_pg0_url

    # Determine pg0 instance name/port from db_url (if it's a pg0:// URL) or use defaults
    if db_url and not _parse_pg0_url(db_url)[0]:
        # Plain postgresql:// URL - use it directly but still run migrations
        from hindsight_api.migrations import run_migrations
        run_migrations(db_url)
        return db_url

    if db_url:
        _, pg0_name, pg0_port = _parse_pg0_url(db_url)
        pg0_instance_name = pg0_name or DEFAULT_PG0_INSTANCE_NAME
        pg0_instance_port = pg0_port or DEFAULT_PG0_PORT
    else:
        pg0_instance_name = DEFAULT_PG0_INSTANCE_NAME
        pg0_instance_port = DEFAULT_PG0_PORT

    # Get shared temp dir for coordination between xdist workers
    if worker_id == "master":
        # Running without xdist (-n 0 or no -n flag)
        root_tmp_dir = tmp_path_factory.getbasetemp()
    else:
        # Running with xdist - use parent dir shared by all workers
        root_tmp_dir = tmp_path_factory.getbasetemp().parent

    # Use a lock file to ensure only one worker starts pg0
    lock_file = root_tmp_dir / f"pg0_setup_{pg0_instance_name}.lock"
    url_file = root_tmp_dir / f"pg0_url_{pg0_instance_name}.txt"

    with filelock.FileLock(str(lock_file)):
        if url_file.exists():
            # Another worker already started pg0
            url = url_file.read_text().strip()
        else:
            # First worker - start pg0
            pg0 = EmbeddedPostgres(name=pg0_instance_name, port=pg0_instance_port)

            # Run ensure_running in a new event loop
            loop = asyncio.new_event_loop()
            try:
                url = loop.run_until_complete(pg0.ensure_running())
            finally:
                loop.close()

            # Save URL for other workers
            url_file.write_text(url)

    # Run migrations - uses PostgreSQL advisory lock internally,
    # so safe to call from multiple workers (only one will actually run migrations)
    from hindsight_api.migrations import run_migrations
    run_migrations(url)

    # Clean up stale test data from previous sessions. Per-bank vector indexes
    # accumulate across runs (each test bank creates 3 HNSW indexes) and
    # eventually exhaust pg0's shared memory / max_locks_per_transaction.
    # Only one xdist worker needs to do this.
    cleanup_lock = root_tmp_dir / f"pg0_cleanup_{pg0_instance_name}.lock"
    cleanup_done = root_tmp_dir / f"pg0_cleanup_{pg0_instance_name}.done"
    with filelock.FileLock(str(cleanup_lock)):
        if not cleanup_done.exists():
            _cleanup_stale_test_data(url)
            cleanup_done.write_text("done")

    return url


def _cleanup_stale_test_data(db_url: str) -> None:
    """Drop all per-bank vector indexes and test data from previous sessions.

    pg0 persists between test runs, so per-bank HNSW indexes accumulate
    (3 per bank × thousands of test banks = tens of thousands of indexes).
    This eventually causes 'out of shared memory' errors because PostgreSQL
    tracks all indexes in shared lock tables.
    """
    import asyncpg

    async def _do_cleanup():
        conn = await asyncpg.connect(db_url)
        try:
            idx_rows = await conn.fetch(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'public' AND indexname LIKE 'idx_mu_emb_%'"
            )
            if idx_rows:
                for row in idx_rows:
                    await conn.execute(f'DROP INDEX IF EXISTS public."{row["indexname"]}"')

            # Truncate test data in dependency order
            for table in [
                "entity_cooccurrences", "unit_entities", "memory_links",
                "entities", "memory_units", "chunks", "documents",
                "mental_models", "directives", "async_operations",
                "audit_log", "webhooks", "file_storage", "banks",
            ]:
                try:
                    await conn.execute(f"TRUNCATE {table} CASCADE")
                except Exception:
                    pass  # Table may not exist yet
        finally:
            await conn.close()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_do_cleanup())
    finally:
        loop.close()


@pytest.fixture(scope="session")
def _oracle_admin_dsn():
    """
    Parse ORACLE_TEST_DSN into admin connection parameters.

    Accepts either URL format (oracle://user:pass@host:port/service) or
    bare DSN (host:port/service) with separate ORACLE_TEST_USER/PASSWORD env vars.
    Skips the entire test session if ORACLE_TEST_DSN is not set.
    """
    from urllib.parse import urlparse

    dsn = os.getenv("ORACLE_TEST_DSN")
    if not dsn:
        pytest.skip("ORACLE_TEST_DSN not set — skipping Oracle tests")

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
    else:
        return {
            "user": os.getenv("ORACLE_TEST_USER", "SYSTEM"),
            "password": os.getenv("ORACLE_TEST_PASSWORD", "oracle"),
            "dsn": dsn,
        }


@pytest.fixture(scope="session")
def oracle_db_url(_oracle_admin_dsn):
    """
    Bootstrap a dedicated Oracle test user with an ASSM tablespace and return
    a connection URL for that user.

    Oracle 23ai requires VECTOR columns to be in an Automatic Segment Space
    Management (ASSM) tablespace. The default SYSTEM tablespace is not ASSM,
    so connecting as SYSTEM directly would cause ORA-43853 during migrations.

    This fixture creates a ``HINDSIGHT_TEST`` user (idempotent) with the USERS
    tablespace (which is ASSM on Oracle Free/XE) and returns a URL that the
    ``oracle_memory`` fixture and ``run_oracle_migrations()`` can use directly.
    """
    try:
        import oracledb
    except ImportError:
        pytest.skip("oracledb not installed — skipping Oracle tests")

    oracledb.defaults.fetch_lobs = False

    admin_user = _oracle_admin_dsn["user"]
    admin_pass = _oracle_admin_dsn["password"]
    bare_dsn = _oracle_admin_dsn["dsn"]

    test_user = "HINDSIGHT_TEST"
    test_pass = "hindsight_test"

    conn = oracledb.connect(user=admin_user, password=admin_pass, dsn=bare_dsn)
    cursor = conn.cursor()
    try:
        # Create test user (idempotent — skip if already exists)
        try:
            cursor.execute(
                f'CREATE USER {test_user} IDENTIFIED BY "{test_pass}" '
                f"DEFAULT TABLESPACE USERS QUOTA UNLIMITED ON USERS"
            )
        except oracledb.DatabaseError as e:
            if hasattr(e.args[0], "code") and e.args[0].code == 1920:
                # ORA-01920: user name conflicts with another user or role name
                pass
            else:
                raise

        # Grant required privileges (idempotent)
        for grant in [
            f"GRANT CONNECT, RESOURCE, UNLIMITED TABLESPACE TO {test_user}",
            f"GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW TO {test_user}",
            f"GRANT CTXAPP TO {test_user}",
        ]:
            try:
                cursor.execute(grant)
            except oracledb.DatabaseError:
                pass

        # Grant UTL_MATCH for fuzzy entity matching (may not be available)
        try:
            cursor.execute(f"GRANT EXECUTE ON UTL_MATCH TO {test_user}")
        except oracledb.DatabaseError:
            pass

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    # Return URL-format DSN for the test user
    url = f"oracle://{test_user}:{test_pass}@{bare_dsn}"

    # Run idempotent migrations once at session scope (mirrors PG's pg0_db_url).
    # This avoids re-running DDL checks on every function-scoped test.
    from hindsight_api.migrations_oracle import run_oracle_migrations

    run_oracle_migrations(url)

    return url


@pytest_asyncio.fixture(scope="function")
async def oracle_memory(oracle_db_url, embeddings, cross_encoder, query_analyzer):
    """
    Provide a MemoryEngine backed by Oracle 23ai for each test.

    Mirrors the PG `memory` fixture but uses the Oracle backend.
    Migrations are run once at session scope in the `oracle_db_url` fixture.
    """
    from hindsight_api.config import clear_config_cache

    # Temporarily set the database backend env var so the global config
    # (used by fq_table / _is_oracle) returns "oracle".
    old_backend = os.environ.get("HINDSIGHT_API_DATABASE_BACKEND")
    os.environ["HINDSIGHT_API_DATABASE_BACKEND"] = "oracle"
    clear_config_cache()

    try:
        mem = MemoryEngine(
            db_url=oracle_db_url,
            # Note: config.py loads ../.env with override=True, so these defaults
            # only apply if no .env file is found. The .env file is authoritative.
            memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "openai"),
            memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
            memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini"),
            memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
            embeddings=embeddings,
            cross_encoder=cross_encoder,
            query_analyzer=query_analyzer,
            pool_min_size=1,
            pool_max_size=5,
            run_migrations=False,  # Already ran above
            task_backend=SyncTaskBackend(),
        )
        await mem.initialize()
        yield mem
        try:
            await mem.close()
        except Exception:
            pass
    finally:
        # Restore original env var and clear config cache
        if old_backend is None:
            os.environ.pop("HINDSIGHT_API_DATABASE_BACKEND", None)
        else:
            os.environ["HINDSIGHT_API_DATABASE_BACKEND"] = old_backend
        clear_config_cache()


@pytest.fixture(scope="function")
def request_context():
    """Provide a default RequestContext for tests."""
    return RequestContext()


@pytest.fixture(scope="session")
def llm_config():
    """
    Provide LLM configuration for tests.
    This can be used by tests that need to call LLM directly without memory system.
    """
    return LLMConfig.from_env()


@pytest.fixture(scope="session")
def embeddings(tmp_path_factory, worker_id):
    """
    Session-scoped embeddings fixture with filelock to prevent race conditions.

    When pytest-xdist runs multiple workers in parallel, they all try to load
    models from the HuggingFace cache simultaneously, which can cause race
    conditions and meta tensor errors. We use a filelock to serialize model
    initialization across workers.
    """
    # Get shared temp dir for coordination between xdist workers
    if worker_id == "master":
        root_tmp_dir = tmp_path_factory.getbasetemp()
    else:
        root_tmp_dir = tmp_path_factory.getbasetemp().parent

    lock_file = root_tmp_dir / "embeddings_init.lock"

    emb = LocalSTEmbeddings()

    # Serialize model initialization across workers
    with filelock.FileLock(str(lock_file)):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(emb.initialize())
        finally:
            loop.close()

    return emb


@pytest.fixture(scope="session")
def cross_encoder(tmp_path_factory, worker_id):
    """
    Session-scoped cross-encoder fixture with filelock to prevent race conditions.

    When pytest-xdist runs multiple workers in parallel, they all try to load
    models from the HuggingFace cache simultaneously, which can cause race
    conditions and meta tensor errors. We use a filelock to serialize model
    initialization across workers.
    """
    # Get shared temp dir for coordination between xdist workers
    if worker_id == "master":
        root_tmp_dir = tmp_path_factory.getbasetemp()
    else:
        root_tmp_dir = tmp_path_factory.getbasetemp().parent

    lock_file = root_tmp_dir / "cross_encoder_init.lock"

    ce = LocalSTCrossEncoder()

    # Serialize model initialization across workers
    with filelock.FileLock(str(lock_file)):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(ce.initialize())
        finally:
            loop.close()

    return ce

@pytest.fixture(scope="session")
def query_analyzer():
    return DateparserQueryAnalyzer()




@pytest_asyncio.fixture(scope="function")
async def memory(pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """
    Provide a MemoryEngine instance for each test.

    Must be function-scoped because:
    1. pytest-xdist runs tests in separate processes with different event loops
    2. asyncpg pools are bound to the event loop that created them
    3. Each test needs its own pool in its own event loop

    Uses small pool sizes since tests run in parallel.
    Uses pg0_db_url (a postgresql:// URL) directly, so MemoryEngine won't try to
    manage pg0 lifecycle - that's handled by the session-scoped pg0_db_url fixture.
    Migrations are disabled here since they're run once at session scope in pg0_db_url.
    Uses SyncTaskBackend so async tasks execute immediately (no worker needed).
    """
    mem = MemoryEngine(
        db_url=pg0_db_url,  # Direct postgresql:// URL, not pg0://
        memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
        memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
        memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,  # Migrations already run at session scope
        task_backend=SyncTaskBackend(),  # Execute tasks immediately in tests
    )
    await mem.initialize()
    yield mem
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass


@pytest_asyncio.fixture(scope="function")
async def memory_no_llm_verify(pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """
    Provide a MemoryEngine instance that skips LLM connection verification.

    This fixture is useful for tests that override the LLM configuration
    after initialization (e.g., to test specific providers).
    """
    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",  # Use mock provider as placeholder
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=True,  # Skip verification - will be overridden by test
    )
    await mem.initialize()
    yield mem
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass
