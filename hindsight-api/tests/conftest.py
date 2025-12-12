"""
Pytest configuration and shared fixtures.
"""
import pytest
import pytest_asyncio
import asyncio
import os
import filelock
from pathlib import Path
from dotenv import load_dotenv
from hindsight_api import MemoryEngine, LLMConfig, LocalSTEmbeddings

from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
from hindsight_api.pg0 import EmbeddedPostgres

# Default pg0 instance configuration for tests
DEFAULT_PG0_INSTANCE_NAME = "hindsight-test"
DEFAULT_PG0_PORT = 5556


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

    If HINDSIGHT_API_DATABASE_URL is set, uses that directly (no pg0 management).
    Otherwise, starts pg0 once for the entire test session.

    Uses filelock to ensure only one pytest-xdist worker starts pg0.
    Migrations use PostgreSQL advisory locks internally, so they're safe to call
    from multiple workers - only one will actually run migrations.

    Note: We don't stop pg0 at the end because pytest-xdist runs workers in separate
    processes that share the same pg0 instance. pg0 will persist for the next test run.
    """
    if db_url:
        # Use provided database URL directly
        return db_url

    # Get shared temp dir for coordination between xdist workers
    if worker_id == "master":
        # Running without xdist (-n 0 or no -n flag)
        root_tmp_dir = tmp_path_factory.getbasetemp()
    else:
        # Running with xdist - use parent dir shared by all workers
        root_tmp_dir = tmp_path_factory.getbasetemp().parent

    # Use a lock file to ensure only one worker starts pg0
    lock_file = root_tmp_dir / "pg0_setup.lock"
    url_file = root_tmp_dir / "pg0_url.txt"

    with filelock.FileLock(str(lock_file)):
        if url_file.exists():
            # Another worker already started pg0
            url = url_file.read_text().strip()
        else:
            # First worker - start pg0
            pg0 = EmbeddedPostgres(name=DEFAULT_PG0_INSTANCE_NAME, port=DEFAULT_PG0_PORT)

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

    return url


@pytest.fixture(scope="session")
def llm_config():
    """
    Provide LLM configuration for tests.
    This can be used by tests that need to call LLM directly without memory system.
    """
    return LLMConfig.for_memory()


@pytest.fixture(scope="session")
def embeddings():

    return LocalSTEmbeddings()



@pytest.fixture(scope="session")
def cross_encoder():

    return LocalSTCrossEncoder()

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
    )
    await mem.initialize()
    yield mem
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass
