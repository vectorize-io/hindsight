import uuid

import asyncpg
import pytest
from typer.testing import CliRunner

from hindsight_api._vector_index import SCANN_MIN_ROWS_FOR_AUTO_INDEX
from hindsight_api.admin import cli as admin_cli
from hindsight_api.admin import reembed as reembed_module
from hindsight_api.admin.reembed import ReembedOptions, abandon_reembed, reembed_schema
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.retain.embedding_processing import augment_texts_with_dates, format_readable_date
from hindsight_api.engine.retain.types import ExtractedFact
from hindsight_api.migrations import run_migrations


class FakeEmbeddings:
    def __init__(self, dimension: int):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def initialize(self) -> None:
        return None

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] * self._dimension for index, _text in enumerate(texts)]

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        return self.encode_documents(texts)


class ShadowIndexConn:
    def __init__(
        self,
        *,
        extensions: set[str],
        memory_units_count: int,
        mental_models_count: int,
        banks: list[dict] | None = None,
    ):
        self.extensions = extensions
        self.counts = {
            "memory_units": memory_units_count,
            "mental_models": mental_models_count,
        }
        self.banks = banks or []
        self.statements: list[tuple[str, tuple]] = []

    async def fetchval(self, query, *args):
        if "FROM pg_extension" in query:
            return 1 if args[0] in self.extensions else None
        raise AssertionError(f"Unexpected fetchval query: {query}")

    async def fetchrow(self, query, *args):
        if "COUNT(*)" in query and "embedding_reembed" in query:
            return self.counts
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def fetch(self, query, *args):
        if "FROM pg_extension" in query:
            return [{"extname": extension} for extension in self.extensions]
        if "banks" in query and "ORDER BY bank_id" in query:
            return self.banks
        raise AssertionError(f"Unexpected fetch query: {query}")

    async def execute(self, query, *args):
        self.statements.append((query, args))
        return "UPDATE 1"


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class SemanticStageConn:
    def __init__(self):
        self.statements: list[tuple[str, tuple]] = []

    async def fetch(self, query, *args):
        if "FROM pg_extension" in query:
            return [{"extname": "vchord"}]
        if "GROUP BY bank_id, fact_type" in query:
            return [
                {"bank_id": "bank-a", "fact_type": "world"},
                {"bank_id": "bank-a", "fact_type": "experience"},
            ]
        raise AssertionError(f"Unexpected fetch query: {query}")

    async def execute(self, query, *args):
        self.statements.append((query, args))
        return "UPDATE 1"

    async def fetchval(self, query, *args):
        self.statements.append((query, args))
        if "INSERT INTO" in query and "embedding_reembed_semantic_links" in query:
            return 2
        raise AssertionError(f"Unexpected fetchval query: {query}")

    def transaction(self):
        return _Transaction()


class SchemaDiscoveryConn:
    def __init__(self, schemas: list[str]):
        self.schemas = schemas

    async def fetch(self, query, *args):
        assert "FROM pg_namespace" in query
        return [{"schema_name": schema} for schema in self.schemas]


class ActiveMigrationConn:
    async def fetch(self, query, *args):
        assert "LIMIT 1" not in query
        return [
            {"migration_id": uuid.UUID("00000000-0000-0000-0000-000000000001")},
            {"migration_id": uuid.UUID("00000000-0000-0000-0000-000000000002")},
        ]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["reembed-status"], "Specify --schema for reembed-status"),
        (["reembed-abandon", "--yes"], "Specify --schema for reembed-abandon"),
    ],
)
def test_reembed_schema_scoped_commands_require_explicit_schema(args, message):
    result = CliRunner().invoke(admin_cli.app, args)

    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--batch-size", "0"], "batch-size must be at least 1"),
        (["--max-retries", "-1"], "max-retries must be at least 0"),
        (
            ["--index-max-parallel-maintenance-workers", "-1"],
            "index-max-parallel-maintenance-workers must be at least 0",
        ),
    ],
)
def test_reembed_cli_rejects_invalid_batch_options(args, message):
    result = CliRunner().invoke(admin_cli.app, ["reembed", "--schema", "public", *args, "--yes"])

    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size must be at least 1"),
        ({"max_retries": -1}, "max_retries must be at least 0"),
        (
            {"index_max_parallel_maintenance_workers": -1},
            "index_max_parallel_maintenance_workers must be at least 0",
        ),
    ],
)
def test_reembed_options_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReembedOptions(**kwargs)


@pytest.mark.asyncio
async def test_discover_hindsight_schemas_returns_all_schemas_with_base_first():
    conn = SchemaDiscoveryConn(["user_acme", "public", "tenant_beta"])

    schemas = await reembed_module.discover_hindsight_schemas(conn, "public")

    assert schemas == ["public", "user_acme", "tenant_beta"]


@pytest.mark.asyncio
async def test_fetch_active_migration_detects_multiple_active_rows():
    with pytest.raises(RuntimeError, match="multiple active reembed migrations"):
        await reembed_module._fetch_active_migration(ActiveMigrationConn(), "public")


@pytest.mark.asyncio
async def test_resolve_reembed_schemas_requires_explicit_scope():
    with pytest.raises(ValueError, match="Specify --schema or --all-schemas"):
        await admin_cli._resolve_reembed_schemas(
            "postgresql://unused",
            HindsightConfig.from_env(),
            schemas=None,
            all_schemas=False,
        )


@pytest.mark.asyncio
async def test_reembed_schema_rejects_concurrent_runner(pg0_db_url):
    schema = f"reembed_locked_{uuid.uuid4().hex[:8]}"
    lock_key = reembed_module._reembed_lock_key(schema)
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute("SELECT pg_advisory_lock(hashtext($1))", lock_key)

        with pytest.raises(RuntimeError, match=f"Another reembed command is already running for schema '{schema}'"):
            await reembed_schema(
                pg0_db_url,
                schema,
                HindsightConfig.from_env(),
                ReembedOptions(batch_size=10),
            )
    finally:
        await conn.execute("SELECT pg_advisory_unlock(hashtext($1))", lock_key)
        await conn.close()


@pytest.mark.asyncio
async def test_reembed_migrations_allow_only_one_active_row_per_schema(pg0_db_url):
    schema = f"reembed_active_unique_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
    finally:
        await conn.close()

    run_migrations(pg0_db_url, schema=schema)

    conn = await asyncpg.connect(pg0_db_url)
    try:
        index_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = $1
                  AND indexname = 'idx_embedding_reembed_migrations_one_active'
                  AND indexdef ILIKE '%WHERE (status = ANY%'
            )
            """,
            schema,
        )
        migration_id = uuid.uuid4()
        await conn.execute(
            f"""
            INSERT INTO {schema}.embedding_reembed_migrations
            (migration_id, schema_name, provider, model, dimension, vector_extension,
             config_fingerprint, model_identity, status)
            VALUES ($1, $2, 'local', 'test-model', 384, 'pgvector', 'fingerprint-a', '{{}}'::jsonb, 'running')
            """,
            migration_id,
            schema,
        )

        with pytest.raises(asyncpg.exceptions.UniqueViolationError):
            await conn.execute(
                f"""
                INSERT INTO {schema}.embedding_reembed_migrations
                (migration_id, schema_name, provider, model, dimension, vector_extension,
                 config_fingerprint, model_identity, status)
                VALUES ($1, $2, 'local', 'test-model', 384, 'pgvector', 'fingerprint-b', '{{}}'::jsonb, 'failed')
                """,
                uuid.uuid4(),
                schema,
            )

        await conn.execute(
            f"""
            INSERT INTO {schema}.embedding_reembed_migrations
            (migration_id, schema_name, provider, model, dimension, vector_extension,
             config_fingerprint, model_identity, status)
            VALUES ($1, $2, 'local', 'test-model', 384, 'pgvector', 'fingerprint-c', '{{}}'::jsonb, 'completed')
            """,
            uuid.uuid4(),
            schema,
        )

        assert index_exists is True
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_create_shadow_indexes_defers_scann_when_one_indexed_table_is_too_small():
    conn = ShadowIndexConn(
        extensions={"alloydb_scann"},
        memory_units_count=SCANN_MIN_ROWS_FOR_AUTO_INDEX,
        mental_models_count=0,
    )

    message = await reembed_module._create_shadow_indexes(conn, "public", "scann", uuid.uuid4())

    assert "deferred: scann" in message
    assert "mental_models=0" in message
    assert any("shadow_indexes_state = 'deferred'" in statement for statement, _args in conn.statements)
    assert not any("CREATE INDEX" in statement for statement, _args in conn.statements)


@pytest.mark.asyncio
async def test_create_shadow_indexes_uses_resolved_azure_diskann_clause():
    conn = ShadowIndexConn(
        extensions={"vector", "pg_diskann"},
        memory_units_count=1,
        mental_models_count=1,
        banks=[{"bank_id": "bank-a", "internal_id": uuid.UUID("00000000-0000-0000-0000-000000000001")}],
    )

    message = await reembed_module._create_shadow_indexes(conn, "public", "pg_diskann", uuid.uuid4())

    create_index_sql = "\n".join(statement for statement, _args in conn.statements if "CREATE INDEX" in statement)
    assert message == "ready"
    assert "max_neighbors = 50" in create_index_sql
    assert "num_neighbors = 50" not in create_index_sql
    assert create_index_sql.count('ON "public"."memory_units"') == 3
    assert "idx_mu_reemb_worl_0000000000000000" in create_index_sql
    assert "idx_mu_reemb_expr_0000000000000000" in create_index_sql
    assert "idx_mu_reemb_obsv_0000000000000000" in create_index_sql


@pytest.mark.asyncio
async def test_create_shadow_indexes_overrides_parallel_maintenance_workers_only_when_requested():
    conn = ShadowIndexConn(
        extensions={"vector"},
        memory_units_count=1,
        mental_models_count=1,
    )

    await reembed_module._create_shadow_indexes(conn, "public", "pgvector", uuid.uuid4())

    assert not any("max_parallel_maintenance_workers" in statement for statement, _args in conn.statements)

    conn = ShadowIndexConn(
        extensions={"vector"},
        memory_units_count=1,
        mental_models_count=1,
    )

    await reembed_module._create_shadow_indexes(
        conn,
        "public",
        "pgvector",
        uuid.uuid4(),
        index_max_parallel_maintenance_workers=0,
    )

    assert conn.statements[0] == (
        "SELECT set_config('max_parallel_maintenance_workers', $1, false)",
        ("0",),
    )
    assert conn.statements[-2] == ("RESET max_parallel_maintenance_workers", ())
    assert any("shadow_indexes_state = 'ready'" in statement for statement, _args in conn.statements)


@pytest.mark.asyncio
async def test_stage_semantic_links_batches_by_bank_and_fact_type_with_ann_tuning():
    conn = SemanticStageConn()
    migration_id = uuid.uuid4()

    staged = await reembed_module._stage_semantic_links(conn, "public", migration_id, "pgvector")

    insert_statements = [
        (query, args)
        for query, args in conn.statements
        if "INSERT INTO" in query and "embedding_reembed_semantic_links" in query
    ]
    assert staged == 4
    assert len(insert_statements) == 2
    assert any("SET LOCAL hnsw.ef_search = 60" in query for query, _args in conn.statements)
    assert all("mu.bank_id = $2" in query for query, _args in insert_statements)
    assert all("mu.fact_type = $3" in query for query, _args in insert_statements)
    assert all("mu.bank_id = seed.bank_id" not in query for query, _args in insert_statements)
    assert [args[1:] for _query, args in insert_statements] == [
        ("bank-a", "world"),
        ("bank-a", "experience"),
    ]


def test_reembed_memory_unit_embedding_text_matches_retain_fact_helper():
    from datetime import UTC, datetime

    occurred_start = datetime(2024, 6, 1, tzinfo=UTC)
    occurred_end = datetime(2024, 6, 3, tzinfo=UTC)
    mentioned_at = datetime(2024, 6, 5, tzinfo=UTC)
    entities = ["pedagogy:scaffolding", "user"]
    fact = ExtractedFact(
        fact_text="User attended workshop",
        fact_type="world",
        entities=entities,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
    )
    row = {
        "text": fact.fact_text,
        "fact_type": fact.fact_type,
        "occurred_start": occurred_start,
        "occurred_end": occurred_end,
        "mentioned_at": mentioned_at,
        "entities": entities,
        "source_count": 0,
    }

    retain_text = augment_texts_with_dates([fact], format_readable_date)[0]
    reembed_text = reembed_module._memory_unit_embedding_text(row)

    assert reembed_text == retain_text


def test_reembed_consolidation_observation_embedding_text_stays_bare():
    row = {
        "text": "Alice prefers handwritten notes",
        "fact_type": "observation",
        "occurred_start": None,
        "occurred_end": None,
        "mentioned_at": None,
        "entities": ["Alice"],
        "source_count": 2,
    }

    assert reembed_module._memory_unit_embedding_text(row) == "Alice prefers handwritten notes"


def test_model_identity_uses_active_provider_whitelist(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL", "identity-test-model")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_BATCH_SIZE", "17")
    reembed_module.clear_config_cache()
    config = HindsightConfig.from_env()

    identity = reembed_module._model_identity(config, dimension=384)

    assert identity["provider"] == config.embeddings_provider
    assert identity["model"] == "identity-test-model"
    assert identity["embeddings_local_model"] == "identity-test-model"
    assert "embeddings_openai_batch_size" not in identity
    assert not any(key.endswith("_api_key") or "service_account_key" in key for key in identity)
    reembed_module.clear_config_cache()


@pytest.mark.asyncio
async def test_reembed_schema_recomputes_embeddings_and_cuts_over(pg0_db_url, embeddings, monkeypatch):
    await embeddings.initialize()
    monkeypatch.setattr(reembed_module, "create_embeddings_from_env", lambda: embeddings)

    schema = f"reembed_test_{uuid.uuid4().hex[:8]}"
    bank_id = f"bank-{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
    finally:
        await conn.close()

    run_migrations(pg0_db_url, schema=schema)

    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"INSERT INTO {schema}.banks (bank_id) VALUES ($1)", bank_id)
        embedding = embeddings.encode(["old embedding seed"])[0]
        embedding_str = "[" + ",".join(str(value) for value in embedding) + "]"
        for _ in range(2):
            await conn.execute(
                f"""
                INSERT INTO {schema}.memory_units (bank_id, text, fact_type, embedding, event_date)
                VALUES ($1, 'Alice likes chess', 'world', $2::vector, now())
                """,
                bank_id,
                embedding_str,
            )
    finally:
        await conn.close()

    progress_events = []
    report = await reembed_schema(
        pg0_db_url,
        schema,
        HindsightConfig.from_env(),
        ReembedOptions(batch_size=1),
        progress_events.append,
    )

    assert report.status == "completed"
    assert report.memory_units_total == 2
    assert report.memory_units_done == 2
    assert report.mental_models_total == 0
    assert report.semantic_links_staged == 2
    assert any(event.phase == "embedding" and event.memory_units_done == 1 for event in progress_events)
    assert any(event.phase == "embedding" and event.memory_units_done == 2 for event in progress_events)
    assert any(event.phase == "building-shadow-indexes" for event in progress_events)
    assert any(event.phase == "staging-semantic-links" and event.group_total == 1 for event in progress_events)
    assert any(event.phase == "cutover" for event in progress_events)
    assert any(event.phase == "completed" for event in progress_events)

    conn = await asyncpg.connect(pg0_db_url)
    try:
        shadow_column_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = $1
                  AND table_name = 'memory_units'
                  AND column_name = 'embedding_reembed'
            )
            """,
            schema,
        )
        assert shadow_column_exists is False

        link_rows = await conn.fetch(
            f"""
            SELECT from_unit_id, to_unit_id, link_type, entity_id, bank_id
            FROM {schema}.memory_links
            WHERE link_type = 'semantic'
            """
        )
        assert len(link_rows) == 2
        assert all(row["from_unit_id"] != row["to_unit_id"] for row in link_rows)
        assert all(row["entity_id"] is None for row in link_rows)
        assert all(row["bank_id"] == bank_id for row in link_rows)

        staging_rows = await conn.fetchval(f"SELECT COUNT(*) FROM {schema}.embedding_reembed_semantic_links")
        assert staging_rows == 0
        worklist_indexes = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM pg_indexes
            WHERE schemaname = $1
              AND indexname IN ('idx_memory_units_reembed_worklist', 'idx_mental_models_reembed_worklist')
            """,
            schema,
        )
        assert worklist_indexes == 0

        migration_status = await conn.fetchval(
            f"SELECT status FROM {schema}.embedding_reembed_migrations WHERE migration_id = $1",
            uuid.UUID(report.migration_id),
        )
        assert migration_status == "completed"

        with pytest.raises(RuntimeError, match=f"Schema '{schema}' has no active reembed migration"):
            await abandon_reembed(
                pg0_db_url,
                schema,
                orphan_shadow_state=False,
            )
        migration_status_after_abandon_attempt = await conn.fetchval(
            f"SELECT status FROM {schema}.embedding_reembed_migrations WHERE migration_id = $1",
            uuid.UUID(report.migration_id),
        )
        assert migration_status_after_abandon_attempt == "completed"
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_reembed_schema_dry_run_reports_shadow_index_phase_state(pg0_db_url, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector")
    new_embeddings = FakeEmbeddings(dimension=384)
    monkeypatch.setattr(reembed_module, "create_embeddings_from_env", lambda: new_embeddings)

    schema = f"reembed_dry_run_{uuid.uuid4().hex[:8]}"
    migration_id = uuid.uuid4()
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
    finally:
        await conn.close()

    run_migrations(pg0_db_url, schema=schema)

    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(
            f"""
            INSERT INTO {schema}.embedding_reembed_migrations
            (migration_id, schema_name, provider, model, dimension, vector_extension,
             config_fingerprint, model_identity, status)
            VALUES ($1, $2, 'local', 'test-model', 384, 'pgvector', 'fingerprint', '{{}}'::jsonb, 'running')
            """,
            migration_id,
            schema,
        )
    finally:
        await conn.close()

    report = await reembed_schema(
        pg0_db_url,
        schema,
        HindsightConfig.from_env(),
        ReembedOptions(batch_size=10, dry_run=True),
    )

    assert report.status == "dry-run"
    assert report.migration_id == str(migration_id)
    assert report.shadow_indexes_state == "pending"

    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_reembed_schema_can_change_embedding_dimension(pg0_db_url, monkeypatch):
    new_embeddings = FakeEmbeddings(dimension=8)
    monkeypatch.setattr(reembed_module, "create_embeddings_from_env", lambda: new_embeddings)

    schema = f"reembed_dim_{uuid.uuid4().hex[:8]}"
    bank_id = f"bank-{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
    finally:
        await conn.close()

    run_migrations(pg0_db_url, schema=schema)

    old_embedding = "[" + ",".join("0.1" for _ in range(384)) + "]"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"INSERT INTO {schema}.banks (bank_id) VALUES ($1)", bank_id)
        await conn.execute(
            f"""
            INSERT INTO {schema}.memory_units (bank_id, text, fact_type, embedding, event_date)
            VALUES ($1, 'Alice likes chess', 'world', $2::vector, now())
            """,
            bank_id,
            old_embedding,
        )
    finally:
        await conn.close()

    report = await reembed_schema(
        pg0_db_url,
        schema,
        HindsightConfig.from_env(),
        ReembedOptions(batch_size=10),
    )

    assert report.status == "completed"

    conn = await asyncpg.connect(pg0_db_url)
    try:
        memory_units_embedding_type = await conn.fetchval(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = $1
              AND c.relname = 'memory_units'
              AND a.attname = 'embedding'
            """,
            schema,
        )
        mental_models_embedding_type = await conn.fetchval(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = $1
              AND c.relname = 'mental_models'
              AND a.attname = 'embedding'
            """,
            schema,
        )
        migration_dimension = await conn.fetchval(
            f"SELECT dimension FROM {schema}.embedding_reembed_migrations WHERE migration_id = $1",
            uuid.UUID(report.migration_id),
        )

        assert memory_units_embedding_type == "vector(8)"
        assert mental_models_embedding_type == "vector(8)"
        assert migration_dimension == 8
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


@pytest.mark.asyncio
async def test_reembed_schema_rejects_pgvector_dimensions_above_hnsw_limit_before_shadow_state(
    pg0_db_url,
    monkeypatch,
):
    monkeypatch.setenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector")
    new_embeddings = FakeEmbeddings(dimension=2001)
    monkeypatch.setattr(reembed_module, "create_embeddings_from_env", lambda: new_embeddings)

    schema = f"reembed_pgvector_dim_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
    finally:
        await conn.close()

    run_migrations(pg0_db_url, schema=schema)

    with pytest.raises(RuntimeError, match="exceeds pgvector HNSW index limit"):
        await reembed_schema(
            pg0_db_url,
            schema,
            HindsightConfig.from_env(),
            ReembedOptions(batch_size=10),
        )

    conn = await asyncpg.connect(pg0_db_url)
    try:
        shadow_columns = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name IN ('memory_units', 'mental_models')
              AND column_name = 'embedding_reembed'
            """,
            schema,
        )
        migration_rows = await conn.fetchval(f"SELECT COUNT(*) FROM {schema}.embedding_reembed_migrations")

        assert shadow_columns == 0
        assert migration_rows == 0
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()
