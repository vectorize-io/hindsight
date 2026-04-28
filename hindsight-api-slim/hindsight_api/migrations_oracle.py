"""
Oracle 23ai database migrations.

Uses idempotent DDL (CREATE TABLE IF NOT EXISTS) so migrations can safely
run multiple times. Oracle 23ai natively supports IF NOT EXISTS for DDL.

Tables mirror the PostgreSQL schema defined in alembic/versions/ but use
Oracle-native types:
  - UUID        → RAW(16)  with DEFAULT SYS_GUID()
  - TEXT/VARCHAR → VARCHAR2 / CLOB
  - JSONB       → CLOB     (with IS JSON CHECK)
  - BOOLEAN     → NUMBER(1)
  - FLOAT       → BINARY_DOUBLE
  - TIMESTAMP WITH TIME ZONE → TIMESTAMP WITH TIME ZONE
  - VARCHAR[]   → CLOB     (JSON array stored as string)
  - BYTEA       → BLOB
  - vector(384) → VECTOR(384, FLOAT32) (Oracle 23ai native)
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL statements — executed in dependency order
# ---------------------------------------------------------------------------

_DDL_TABLES = [
    # -----------------------------------------------------------------------
    # 1. BANKS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS banks (
        bank_id           VARCHAR2(256)  NOT NULL,
        internal_id       RAW(16)        DEFAULT SYS_GUID() NOT NULL,
        name              VARCHAR2(512),
        disposition       CLOB           DEFAULT '{"skepticism":3,"literalism":3,"empathy":3}' NOT NULL
                                         CONSTRAINT banks_disposition_json CHECK (disposition IS JSON),
        mission           CLOB,
        personality       CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT banks_personality_json CHECK (personality IS JSON),
        config            CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT banks_config_json CHECK (config IS JSON),
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_banks PRIMARY KEY (bank_id),
        CONSTRAINT banks_internal_id_unique UNIQUE (internal_id)
    )
    """,
    # -----------------------------------------------------------------------
    # 2. DOCUMENTS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS documents (
        id                VARCHAR2(512)  NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        original_text     CLOB,
        content_hash      VARCHAR2(128),
        metadata          CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT docs_metadata_json CHECK (metadata IS JSON),
        retain_params     CLOB           CONSTRAINT docs_retain_params_json CHECK (retain_params IS JSON OR retain_params IS NULL),
        file_storage_key  VARCHAR2(512),
        file_original_name VARCHAR2(512),
        file_content_type VARCHAR2(256),
        tags              CLOB           DEFAULT '[]' NOT NULL,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_documents PRIMARY KEY (id, bank_id),
        CONSTRAINT fk_documents_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
    )
    """,
    # -----------------------------------------------------------------------
    # 3. CHUNKS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id          VARCHAR2(512)  NOT NULL,
        document_id       VARCHAR2(512)  NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        chunk_index       NUMBER(10)     NOT NULL,
        chunk_text        CLOB           NOT NULL,
        content_hash      VARCHAR2(128),
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_chunks PRIMARY KEY (chunk_id),
        CONSTRAINT fk_chunks_document FOREIGN KEY (document_id, bank_id)
            REFERENCES documents(id, bank_id) ON DELETE CASCADE
    )
    """,
    # -----------------------------------------------------------------------
    # 4. MEMORY_UNITS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS memory_units (
        id                RAW(16)        DEFAULT SYS_GUID() NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        document_id       VARCHAR2(512),
        chunk_id          VARCHAR2(512),
        text              CLOB           NOT NULL,
        embedding         VECTOR(384, FLOAT32),
        context           CLOB,
        event_date        TIMESTAMP WITH TIME ZONE NOT NULL,
        occurred_start    TIMESTAMP WITH TIME ZONE,
        occurred_end      TIMESTAMP WITH TIME ZONE,
        mentioned_at      TIMESTAMP WITH TIME ZONE,
        fact_type         VARCHAR2(64)   DEFAULT 'world' NOT NULL,
        confidence_score  BINARY_DOUBLE,
        access_count      NUMBER(10)     DEFAULT 0 NOT NULL,
        consolidated_at   TIMESTAMP WITH TIME ZONE,
        observation_scopes CLOB          CONSTRAINT mu_obs_scopes_json CHECK (observation_scopes IS JSON OR observation_scopes IS NULL),
        tags              CLOB           DEFAULT '[]' NOT NULL,
        metadata          CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT mu_metadata_json CHECK (metadata IS JSON),
        proof_count       NUMBER(10)     DEFAULT 1,
        source_memory_ids CLOB,
        history           CLOB           DEFAULT '[]'
                                         CONSTRAINT mu_history_json CHECK (history IS JSON OR history IS NULL),
        text_signals      CLOB,
        consolidation_failed_at TIMESTAMP WITH TIME ZONE,
        search_vector     CLOB,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_memory_units PRIMARY KEY (id),
        CONSTRAINT fk_mu_document FOREIGN KEY (document_id, bank_id)
            REFERENCES documents(id, bank_id) ON DELETE CASCADE,
        CONSTRAINT fk_mu_chunk FOREIGN KEY (chunk_id)
            REFERENCES chunks(chunk_id) ON DELETE SET NULL,
        CONSTRAINT chk_mu_fact_type CHECK (fact_type IN ('world', 'experience', 'observation')),
        CONSTRAINT chk_mu_confidence CHECK (
            confidence_score IS NULL
            OR (confidence_score >= 0.0 AND confidence_score <= 1.0)
        )
    )
    PARTITION BY LIST (bank_id) AUTOMATIC
    (PARTITION p_default VALUES ('__default__'))
    """,
    # -----------------------------------------------------------------------
    # 5. ENTITIES
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS entities (
        id                RAW(16)        DEFAULT SYS_GUID() NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        canonical_name    VARCHAR2(512)  NOT NULL,
        metadata          CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT ent_metadata_json CHECK (metadata IS JSON),
        first_seen        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        last_seen         TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        mention_count     NUMBER(10)     DEFAULT 1 NOT NULL,
        CONSTRAINT pk_entities PRIMARY KEY (id)
    )
    """,
    # -----------------------------------------------------------------------
    # 6. UNIT_ENTITIES (junction)
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS unit_entities (
        unit_id           RAW(16)        NOT NULL,
        entity_id         RAW(16)        NOT NULL,
        CONSTRAINT pk_unit_entities PRIMARY KEY (unit_id, entity_id),
        CONSTRAINT fk_ue_unit FOREIGN KEY (unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
        CONSTRAINT fk_ue_entity FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
    )
    """,
    # -----------------------------------------------------------------------
    # 7. ENTITY_COOCCURRENCES
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS entity_cooccurrences (
        entity_id_1       RAW(16)        NOT NULL,
        entity_id_2       RAW(16)        NOT NULL,
        cooccurrence_count NUMBER(10)    DEFAULT 1 NOT NULL,
        last_cooccurred   TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_entity_cooccurrences PRIMARY KEY (entity_id_1, entity_id_2),
        CONSTRAINT fk_ec_entity1 FOREIGN KEY (entity_id_1) REFERENCES entities(id) ON DELETE CASCADE,
        CONSTRAINT fk_ec_entity2 FOREIGN KEY (entity_id_2) REFERENCES entities(id) ON DELETE CASCADE
    )
    """,
    # -----------------------------------------------------------------------
    # 8. MEMORY_LINKS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS memory_links (
        from_unit_id      RAW(16)        NOT NULL,
        to_unit_id        RAW(16)        NOT NULL,
        link_type         VARCHAR2(64)   NOT NULL,
        entity_id         RAW(16),
        bank_id           VARCHAR2(256),
        weight            BINARY_DOUBLE  DEFAULT 1.0 NOT NULL,
        source_memory_ids CLOB,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT fk_ml_from FOREIGN KEY (from_unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
        CONSTRAINT fk_ml_to FOREIGN KEY (to_unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
        CONSTRAINT fk_ml_entity FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
        CONSTRAINT chk_ml_link_type CHECK (
            link_type IN ('temporal', 'semantic', 'entity', 'causes', 'caused_by', 'enables', 'prevents')
        ),
        CONSTRAINT chk_ml_weight CHECK (weight >= 0.0 AND weight <= 1.0)
    )
    """,
    # -----------------------------------------------------------------------
    # 9. MENTAL_MODELS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS mental_models (
        id                VARCHAR2(256)  NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        subtype           VARCHAR2(32)   NOT NULL,
        name              VARCHAR2(256)  NOT NULL,
        description       CLOB           NOT NULL,
        source_query      CLOB,
        content           CLOB,
        embedding         VECTOR(384, FLOAT32),
        entity_id         RAW(16),
        observations      CLOB           DEFAULT '{"observations":[]}' NOT NULL
                                         CONSTRAINT mm_obs_json CHECK (observations IS JSON),
        links             CLOB,
        tags              CLOB           DEFAULT '[]' NOT NULL,
        max_tokens        NUMBER(10)     DEFAULT 2048 NOT NULL,
        "trigger"         CLOB           DEFAULT '{"refresh_after_consolidation":false}' NOT NULL
                                         CONSTRAINT mm_trigger_json CHECK ("trigger" IS JSON),
        structured_content CLOB          CONSTRAINT mm_sc_json CHECK (structured_content IS JSON OR structured_content IS NULL),
        last_refreshed_source_query CLOB,
        reflect_response  CLOB           CONSTRAINT mm_reflect_resp_json CHECK (reflect_response IS JSON OR reflect_response IS NULL),
        history           CLOB           DEFAULT '[]' NOT NULL
                                         CONSTRAINT mm_history_json CHECK (history IS JSON),
        last_refreshed_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        last_updated      TIMESTAMP WITH TIME ZONE,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_mental_models PRIMARY KEY (id, bank_id),
        CONSTRAINT fk_mm_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE,
        CONSTRAINT fk_mm_entity FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE SET NULL,
        CONSTRAINT chk_mm_subtype CHECK (subtype IN ('structural', 'emergent', 'pinned', 'learned'))
    )
    """,
    # -----------------------------------------------------------------------
    # 10. DIRECTIVES
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS directives (
        id                RAW(16)        DEFAULT SYS_GUID() NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        name              VARCHAR2(256)  NOT NULL,
        content           CLOB           NOT NULL,
        priority          NUMBER(10)     DEFAULT 0 NOT NULL,
        is_active         NUMBER(1)      DEFAULT 1 NOT NULL,
        tags              CLOB           DEFAULT '[]' NOT NULL,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_directives PRIMARY KEY (id),
        CONSTRAINT fk_dir_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
    )
    """,
    # -----------------------------------------------------------------------
    # 11. ASYNC_OPERATIONS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS async_operations (
        operation_id      RAW(16)        DEFAULT SYS_GUID() NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        operation_type    VARCHAR2(128)  NOT NULL,
        status            VARCHAR2(32)   DEFAULT 'pending' NOT NULL,
        worker_id         VARCHAR2(256),
        claimed_at        TIMESTAMP WITH TIME ZONE,
        retry_count       NUMBER(10)     DEFAULT 0 NOT NULL,
        next_retry_at     TIMESTAMP WITH TIME ZONE,
        task_payload      CLOB           CONSTRAINT ao_payload_json CHECK (task_payload IS JSON OR task_payload IS NULL),
        result_metadata   CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT ao_result_json CHECK (result_metadata IS JSON),
        error_message     CLOB,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        completed_at      TIMESTAMP WITH TIME ZONE,
        CONSTRAINT pk_async_operations PRIMARY KEY (operation_id),
        CONSTRAINT fk_ao_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE,
        CONSTRAINT chk_ao_status CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
    )
    """,
    # -----------------------------------------------------------------------
    # 11. WEBHOOKS
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS webhooks (
        id                RAW(16)        DEFAULT SYS_GUID() NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        url               VARCHAR2(2048) NOT NULL,
        secret            VARCHAR2(512),
        event_types       CLOB           DEFAULT '[]' NOT NULL,
        http_config       CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT wh_http_config_json CHECK (http_config IS JSON),
        enabled           NUMBER(1)      DEFAULT 1 NOT NULL,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        CONSTRAINT pk_webhooks PRIMARY KEY (id),
        CONSTRAINT fk_wh_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
    )
    """,
    # -----------------------------------------------------------------------
    # 12. FILE_STORAGE
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS file_storage (
        storage_key       VARCHAR2(512)  NOT NULL,
        data              BLOB           NOT NULL,
        CONSTRAINT pk_file_storage PRIMARY KEY (storage_key)
    )
    """,
    # -----------------------------------------------------------------------
    # 13. AUDIT_LOG
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id                RAW(16)        DEFAULT SYS_GUID() NOT NULL,
        action            VARCHAR2(128)  NOT NULL,
        transport         VARCHAR2(64)   NOT NULL,
        bank_id           VARCHAR2(256),
        started_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        ended_at          TIMESTAMP WITH TIME ZONE,
        request           CLOB           CONSTRAINT al_request_json CHECK (request IS JSON OR request IS NULL),
        response          CLOB           CONSTRAINT al_response_json CHECK (response IS JSON OR response IS NULL),
        metadata          CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT al_metadata_json CHECK (metadata IS JSON),
        CONSTRAINT pk_audit_log PRIMARY KEY (id)
    )
    """,
    # -----------------------------------------------------------------------
    # 11. OBSERVATION_SOURCES — junction table replacing source_memory_ids
    # column. Enables standard SQL joins instead of dialect-specific array
    # operators (PG unnest/&&) or JSON_TABLE (Oracle).
    # -----------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS observation_sources (
        observation_id    RAW(16)        NOT NULL,
        source_id         RAW(16)        NOT NULL,
        CONSTRAINT pk_observation_sources PRIMARY KEY (observation_id, source_id),
        CONSTRAINT fk_obs_src_observation FOREIGN KEY (observation_id)
            REFERENCES memory_units(id) ON DELETE CASCADE
    )
    """,
]


# ---------------------------------------------------------------------------
# Indexes — created with IF NOT EXISTS where Oracle 23ai supports it,
# otherwise guarded by PL/SQL exception handler.
# ---------------------------------------------------------------------------


def _idx(name: str, ddl: str) -> str:
    """Wrap CREATE INDEX in a PL/SQL block that silently ignores ORA-00955 (name already used)."""
    return f"""
    BEGIN
        EXECUTE IMMEDIATE '{ddl.strip().replace("'", "''")}';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE = -955 THEN NULL;  -- index already exists
            ELSE RAISE;
            END IF;
    END;
    """


_DDL_INDEXES = [
    # --- documents ---
    _idx("idx_docs_bank_id", "CREATE INDEX idx_docs_bank_id ON documents(bank_id)"),
    _idx("idx_docs_content_hash", "CREATE INDEX idx_docs_content_hash ON documents(content_hash)"),
    # --- chunks ---
    _idx("idx_chunks_document_id", "CREATE INDEX idx_chunks_document_id ON chunks(document_id)"),
    _idx("idx_chunks_bank_id", "CREATE INDEX idx_chunks_bank_id ON chunks(bank_id)"),
    # --- memory_units ---
    _idx("idx_mu_bank_id", "CREATE INDEX idx_mu_bank_id ON memory_units(bank_id)"),
    _idx("idx_mu_document_id", "CREATE INDEX idx_mu_document_id ON memory_units(document_id)"),
    _idx("idx_mu_chunk_id", "CREATE INDEX idx_mu_chunk_id ON memory_units(chunk_id)"),
    _idx("idx_mu_event_date", "CREATE INDEX idx_mu_event_date ON memory_units(event_date DESC)"),
    _idx("idx_mu_bank_date", "CREATE INDEX idx_mu_bank_date ON memory_units(bank_id, event_date DESC)"),
    _idx("idx_mu_access_count", "CREATE INDEX idx_mu_access_count ON memory_units(access_count DESC)"),
    _idx("idx_mu_fact_type", "CREATE INDEX idx_mu_fact_type ON memory_units(fact_type)"),
    _idx("idx_mu_bank_fact_type", "CREATE INDEX idx_mu_bank_fact_type ON memory_units(bank_id, fact_type)"),
    _idx(
        "idx_mu_bank_type_date",
        "CREATE INDEX idx_mu_bank_type_date ON memory_units(bank_id, fact_type, event_date DESC)",
    ),
    # --- entities ---
    _idx("idx_ent_bank_id", "CREATE INDEX idx_ent_bank_id ON entities(bank_id)"),
    _idx("idx_ent_canonical_name", "CREATE INDEX idx_ent_canonical_name ON entities(canonical_name)"),
    _idx("idx_ent_bank_name", "CREATE INDEX idx_ent_bank_name ON entities(bank_id, canonical_name)"),
    _idx(
        "idx_ent_bank_lower_name",
        "CREATE UNIQUE INDEX idx_ent_bank_lower_name ON entities(bank_id, LOWER(canonical_name))",
    ),
    # --- unit_entities ---
    _idx("idx_ue_unit", "CREATE INDEX idx_ue_unit ON unit_entities(unit_id)"),
    _idx("idx_ue_entity", "CREATE INDEX idx_ue_entity ON unit_entities(entity_id)"),
    # --- entity_cooccurrences ---
    _idx("idx_ec_entity1", "CREATE INDEX idx_ec_entity1 ON entity_cooccurrences(entity_id_1)"),
    _idx("idx_ec_entity2", "CREATE INDEX idx_ec_entity2 ON entity_cooccurrences(entity_id_2)"),
    _idx("idx_ec_count", "CREATE INDEX idx_ec_count ON entity_cooccurrences(cooccurrence_count DESC)"),
    # --- memory_links ---
    # Unique constraint matching PG's idx_memory_links_unique — required for ON CONFLICT DO NOTHING
    # duplicate suppression. Oracle function-based unique index uses NVL (Oracle equivalent of COALESCE)
    # with the nil UUID as raw bytes to handle nullable entity_id.
    _idx(
        "idx_memory_links_unique",
        "CREATE UNIQUE INDEX idx_memory_links_unique ON memory_links("
        "from_unit_id, to_unit_id, link_type, "
        "NVL(entity_id, HEXTORAW('00000000000000000000000000000000')))",
    ),
    _idx("idx_ml_from_unit", "CREATE INDEX idx_ml_from_unit ON memory_links(from_unit_id)"),
    _idx("idx_ml_to_unit", "CREATE INDEX idx_ml_to_unit ON memory_links(to_unit_id)"),
    _idx("idx_ml_entity", "CREATE INDEX idx_ml_entity ON memory_links(entity_id)"),
    _idx("idx_ml_link_type", "CREATE INDEX idx_ml_link_type ON memory_links(link_type)"),
    _idx("idx_ml_bank_id", "CREATE INDEX idx_ml_bank_id ON memory_links(bank_id)"),
    # --- directives ---
    _idx("idx_dir_bank_id", "CREATE INDEX idx_dir_bank_id ON directives(bank_id)"),
    _idx("idx_dir_bank_active", "CREATE INDEX idx_dir_bank_active ON directives(bank_id, is_active)"),
    # --- mental_models ---
    _idx("idx_mm_bank_id", "CREATE INDEX idx_mm_bank_id ON mental_models(bank_id)"),
    _idx("idx_mm_subtype", "CREATE INDEX idx_mm_subtype ON mental_models(bank_id, subtype)"),
    _idx("idx_mm_entity_id", "CREATE INDEX idx_mm_entity_id ON mental_models(entity_id)"),
    # --- async_operations ---
    _idx("idx_ao_bank_id", "CREATE INDEX idx_ao_bank_id ON async_operations(bank_id)"),
    _idx("idx_ao_status", "CREATE INDEX idx_ao_status ON async_operations(status)"),
    _idx("idx_ao_bank_status", "CREATE INDEX idx_ao_bank_status ON async_operations(bank_id, status)"),
    _idx("idx_ao_status_retry", "CREATE INDEX idx_ao_status_retry ON async_operations(status, next_retry_at)"),
    # --- webhooks ---
    _idx("idx_wh_bank_id", "CREATE INDEX idx_wh_bank_id ON webhooks(bank_id)"),
    # --- audit_log ---
    _idx("idx_al_action_started", "CREATE INDEX idx_al_action_started ON audit_log(action, started_at DESC)"),
    _idx("idx_al_bank_started", "CREATE INDEX idx_al_bank_started ON audit_log(bank_id, started_at DESC)"),
    _idx("idx_al_started", "CREATE INDEX idx_al_started ON audit_log(started_at DESC)"),
    # --- observation_sources ---
    _idx(
        "idx_obs_sources_source_id",
        "CREATE INDEX idx_obs_sources_source_id ON observation_sources(source_id, observation_id)",
    ),
]


# ---------------------------------------------------------------------------
# Vector and text indexes (Oracle 23ai specific)
# ---------------------------------------------------------------------------

_DDL_VECTOR_INDEX = _idx(
    "idx_mu_embedding_hnsw",
    "CREATE VECTOR INDEX idx_mu_embedding_hnsw ON memory_units(embedding) "
    "ORGANIZATION NEIGHBOR PARTITIONS "
    "DISTANCE COSINE "
    "WITH TARGET ACCURACY 95",
)

_DDL_TEXT_INDEX = """
BEGIN
    EXECUTE IMMEDIATE '
        CREATE INDEX idx_mu_content_text ON memory_units(text)
        INDEXTYPE IS CTXSYS.CONTEXT
        PARAMETERS (''SYNC (ON COMMIT)'')
    ';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE = -955 THEN NULL;
        ELSE RAISE;
        END IF;
END;
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_oracle_migrations(dsn: str, *, schema: str | None = None) -> None:
    """Run Oracle schema migrations.

    Creates all tables, indexes, and constraints using idempotent DDL.
    Safe to call multiple times.

    Args:
        dsn: Oracle connection string (oracle://user:pass@host:port/service)
        schema: Target schema (Oracle user). None uses the connecting user's default.
    """
    try:
        import oracledb  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "python-oracledb is required for Oracle migrations. Install with: pip install oracledb"
        ) from None

    oracledb.defaults.fetch_lobs = False

    # Parse URL-format DSN
    parsed = urlparse(dsn)
    connect_kwargs: dict = {}
    if parsed.scheme in ("oracle", "oracle+oracledb"):
        connect_kwargs["user"] = parsed.username
        connect_kwargs["password"] = parsed.password
        host = parsed.hostname or "localhost"
        port = parsed.port or 1521
        service = parsed.path.lstrip("/") if parsed.path else "FREEPDB1"
        connect_kwargs["dsn"] = f"{host}:{port}/{service}"
    else:
        connect_kwargs["dsn"] = dsn

    logger.info("Running Oracle schema migrations (dsn=%s, schema=%s)", connect_kwargs.get("dsn", dsn), schema)

    conn = oracledb.connect(**connect_kwargs)
    cursor = conn.cursor()

    try:
        # Wait up to 30s for DDL locks instead of failing immediately (ORA-00054)
        cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT = 30")

        # Set schema if specified
        if schema:
            cursor.execute(f'ALTER SESSION SET CURRENT_SCHEMA = "{schema}"')

        # Create tables
        for i, ddl in enumerate(_DDL_TABLES):
            try:
                cursor.execute(ddl.strip())
                conn.commit()
            except oracledb.DatabaseError as e:
                err = e.args[0]
                if hasattr(err, "code") and err.code == 955:
                    # ORA-00955: name is already used by an existing object
                    pass
                else:
                    logger.error("Failed to create table (statement %d): %s", i, e)
                    raise

        # Convert memory_units to automatic list partitioning on bank_id.
        # New installs get this from CREATE TABLE; this handles existing installs.
        # Oracle 12.2+ supports online conversion via ALTER TABLE MODIFY.
        #
        # IMPORTANT: ALTER TABLE MODIFY PARTITION invalidates CTXSYS.CONTEXT
        # domain indexes (ORA-29861). We drop the text index before conversion
        # and recreate it afterward. The text index creation below handles both
        # fresh installs and this post-conversion recreation.
        try:
            # Drop text index first if it exists — it will be invalidated by partitioning.
            try:
                cursor.execute("DROP INDEX idx_mu_content_text FORCE")
                conn.commit()
                logger.debug("Dropped text index before partitioning conversion")
            except oracledb.DatabaseError:
                pass  # Index doesn't exist yet (fresh install)

            cursor.execute("""
                ALTER TABLE memory_units MODIFY
                PARTITION BY LIST (bank_id) AUTOMATIC
                (PARTITION p_default VALUES ('__default__'))
            """)
            conn.commit()
            logger.info("memory_units partitioned by bank_id (automatic list)")
        except oracledb.DatabaseError as e:
            err = e.args[0]
            # ORA-14504: table is already partitioned — safe to ignore
            if hasattr(err, "code") and err.code == 14504:
                logger.debug("memory_units already partitioned")
            else:
                logger.debug("Partitioning memory_units skipped: %s", e)

        # Deduplicate memory_links before creating unique index.
        # Earlier versions lacked a unique constraint, so duplicate rows may exist.
        try:
            cursor.execute("""
                DELETE FROM memory_links WHERE ROWID IN (
                    SELECT rid FROM (
                        SELECT ROWID AS rid,
                               ROW_NUMBER() OVER (
                                   PARTITION BY from_unit_id, to_unit_id, link_type,
                                                NVL(entity_id, HEXTORAW('00000000000000000000000000000000'))
                                   ORDER BY created_at
                               ) AS rn
                        FROM memory_links
                    ) WHERE rn > 1
                )
            """)
            if cursor.rowcount > 0:
                logger.info("Deduplicated %d memory_links rows before unique index creation", cursor.rowcount)
            conn.commit()
        except oracledb.DatabaseError as e:
            logger.debug("memory_links dedup skipped (table may not exist yet): %s", e)

        # Create B-tree indexes
        for idx_ddl in _DDL_INDEXES:
            try:
                cursor.execute(idx_ddl.strip())
                conn.commit()
            except oracledb.DatabaseError as e:
                logger.debug("Index creation (may already exist): %s", e)

        # Create vector index
        try:
            cursor.execute(_DDL_VECTOR_INDEX.strip())
            conn.commit()
        except oracledb.DatabaseError as e:
            logger.debug("Vector index creation (may already exist or VECTOR not supported): %s", e)

        # Create Oracle Text index
        try:
            cursor.execute(_DDL_TEXT_INDEX.strip())
            conn.commit()
        except oracledb.DatabaseError as e:
            logger.debug("Text index creation (may already exist): %s", e)

        # Backfill observation_sources from source_memory_ids CLOB (JSON array).
        # Uses MERGE to be idempotent — safe to run multiple times.
        try:
            cursor.execute("""
                MERGE INTO observation_sources tgt
                USING (
                    SELECT mu.id AS observation_id,
                           HEXTORAW(jt.source_id) AS source_id
                    FROM memory_units mu,
                         JSON_TABLE(mu.source_memory_ids, '$[*]'
                             COLUMNS (source_id VARCHAR2(36) PATH '$')
                         ) jt
                    WHERE mu.fact_type = 'observation'
                      AND mu.source_memory_ids IS NOT NULL
                ) src
                ON (tgt.observation_id = src.observation_id AND tgt.source_id = src.source_id)
                WHEN NOT MATCHED THEN
                    INSERT (observation_id, source_id) VALUES (src.observation_id, src.source_id)
            """)
            conn.commit()
            logger.info("observation_sources backfill completed")
        except oracledb.DatabaseError as e:
            logger.debug("observation_sources backfill (may be empty or already done): %s", e)

        logger.info("Oracle schema migrations completed successfully")

    finally:
        cursor.close()
        conn.close()
