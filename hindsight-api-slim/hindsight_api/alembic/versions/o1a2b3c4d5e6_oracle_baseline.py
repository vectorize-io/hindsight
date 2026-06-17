"""oracle_baseline

Brings a fresh Oracle 23ai database up to the current schema in a single step.
PostgreSQL is a no-op here — the prior 59 revisions already build the PG schema
incrementally; this revision just closes the loop so both dialects share a
single head from this point on.

After this migration ships, *every* new revision must fill both the ``_pg_*``
and ``_oracle_*`` slots (or explicitly leave one ``None``); a CI check enforces
that.

Tables mirror the PostgreSQL schema but use Oracle-native types:
  - UUID                       -> RAW(16) DEFAULT SYS_GUID()
  - TEXT / large VARCHAR        -> CLOB
  - JSONB                       -> CLOB with IS JSON CHECK
  - BOOLEAN                     -> NUMBER(1)
  - FLOAT                       -> BINARY_DOUBLE
  - VARCHAR[]                   -> CLOB (JSON array stored as string)
  - BYTEA                       -> BLOB
  - vector(384)                 -> VECTOR(384, FLOAT32) (Oracle 23ai native)

Revision ID: o1a2b3c4d5e6
Revises: k6l7m8n9o0p1
Create Date: 2026-04-29
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "o1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "k6l7m8n9o0p1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Tables — created in dependency order
# ---------------------------------------------------------------------------

_TABLES: tuple[str, ...] = (
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
    # memory_units uses automatic list partitioning on bank_id at create time —
    # no post-create ALTER required (we used to do that for legacy installs).
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
        edited_at         TIMESTAMP WITH TIME ZONE,
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
    # Cold archive for curation: invalidated facts are MOVED here out of
    # memory_units so the recall hot-path never sees them. Mirrors memory_units
    # plus invalidation bookkeeping and an entity-id snapshot for lossless revert.
    # No `embedding` column: the archive is cold storage and revert recomputes the
    # embedding, so there is no archive vector to fall out of sync with the live
    # model's dimension on a model switch (#2209).
    """
    CREATE TABLE IF NOT EXISTS invalidated_memory_units (
        id                RAW(16)        NOT NULL,
        bank_id           VARCHAR2(256)  NOT NULL,
        document_id       VARCHAR2(512),
        chunk_id          VARCHAR2(512),
        text              CLOB           NOT NULL,
        context           CLOB,
        event_date        TIMESTAMP WITH TIME ZONE NOT NULL,
        occurred_start    TIMESTAMP WITH TIME ZONE,
        occurred_end      TIMESTAMP WITH TIME ZONE,
        mentioned_at      TIMESTAMP WITH TIME ZONE,
        fact_type         VARCHAR2(64)   DEFAULT 'world' NOT NULL,
        confidence_score  BINARY_DOUBLE,
        access_count      NUMBER(10)     DEFAULT 0 NOT NULL,
        consolidated_at   TIMESTAMP WITH TIME ZONE,
        observation_scopes CLOB          CONSTRAINT imu_obs_scopes_json CHECK (observation_scopes IS JSON OR observation_scopes IS NULL),
        tags              CLOB           DEFAULT '[]' NOT NULL,
        metadata          CLOB           DEFAULT '{}' NOT NULL
                                         CONSTRAINT imu_metadata_json CHECK (metadata IS JSON),
        proof_count       NUMBER(10)     DEFAULT 1,
        source_memory_ids CLOB,
        history           CLOB           DEFAULT '[]'
                                         CONSTRAINT imu_history_json CHECK (history IS JSON OR history IS NULL),
        text_signals      CLOB,
        consolidation_failed_at TIMESTAMP WITH TIME ZONE,
        search_vector     CLOB,
        edited_at         TIMESTAMP WITH TIME ZONE,
        created_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        updated_at        TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
        invalidation_reason CLOB,
        invalidated_at    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
        entity_ids        CLOB           CONSTRAINT imu_entity_ids_json CHECK (entity_ids IS JSON OR entity_ids IS NULL),
        CONSTRAINT pk_invalidated_memory_units PRIMARY KEY (id),
        CONSTRAINT fk_imu_document FOREIGN KEY (document_id, bank_id)
            REFERENCES documents(id, bank_id) ON DELETE CASCADE
    )
    """,
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
    """
    CREATE TABLE IF NOT EXISTS unit_entities (
        unit_id           RAW(16)        NOT NULL,
        entity_id         RAW(16)        NOT NULL,
        CONSTRAINT pk_unit_entities PRIMARY KEY (unit_id, entity_id),
        CONSTRAINT fk_ue_unit FOREIGN KEY (unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
        CONSTRAINT fk_ue_entity FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
    )
    """,
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
        CONSTRAINT chk_mm_subtype CHECK (subtype IN ('directive', 'pinned'))
    )
    """,
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
        CONSTRAINT chk_ao_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))
    )
    """,
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
    """
    CREATE TABLE IF NOT EXISTS file_storage (
        storage_key       VARCHAR2(512)  NOT NULL,
        data              BLOB           NOT NULL,
        CONSTRAINT pk_file_storage PRIMARY KEY (storage_key)
    )
    """,
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
    """
    CREATE TABLE IF NOT EXISTS observation_sources (
        observation_id    RAW(16)        NOT NULL,
        source_id         RAW(16)        NOT NULL,
        CONSTRAINT pk_observation_sources PRIMARY KEY (observation_id, source_id),
        CONSTRAINT fk_obs_src_observation FOREIGN KEY (observation_id)
            REFERENCES memory_units(id) ON DELETE CASCADE
    )
    """,
)


# ---------------------------------------------------------------------------
# B-tree indexes
# ---------------------------------------------------------------------------

_INDEXES: tuple[str, ...] = (
    # documents
    "CREATE INDEX idx_docs_bank_id ON documents(bank_id)",
    "CREATE INDEX idx_docs_content_hash ON documents(content_hash)",
    # chunks
    "CREATE INDEX idx_chunks_document_id ON chunks(document_id)",
    "CREATE INDEX idx_chunks_bank_id ON chunks(bank_id)",
    # memory_units
    "CREATE INDEX idx_mu_bank_id ON memory_units(bank_id)",
    "CREATE INDEX idx_mu_document_id ON memory_units(document_id)",
    "CREATE INDEX idx_mu_chunk_id ON memory_units(chunk_id)",
    "CREATE INDEX idx_mu_event_date ON memory_units(event_date DESC)",
    "CREATE INDEX idx_mu_bank_date ON memory_units(bank_id, event_date DESC)",
    "CREATE INDEX idx_mu_access_count ON memory_units(access_count DESC)",
    "CREATE INDEX idx_mu_fact_type ON memory_units(fact_type)",
    "CREATE INDEX idx_mu_bank_fact_type ON memory_units(bank_id, fact_type)",
    "CREATE INDEX idx_mu_bank_type_date ON memory_units(bank_id, fact_type, event_date DESC)",
    # entities
    "CREATE INDEX idx_ent_bank_id ON entities(bank_id)",
    "CREATE INDEX idx_ent_canonical_name ON entities(canonical_name)",
    "CREATE INDEX idx_ent_bank_name ON entities(bank_id, canonical_name)",
    "CREATE UNIQUE INDEX idx_ent_bank_lower_name ON entities(bank_id, LOWER(canonical_name))",
    # unit_entities
    "CREATE INDEX idx_ue_unit ON unit_entities(unit_id)",
    "CREATE INDEX idx_ue_entity ON unit_entities(entity_id)",
    # entity_cooccurrences
    "CREATE INDEX idx_ec_entity1 ON entity_cooccurrences(entity_id_1)",
    "CREATE INDEX idx_ec_entity2 ON entity_cooccurrences(entity_id_2)",
    "CREATE INDEX idx_ec_count ON entity_cooccurrences(cooccurrence_count DESC)",
    # memory_links — function-based unique index uses NVL with the nil UUID raw
    # to handle nullable entity_id (matches PG idx_memory_links_unique).
    "CREATE UNIQUE INDEX idx_memory_links_unique ON memory_links("
    "from_unit_id, to_unit_id, link_type, "
    "NVL(entity_id, HEXTORAW('00000000000000000000000000000000')))",
    "CREATE INDEX idx_ml_from_unit ON memory_links(from_unit_id)",
    "CREATE INDEX idx_ml_to_unit ON memory_links(to_unit_id)",
    "CREATE INDEX idx_ml_entity ON memory_links(entity_id)",
    "CREATE INDEX idx_ml_link_type ON memory_links(link_type)",
    "CREATE INDEX idx_ml_bank_id ON memory_links(bank_id)",
    # directives
    "CREATE INDEX idx_dir_bank_id ON directives(bank_id)",
    "CREATE INDEX idx_dir_bank_active ON directives(bank_id, is_active)",
    # mental_models
    "CREATE INDEX idx_mm_bank_id ON mental_models(bank_id)",
    "CREATE INDEX idx_mm_subtype ON mental_models(bank_id, subtype)",
    "CREATE INDEX idx_mm_entity_id ON mental_models(entity_id)",
    # async_operations
    "CREATE INDEX idx_ao_bank_id ON async_operations(bank_id)",
    "CREATE INDEX idx_ao_status ON async_operations(status)",
    "CREATE INDEX idx_ao_bank_status ON async_operations(bank_id, status)",
    "CREATE INDEX idx_ao_status_retry ON async_operations(status, next_retry_at)",
    # webhooks
    "CREATE INDEX idx_wh_bank_id ON webhooks(bank_id)",
    # audit_log
    "CREATE INDEX idx_al_action_started ON audit_log(action, started_at DESC)",
    "CREATE INDEX idx_al_bank_started ON audit_log(bank_id, started_at DESC)",
    "CREATE INDEX idx_al_started ON audit_log(started_at DESC)",
    # observation_sources
    "CREATE INDEX idx_obs_sources_source_id ON observation_sources(source_id, observation_id)",
)

_VECTOR_INDEX = (
    "CREATE VECTOR INDEX idx_mu_embedding_hnsw ON memory_units(embedding) "
    "ORGANIZATION NEIGHBOR PARTITIONS "
    "DISTANCE COSINE "
    "WITH TARGET ACCURACY 95"
)

# Oracle Text (CTXSYS.CONTEXT) — ``SYNC (ON COMMIT)`` makes it auto-update
# without a maintenance job. Doubled single quotes for the embedded literal.
_TEXT_INDEX = (
    "BEGIN "
    "EXECUTE IMMEDIATE '"
    "CREATE INDEX idx_mu_content_text ON memory_units(text) "
    "INDEXTYPE IS CTXSYS.CONTEXT "
    "PARAMETERS (''SYNC (ON COMMIT)'')"
    "'; "
    "EXCEPTION WHEN OTHERS THEN "
    "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
    "END;"
)


def _execute_ignoring_955(sql: str) -> None:
    """Run a CREATE statement and swallow ORA-00955 (object already exists).

    Wraps the statement in PL/SQL so the exception handler runs server-side —
    no round-trip cost for the common case.
    """
    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_upgrade() -> None:
    bind = op.get_bind()
    # Tolerate concurrent DDL instead of failing immediately (ORA-00054).
    bind.exec_driver_sql("ALTER SESSION SET DDL_LOCK_TIMEOUT = 30")

    for ddl in _TABLES:
        _execute_ignoring_955(ddl)

    for idx in _INDEXES:
        _execute_ignoring_955(idx)

    # Hindsight on Oracle requires 23ai with VECTOR support (ASSM tablespace)
    # and the CTXSYS package for full-text. Both index creations must succeed
    # — the migration fails hard if either feature is unavailable, by design.
    # We only swallow ORA-00955 (object already exists) so reruns are safe.
    _execute_ignoring_955(_VECTOR_INDEX)
    bind.exec_driver_sql(_TEXT_INDEX)


def _oracle_downgrade() -> None:
    # Baseline downgrades aren't supported — dropping every table here would
    # destroy customer data. Use point-in-time recovery instead.
    raise NotImplementedError("Cannot downgrade past the Oracle baseline.")


def upgrade() -> None:
    run_for_dialect(oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(oracle=_oracle_downgrade)
