"""Add managed image assets and their streaming transfer support.

Revision ID: a9c4e7f1b2d6
Revises: b6d2f8a4c1e7
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a9c4e7f1b2d6"
down_revision: str | Sequence[str] | None = "b6d2f8a4c1e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        ALTER TABLE {schema}async_operations
            ADD COLUMN idempotency_key_hash CHAR(64),
            ADD COLUMN request_fingerprint CHAR(64);
        CREATE UNIQUE INDEX uq_async_operations_idempotency
            ON {schema}async_operations(bank_id, operation_type, idempotency_key_hash)
            WHERE idempotency_key_hash IS NOT NULL;

        CREATE TABLE {schema}image_assets (
            bank_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            storage_key TEXT NOT NULL UNIQUE,
            mime_type TEXT NOT NULL,
            size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
            sha256 CHAR(64) NOT NULL,
            width INTEGER NOT NULL CHECK (width > 0),
            height INTEGER NOT NULL CHECK (height > 0),
            status TEXT NOT NULL DEFAULT 'ready'
                CHECK (status IN ('ready', 'failed', 'deleting')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, asset_id),
            FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_image_assets_bank_created
            ON {schema}image_assets(bank_id, created_at DESC, asset_id);

        CREATE TABLE {schema}document_image_links (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            image_context TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, document_id, asset_id),
            UNIQUE (bank_id, document_id, ordinal),
            FOREIGN KEY (document_id, bank_id) REFERENCES {schema}documents(id, bank_id) ON DELETE CASCADE,
            FOREIGN KEY (bank_id, asset_id) REFERENCES {schema}image_assets(bank_id, asset_id) ON DELETE RESTRICT
        );
        CREATE INDEX idx_document_image_links_asset
            ON {schema}document_image_links(bank_id, asset_id);

        CREATE TABLE {schema}file_storage_chunks (
            storage_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            data BYTEA NOT NULL,
            PRIMARY KEY (storage_key, ordinal),
            FOREIGN KEY (storage_key) REFERENCES {schema}file_storage(storage_key) ON DELETE CASCADE
        );

        CREATE TABLE {schema}transfer_staging (
            transfer_id UUID PRIMARY KEY,
            bank_id TEXT NOT NULL,
            storage_key TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL,
            FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_transfer_staging_expiry
            ON {schema}transfer_staging(expires_at);
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}transfer_staging")
    op.execute(f"DROP TABLE IF EXISTS {schema}file_storage_chunks")
    op.execute(f"DROP TABLE IF EXISTS {schema}document_image_links")
    op.execute(f"DROP TABLE IF EXISTS {schema}image_assets")
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_async_operations_idempotency")
    op.execute(f"ALTER TABLE {schema}async_operations DROP COLUMN IF EXISTS request_fingerprint")
    op.execute(f"ALTER TABLE {schema}async_operations DROP COLUMN IF EXISTS idempotency_key_hash")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE async_operations ADD idempotency_key_hash CHAR(64)")
    op.execute("ALTER TABLE async_operations ADD request_fingerprint CHAR(64)")
    op.execute(
        "CREATE UNIQUE INDEX uq_async_operations_idempotency "
        "ON async_operations(bank_id, operation_type, idempotency_key_hash)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS image_assets (
            bank_id VARCHAR2(256) NOT NULL,
            asset_id VARCHAR2(512) NOT NULL,
            storage_key VARCHAR2(512) NOT NULL,
            mime_type VARCHAR2(64) NOT NULL,
            size_bytes NUMBER(19) NOT NULL,
            sha256 CHAR(64) NOT NULL,
            width NUMBER(10) NOT NULL,
            height NUMBER(10) NOT NULL,
            status VARCHAR2(32) DEFAULT 'ready' NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_image_assets PRIMARY KEY (bank_id, asset_id),
            CONSTRAINT uq_image_assets_storage UNIQUE (storage_key),
            CONSTRAINT fk_image_assets_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT ck_image_assets_size CHECK (size_bytes >= 0),
            CONSTRAINT ck_image_assets_width CHECK (width > 0),
            CONSTRAINT ck_image_assets_height CHECK (height > 0),
            CONSTRAINT ck_image_assets_status CHECK (status IN ('ready','failed','deleting'))
        )
        """
    )
    op.execute("CREATE INDEX idx_image_assets_bank_created ON image_assets(bank_id, created_at, asset_id)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_image_links (
            bank_id VARCHAR2(256) NOT NULL,
            document_id VARCHAR2(512) NOT NULL,
            asset_id VARCHAR2(512) NOT NULL,
            ordinal NUMBER(10) NOT NULL,
            image_context CLOB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_document_image_links PRIMARY KEY (bank_id, document_id, asset_id),
            CONSTRAINT uq_dil_ordinal UNIQUE (bank_id, document_id, ordinal),
            CONSTRAINT ck_dil_ordinal CHECK (ordinal >= 0),
            CONSTRAINT fk_dil_document FOREIGN KEY (document_id, bank_id)
                REFERENCES documents(id, bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_dil_asset FOREIGN KEY (bank_id, asset_id)
                REFERENCES image_assets(bank_id, asset_id)
        )
        """
    )
    op.execute("CREATE INDEX idx_dil_asset ON document_image_links(bank_id, asset_id)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS file_storage_chunks (
            storage_key VARCHAR2(512) NOT NULL,
            ordinal NUMBER(10) NOT NULL,
            data BLOB NOT NULL,
            CONSTRAINT pk_file_storage_chunks PRIMARY KEY (storage_key, ordinal),
            CONSTRAINT ck_file_storage_chunks_ordinal CHECK (ordinal >= 0),
            CONSTRAINT fk_file_storage_chunks_parent FOREIGN KEY (storage_key)
                REFERENCES file_storage(storage_key) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS transfer_staging (
            transfer_id RAW(16) NOT NULL,
            bank_id VARCHAR2(256) NOT NULL,
            storage_key VARCHAR2(512) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT pk_transfer_staging PRIMARY KEY (transfer_id),
            CONSTRAINT uq_transfer_staging_key UNIQUE (storage_key),
            CONSTRAINT fk_transfer_staging_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX idx_transfer_staging_expiry ON transfer_staging(expires_at)")


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS transfer_staging")
    op.execute("DROP TABLE IF EXISTS file_storage_chunks")
    op.execute("DROP TABLE IF EXISTS document_image_links")
    op.execute("DROP TABLE IF EXISTS image_assets")
    op.execute("DROP INDEX uq_async_operations_idempotency")
    op.execute("ALTER TABLE async_operations DROP COLUMN request_fingerprint")
    op.execute("ALTER TABLE async_operations DROP COLUMN idempotency_key_hash")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
