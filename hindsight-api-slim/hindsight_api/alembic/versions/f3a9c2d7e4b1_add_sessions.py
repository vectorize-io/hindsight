"""Add sessions: a live conversation a client streams in, and the watcher's side of it.

A session is one conversation (a coding agent's run, a support chat) that a client sends piece by
piece while it happens, and that may get notes back from Hindsight's memory watcher. Two tables:

- ``sessions``: one row per conversation. It names the document the conversation is saved into,
  and holds the watcher's running state (what the agent is doing, what it was already told) and the
  cursors that make the protocol safe to retry: the next log position, how far the conversation was
  saved into its document, and how far the client acknowledged the notes it was given. ``save_plan``
  is a save in flight (its range and retain operation id), written before the retain is submitted so
  a retried save resubmits the same operation and the document never gets a turn twice.
  ``pending_since`` and ``last_event_at`` are database times, so the watcher can wait for a session
  to settle before checking it whichever server took the append and whichever worker runs the check.
- ``session_events``: the session's log, in order. The client's own events (a message, a tool call,
  its result) and the watcher's (a check it ran, a note it wrote) share one sequence, so a reader
  following ``seq`` sees the conversation and the watcher's reactions interleaved as they happened.

The session is not the document. The log is the raw stream, written on every append; the document
is what the conversation becomes in memory, written when the session decides to save (end of a
turn, close). Keeping them apart is what lets the session choose when to pay for extraction.

``event_id`` is the client's id for an event, unique in its session: an append that is retried
after a lost response inserts nothing twice. ``seq`` is the server's order.

Revision ID: f3a9c2d7e4b1
Revises: c7e2f9a4d1b8
Create Date: 2026-10-19
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3a9c2d7e4b1"
down_revision: str | Sequence[str] | None = "c7e2f9a4d1b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}sessions (
            bank_id         TEXT NOT NULL,
            session_id      TEXT NOT NULL,
            document_id     TEXT NOT NULL,
            settings        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            watch_state     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            next_seq        BIGINT NOT NULL DEFAULT 1,
            saved_seq       BIGINT NOT NULL DEFAULT 0,
            acked_seq       BIGINT NOT NULL DEFAULT 0,
            event_count     INTEGER NOT NULL DEFAULT 0,
            check_count     INTEGER NOT NULL DEFAULT 0,
            note_count      INTEGER NOT NULL DEFAULT 0,
            credit_count    INTEGER NOT NULL DEFAULT 0,
            dismissed_count INTEGER NOT NULL DEFAULT 0,
            checking_since  TIMESTAMPTZ,
            pending_since   TIMESTAMPTZ,
            last_event_at   TIMESTAMPTZ,
            save_plan       JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, session_id),
            FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    # The session list is newest-first per bank.
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_sessions_bank_updated ON {schema}sessions (bank_id, updated_at DESC)")
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}session_events (
            bank_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            seq         BIGINT NOT NULL,
            event_id    TEXT NOT NULL,
            source      TEXT NOT NULL,
            kind        TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            payload     JSONB NOT NULL,
            PRIMARY KEY (bank_id, session_id, seq),
            FOREIGN KEY (bank_id, session_id) REFERENCES {schema}sessions(bank_id, session_id) ON DELETE CASCADE
        )
        """
    )
    # What makes an append idempotent: a retried event finds its id taken.
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_session_events_event_id "
        f"ON {schema}session_events (bank_id, session_id, event_id)"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}session_events")
    op.execute(f"DROP TABLE IF EXISTS {schema}sessions")


def _oracle_execute_ignoring_955(sql: str) -> None:
    """Run a CREATE statement and swallow ORA-00955 (object already exists)."""
    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_upgrade() -> None:
    # Same shape as PG. JSON columns are CLOB IS JSON, as in the rest of the Oracle schema;
    # VARCHAR2(256) ids mirror bank_id/document_id elsewhere.
    _oracle_execute_ignoring_955(
        """
        CREATE TABLE sessions (
            bank_id         VARCHAR2(256) NOT NULL,
            session_id      VARCHAR2(256) NOT NULL,
            document_id     VARCHAR2(512) NOT NULL,
            settings        CLOB DEFAULT '{}' NOT NULL CONSTRAINT sessions_settings_json CHECK (settings IS JSON),
            metadata        CLOB DEFAULT '{}' NOT NULL CONSTRAINT sessions_metadata_json CHECK (metadata IS JSON),
            watch_state     CLOB DEFAULT '{}' NOT NULL CONSTRAINT sessions_watch_state_json CHECK (watch_state IS JSON),
            next_seq        NUMBER(19) DEFAULT 1 NOT NULL,
            saved_seq       NUMBER(19) DEFAULT 0 NOT NULL,
            acked_seq       NUMBER(19) DEFAULT 0 NOT NULL,
            event_count     NUMBER(10) DEFAULT 0 NOT NULL,
            check_count     NUMBER(10) DEFAULT 0 NOT NULL,
            note_count      NUMBER(10) DEFAULT 0 NOT NULL,
            credit_count    NUMBER(10) DEFAULT 0 NOT NULL,
            dismissed_count NUMBER(10) DEFAULT 0 NOT NULL,
            checking_since  TIMESTAMP WITH TIME ZONE,
            pending_since   TIMESTAMP WITH TIME ZONE,
            last_event_at   TIMESTAMP WITH TIME ZONE,
            save_plan       CLOB CONSTRAINT sessions_save_plan_json CHECK (save_plan IS JSON),
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            updated_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_sessions PRIMARY KEY (bank_id, session_id),
            CONSTRAINT fk_sessions_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    _oracle_execute_ignoring_955("CREATE INDEX idx_sessions_bank_updated ON sessions (bank_id, updated_at DESC)")
    _oracle_execute_ignoring_955(
        """
        CREATE TABLE session_events (
            bank_id     VARCHAR2(256) NOT NULL,
            session_id  VARCHAR2(256) NOT NULL,
            seq         NUMBER(19) NOT NULL,
            event_id    VARCHAR2(512) NOT NULL,
            source      VARCHAR2(32) NOT NULL,
            kind        VARCHAR2(64) NOT NULL,
            created_at  TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            payload     CLOB NOT NULL CONSTRAINT session_events_payload_json CHECK (payload IS JSON),
            CONSTRAINT pk_session_events PRIMARY KEY (bank_id, session_id, seq),
            CONSTRAINT fk_session_events_session FOREIGN KEY (bank_id, session_id)
                REFERENCES sessions(bank_id, session_id) ON DELETE CASCADE
        )
        """
    )
    _oracle_execute_ignoring_955(
        "CREATE UNIQUE INDEX uq_session_events_event_id ON session_events (bank_id, session_id, event_id)"
    )


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE session_events")
    op.execute("DROP TABLE sessions")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
