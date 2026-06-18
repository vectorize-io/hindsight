"""Tests for migration g7h8i9j0k1l2 (backsweep orphaned memory_units).

Uses a dedicated pg0 instance (port 5562) so the test can control exactly
which migrations have run before inserting the orphan seed data.
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


def _upgrade(db_url: str, revision: str) -> None:
    command.upgrade(_alembic_cfg(db_url), revision)


def _reset_public_schema(db_url: str) -> None:
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            # This test rewinds/replays migration history against a persistent
            # pg0 instance. Rebuild only its dedicated public schema so a
            # previous run cannot leave alembic_version ahead of the real DDL.
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Fixture: fresh database at the revision just before the backsweep
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pre_backsweep_db_url() -> str:
    """
    Spin up a dedicated pg0 instance and ensure schema is at the revision
    just before the backsweep so each test can seed orphan data and then
    apply the backsweep itself.

    Because pg0 data directories persist across test runs, the DB may already
    have schema from a previous test run. Reset this test's dedicated schema
    first, then migrate to the real pre-backsweep revision instead of stamping
    a head schema backward.
    """
    from hindsight_api.pg0 import EmbeddedPostgres

    pg0 = EmbeddedPostgres(name="hindsight-backsweep-test", port=5562)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    _reset_public_schema(url)
    _upgrade(url, "f6g7h8i9j0k1")
    return url


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_backsweep_removes_orphans_and_preserves_legit_rows(pre_backsweep_db_url: str) -> None:
    """
    Seed four kinds of rows then apply the backsweep migration and verify:

    Rows that MUST be deleted
    ─────────────────────────
    A. Any fact_type, bank_id missing from banks
       → Pass 1 deletes these regardless of fact_type or source links.

    B. observation, bank exists, but ALL source_memory_ids are gone
       → Pass 2 deletes these.

    Rows that MUST survive
    ──────────────────────
    C. observation, bank exists, at least ONE source_memory_id still live
       → Pass 2 must not touch these.

    D. Non-observation (world), bank exists, no sources (not relevant)
       → Pass 1 must not touch these (bank exists).
    """
    db_url = pre_backsweep_db_url
    engine = create_engine(db_url)

    alive_bank = f"bank_{uuid.uuid4().hex[:8]}"
    ghost_bank = f"bank_{uuid.uuid4().hex[:8]}"  # never inserted into banks

    # UUIDs for memory units
    id_pass1_world = uuid.uuid4()  # A: world unit, ghost bank
    id_pass1_obs = uuid.uuid4()  # A: observation, ghost bank
    id_pass2_obs = uuid.uuid4()  # B: observation, all sources gone
    id_keep_obs = uuid.uuid4()  # C: observation with one live source
    id_keep_world = uuid.uuid4()  # D: world unit, alive bank
    id_live_source = uuid.uuid4()  # live source for C

    with engine.connect() as conn:
        # --- banks ---
        conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": alive_bank})

        # --- seed memory_units ---
        def insert_mu(
            uid: uuid.UUID,
            bank: str,
            fact_type: str,
            sources: list[uuid.UUID] | None = None,
        ) -> None:
            src_arr = "{" + ",".join(str(s) for s in (sources or [])) + "}"
            conn.execute(
                text(
                    """
                    INSERT INTO memory_units
                        (id, bank_id, text, event_date, fact_type, source_memory_ids)
                    VALUES
                        (:id, :bank, :text, now(), :ft, CAST(:src AS uuid[]))
                    """
                ),
                {"id": uid, "bank": bank, "text": "test", "ft": fact_type, "src": src_arr},
            )

        # A: ghost-bank rows (Pass 1 targets)
        insert_mu(id_pass1_world, ghost_bank, "world")
        insert_mu(id_pass1_obs, ghost_bank, "observation", sources=[uuid.uuid4()])

        # B: observation with all-dead sources (Pass 2 target)
        insert_mu(id_pass2_obs, alive_bank, "observation", sources=[uuid.uuid4(), uuid.uuid4()])

        # C: observation with one live source (must survive)
        insert_mu(id_live_source, alive_bank, "world")
        insert_mu(id_keep_obs, alive_bank, "observation", sources=[id_live_source, uuid.uuid4()])

        # D: world unit in alive bank (must survive)
        insert_mu(id_keep_world, alive_bank, "world")

        conn.commit()

    # --- apply the backsweep ---
    _upgrade(db_url, "g7h8i9j0k1l2")

    # --- verify ---
    with engine.connect() as conn:

        def exists(uid: uuid.UUID) -> bool:
            return conn.execute(text("SELECT 1 FROM memory_units WHERE id = :id"), {"id": uid}).fetchone() is not None

        # Must be gone
        assert not exists(id_pass1_world), "Pass 1: world unit with ghost bank should be deleted"
        assert not exists(id_pass1_obs), "Pass 1: observation with ghost bank should be deleted"
        assert not exists(id_pass2_obs), "Pass 2: observation with all-dead sources should be deleted"

        # Must survive
        assert exists(id_keep_obs), "observation with a live source must not be deleted"
        assert exists(id_keep_world), "world unit in alive bank must not be deleted"
        assert exists(id_live_source), "live source memory unit must not be deleted"

    engine.dispose()
