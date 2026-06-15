"""Orchestration tests for run_migrations_for_schemas (per-tenant parallelism).

These cover the fan-out logic deterministically without a real database by
stubbing the per-step migration functions. The real cross-process path is
exercised by the standard migration/integration suites that run against pg0.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hindsight_api import migrations


@pytest.fixture
def record_steps(monkeypatch):
    """Replace the real migration steps with recorders; return the call log."""
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def make(step):
        def _step(database_url, *args, schema=None, **kwargs):
            with lock:
                calls.append((step, schema))

        return _step

    monkeypatch.setattr(migrations, "run_migrations", make("run_migrations"))
    monkeypatch.setattr(migrations, "ensure_embedding_dimension", make("embedding_dimension"))
    monkeypatch.setattr(migrations, "ensure_vector_extension", make("vector_extension"))
    monkeypatch.setattr(migrations, "ensure_text_search_extension", make("text_search_extension"))
    return calls


def test_empty_schema_list_is_noop(record_steps):
    migrations.run_migrations_for_schemas("postgresql://x/db", [])
    assert record_steps == []


def test_sequential_runs_all_steps_in_order_per_schema(record_steps):
    migrations.run_migrations_for_schemas(
        "postgresql://x/db",
        ["a", "b"],
        concurrency=1,
        embedding_dimension=768,
    )
    # Each schema: migrate -> embedding dim -> vector ext -> text-search ext.
    assert record_steps == [
        ("run_migrations", "a"),
        ("embedding_dimension", "a"),
        ("vector_extension", "a"),
        ("text_search_extension", "a"),
        ("run_migrations", "b"),
        ("embedding_dimension", "b"),
        ("vector_extension", "b"),
        ("text_search_extension", "b"),
    ]


def test_skips_embedding_dim_when_none_and_extensions_when_disabled(record_steps):
    migrations.run_migrations_for_schemas(
        "postgresql://x/db",
        ["a"],
        concurrency=1,
        embedding_dimension=None,
        ensure_extensions=False,
    )
    assert record_steps == [("run_migrations", "a")]


def test_parallel_fans_out_across_schemas(monkeypatch):
    """concurrency>1 runs distinct schemas at the same time (not serialized)."""
    max_active = 0
    active = 0
    lock = threading.Lock()

    def slow_migrate(database_url, *args, schema=None, **kwargs):
        nonlocal max_active, active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    monkeypatch.setattr(migrations, "run_migrations", slow_migrate)
    monkeypatch.setattr(migrations, "ensure_embedding_dimension", lambda *a, **k: None)
    monkeypatch.setattr(migrations, "ensure_vector_extension", lambda *a, **k: None)
    monkeypatch.setattr(migrations, "ensure_text_search_extension", lambda *a, **k: None)
    # Run the parallel branch in-process so the monkeypatched steps are visible.
    monkeypatch.setattr(
        migrations,
        "_make_migration_executor",
        lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )

    migrations.run_migrations_for_schemas(
        "postgresql://x/db",
        ["a", "b", "c", "d"],
        concurrency=3,
    )
    assert max_active == 3


def test_parallel_aggregates_per_schema_failures(monkeypatch):
    """One failing schema does not hide the others, and all are still attempted."""
    attempted: list[str] = []
    lock = threading.Lock()

    def migrate(database_url, *args, schema=None, **kwargs):
        with lock:
            attempted.append(schema)
        if schema in ("b", "d"):
            raise RuntimeError(f"boom {schema}")

    monkeypatch.setattr(migrations, "run_migrations", migrate)
    monkeypatch.setattr(migrations, "ensure_embedding_dimension", lambda *a, **k: None)
    monkeypatch.setattr(migrations, "ensure_vector_extension", lambda *a, **k: None)
    monkeypatch.setattr(migrations, "ensure_text_search_extension", lambda *a, **k: None)
    monkeypatch.setattr(
        migrations,
        "_make_migration_executor",
        lambda max_workers: ThreadPoolExecutor(max_workers=max_workers),
    )

    with pytest.raises(RuntimeError) as exc_info:
        migrations.run_migrations_for_schemas(
            "postgresql://x/db",
            ["a", "b", "c", "d"],
            concurrency=2,
        )

    assert set(attempted) == {"a", "b", "c", "d"}
    message = str(exc_info.value)
    assert "b" in message and "d" in message
    assert "2 of 4" in message


def test_worker_is_picklable():
    """ProcessPoolExecutor requires the worker to be importable/picklable."""
    import pickle

    pickle.loads(pickle.dumps(migrations._migrate_one_schema_pg))
