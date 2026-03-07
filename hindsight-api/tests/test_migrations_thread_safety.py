import threading
import time

from hindsight_api import migrations


def test_run_migrations_internal_serializes_alembic_upgrade(monkeypatch):
    max_concurrent_upgrades = 0
    active_upgrades = 0
    active_lock = threading.Lock()
    start_barrier = threading.Barrier(2)

    def fake_upgrade(_cfg, _revision):
        nonlocal max_concurrent_upgrades, active_upgrades
        with active_lock:
            active_upgrades += 1
            max_concurrent_upgrades = max(max_concurrent_upgrades, active_upgrades)
        time.sleep(0.05)
        with active_lock:
            active_upgrades -= 1

    monkeypatch.setattr(migrations.command, "upgrade", fake_upgrade)

    errors = []

    def run_in_thread(schema):
        try:
            start_barrier.wait()
            migrations._run_migrations_internal(
                "postgresql://user:pass@localhost/db",
                "/tmp/alembic",
                schema=schema,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    threads = [
        threading.Thread(target=run_in_thread, args=("tenant_alpha",)),
        threading.Thread(target=run_in_thread, args=("tenant_beta",)),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert max_concurrent_upgrades == 1
