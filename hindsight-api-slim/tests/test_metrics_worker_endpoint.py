"""Each worker process gets its own metrics port, claimed by lock file.

What matters is that two workers never serve the same port, that a slot freed by a dead worker is
claimable again, and that the port really serves that worker's registry.
"""

import socket
import urllib.request

import pytest
from prometheus_client import CollectorRegistry, Counter

from hindsight_api import metrics_worker_endpoint


def _free_port_block(n: int) -> int:
    """A base port with ``n`` consecutive ports free right now."""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        if base + n >= 65535:
            continue
        ok = True
        for p in range(base, base + n):
            with socket.socket() as t:
                try:
                    t.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
                    break
        if ok:
            return base
    pytest.skip("no block of free ports")


@pytest.fixture(autouse=True)
def _fresh_module_state(monkeypatch):
    monkeypatch.setattr(metrics_worker_endpoint, "_held", None)
    monkeypatch.setattr(metrics_worker_endpoint, "_port", None)


def test_concurrent_claims_get_distinct_slots(tmp_path):
    first = metrics_worker_endpoint.claim_slot(9100, 3, lock_dir=str(tmp_path))
    second = metrics_worker_endpoint.claim_slot(9100, 3, lock_dir=str(tmp_path))
    assert first is not None and second is not None
    assert (first[0], second[0]) == (9100, 9101)
    first[1].close()
    second[1].close()


def test_no_slot_when_every_slot_is_held(tmp_path):
    held = [metrics_worker_endpoint.claim_slot(9100, 2, lock_dir=str(tmp_path)) for _ in range(2)]
    assert metrics_worker_endpoint.claim_slot(9100, 2, lock_dir=str(tmp_path)) is None
    for claim in held:
        claim[1].close()


def test_a_released_slot_is_claimed_again(tmp_path):
    """A worker that dies releases its lock, so its replacement takes over the same port."""
    first = metrics_worker_endpoint.claim_slot(9100, 2, lock_dir=str(tmp_path))
    second = metrics_worker_endpoint.claim_slot(9100, 2, lock_dir=str(tmp_path))
    first[1].close()  # the process holding slot 0 exits
    again = metrics_worker_endpoint.claim_slot(9100, 2, lock_dir=str(tmp_path))
    assert again is not None and again[0] == 9100
    again[1].close()
    second[1].close()


def test_disabled_without_a_base_port(tmp_path):
    assert metrics_worker_endpoint.start(0, 2, lock_dir=str(tmp_path)) is None


def test_serves_this_workers_registry_on_its_port(tmp_path):
    registry = CollectorRegistry()
    Counter("hindsight_test_worker_requests", "test counter", registry=registry).inc(3)
    base = _free_port_block(2)

    port = metrics_worker_endpoint.start(base, 2, lock_dir=str(tmp_path), registry=registry, addr="127.0.0.1")

    assert port == base
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    assert "hindsight_test_worker_requests_total 3.0" in body
    # Idempotent: a second start in the same process keeps the same port instead of claiming another.
    assert metrics_worker_endpoint.start(base, 2, lock_dir=str(tmp_path), registry=registry) == port


def test_a_port_already_in_use_does_not_raise(tmp_path):
    base = _free_port_block(1)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", base))
        busy.listen()
        assert metrics_worker_endpoint.start(base, 1, lock_dir=str(tmp_path), addr="127.0.0.1") is None
    # The slot's lock was released with the failure, so a later attempt can claim it.
    assert metrics_worker_endpoint.claim_slot(base, 1, lock_dir=str(tmp_path)) is not None
