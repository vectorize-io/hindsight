"""Which retain-write gate a bank gets depends on who owns its write path.

`retain_max_concurrent` is sized for contention on the SQL entity/link/HNSW tables — concurrent
index work on shared tables. A store that owns the write path keeps its own index and has none of
that contention, so the SQL number is a throughput ceiling there for a reason that does not apply.

It still gets *a* gate rather than none: the phase holds a decoded batch for its duration, so
unbounded concurrency would be unbounded memory.
"""

from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import get_config


@pytest.fixture
def engine():
    """A MemoryEngine with only the two semaphores initialised.

    Built without `__init__` because constructing a real engine needs a database, an embeddings
    model and an LLM config — none of which this decision consults.
    """
    import asyncio

    from hindsight_api.engine.memory_engine import MemoryEngine

    e = MemoryEngine.__new__(MemoryEngine)
    e._put_semaphore = asyncio.Semaphore(get_config().retain_max_concurrent)
    e._store_put_semaphore = asyncio.Semaphore(get_config().retain_store_max_concurrent)
    return e


def _store(*, owns: bool):
    s = MagicMock()
    s.store_owned_for.return_value = owns
    return s


def test_a_sql_backed_bank_gets_the_sql_gate(engine):
    with patch("hindsight_api.engine.memories.get_memories", return_value=_store(owns=False)):
        assert engine._db_semaphore_for("bank") is engine._put_semaphore


def test_a_store_owned_bank_gets_the_store_gate(engine):
    with patch("hindsight_api.engine.memories.get_memories", return_value=_store(owns=True)):
        assert engine._db_semaphore_for("bank") is engine._store_put_semaphore


def test_a_store_that_cannot_answer_falls_back_to_the_tighter_gate(engine):
    """Resolving ownership must never be what fails a retain, and must fail safe."""
    broken = MagicMock()
    broken.store_owned_for.side_effect = RuntimeError("store unavailable")
    with patch("hindsight_api.engine.memories.get_memories", return_value=broken):
        assert engine._db_semaphore_for("bank") is engine._put_semaphore


def test_the_two_gates_are_distinct_and_the_store_one_is_not_tighter(engine):
    """The point of the split: the store path is not held to the SQL path's contention limit."""
    assert engine._put_semaphore is not engine._store_put_semaphore
    assert get_config().retain_store_max_concurrent >= get_config().retain_max_concurrent
