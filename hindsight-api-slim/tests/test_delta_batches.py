"""Pure helpers behind the batched delta retain (#6394): how a delta's chunks are split into the
batches it commits one by one, and how a batch notices another writer's chunks."""

from hindsight_api.engine.retain.orchestrator import _chunks_moved_since_plan, _delta_batches


def test_delta_batches_bound_every_batch():
    assert _delta_batches(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert _delta_batches([4, 9], 3) == [[4, 9]]


def test_delta_batches_always_yield_one_batch():
    # A delta that only removes chunks has nothing to extract but still needs its final write.
    assert _delta_batches([], 3) == [[]]
    # 0 disables batching, as it does for the streaming path.
    assert _delta_batches(list(range(5)), 0) == [list(range(5))]


def test_chunks_moved_accepts_the_planned_and_the_written_chunk():
    planned: dict[int, str | None] = {0: "a", 1: "b"}
    new = {0: "a", 1: "B", 2: "C"}
    assert _chunks_moved_since_plan({0: "a", 1: "b"}, planned, new) == []  # nothing written yet
    assert _chunks_moved_since_plan({0: "a", 1: "B", 2: "C"}, planned, new) == []  # earlier batches


def test_chunks_moved_flags_another_writers_chunks():
    planned: dict[int, str | None] = {0: "a", 1: "b"}
    new = {0: "a", 1: "B", 2: "C"}
    assert _chunks_moved_since_plan({0: "a", 1: "b", 2: "X"}, planned, new) == [2]  # added elsewhere
    assert _chunks_moved_since_plan({0: "Y", 1: "b"}, planned, new) == [0]  # replaced an unchanged one
    assert _chunks_moved_since_plan({1: "b"}, planned, new) == [0]  # deleted one
