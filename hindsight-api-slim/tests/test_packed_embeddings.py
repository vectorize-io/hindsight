"""Retain carries embeddings packed, and packing changes nothing that reaches the DB (#3756).

A 384-dim embedding costs 12,344 bytes as ``list[float]`` — every element a separately
boxed ``PyFloat`` — against 1,616 bytes as ``array("f")``. Retain holds a batch of facts at
a time, so that 7.6x rides on every fact in flight.

The width is not a compromise: pgvector's ``vector`` column is float32, so the rounding
that used to happen at the INSERT now happens one step earlier and the stored bytes are
unchanged. These tests hold that line — the packed vector must render to a literal
Postgres accepts, and to the same value the unpacked one would have produced.
"""

import math
from array import array

import numpy as np

from hindsight_api.engine.retain.types import (
    ExtractedFact,
    ProcessedFact,
    embedding_to_pgvector,
    pack_embedding,
)

_VECTOR = [0.1, -0.25, 3.5, 0.0, -1.0, 1e-7, 123.456]


def test_packing_is_smaller_than_a_float_list():
    """The whole point, asserted rather than assumed."""
    vector = [0.1234567] * 384
    boxed = 384 * 24  # a PyFloat is 24 bytes on CPython 3.11+, before the list's pointers
    packed = pack_embedding(vector)

    assert packed.itemsize == 4
    assert packed.buffer_info()[1] * packed.itemsize == 384 * 4
    assert 384 * 4 < boxed


def test_pgvector_literal_round_trips_through_float32():
    """The literal parses back to the value pgvector would have stored either way."""
    from_packed = embedding_to_pgvector(pack_embedding(_VECTOR))
    parsed = [float(part) for part in from_packed.strip("[]").split(",")]

    assert from_packed.startswith("[") and from_packed.endswith("]")
    assert len(parsed) == len(_VECTOR)
    # float32 is what the column holds, so the comparison is against the float32 image of
    # the original — not against the float64 the embedding model handed us.
    assert parsed == list(array("f", _VECTOR))


def test_pgvector_literal_accepts_every_form_a_caller_may_hold():
    """Packed, plain list, or an already-rendered literal — one helper covers all three."""
    packed = embedding_to_pgvector(pack_embedding(_VECTOR))

    assert embedding_to_pgvector(list(array("f", _VECTOR))) == packed
    assert embedding_to_pgvector(packed) == packed


def test_processed_fact_packs_what_it_is_given():
    """``from_extracted_fact`` is where the model's float list becomes a packed vector."""
    fact = ProcessedFact.from_extracted_fact(
        ExtractedFact(fact_text="Ada shipped the parser", fact_type="world"),
        _VECTOR,
    )

    assert fact is not None
    assert isinstance(fact.embedding, array)
    assert fact.embedding.typecode == "f"
    assert list(fact.embedding) == list(array("f", _VECTOR))


def test_packed_vectors_still_work_as_a_numpy_matrix():
    """In-batch semantic linking builds a matrix straight from the carried vectors.

    ``np.asarray`` reads the packed form through the buffer protocol, so the similarity
    maths that decides which facts get linked keeps working on it unchanged.
    """
    vectors = [pack_embedding([1.0, 0.0, 0.0]), pack_embedding([0.0, 1.0, 0.0])]

    matrix = np.asarray(vectors, dtype=float)

    assert matrix.shape == (2, 3)
    assert math.isclose(float(np.dot(matrix[0], matrix[1])), 0.0)
    assert math.isclose(float(np.linalg.norm(matrix[0])), 1.0)


def test_non_finite_values_survive_the_round_trip():
    """A degenerate embedding stays degenerate rather than becoming an unparseable literal.

    Semantic linking screens non-finite vectors out explicitly; it can only do that if they
    arrive intact instead of having been mangled on the way in.
    """
    packed = pack_embedding([float("nan"), float("inf"), -float("inf"), 1.0])

    assert math.isnan(packed[0])
    assert math.isinf(packed[1]) and packed[1] > 0
    assert math.isinf(packed[2]) and packed[2] < 0
    assert not np.isfinite(np.asarray([packed], dtype=float)).all()
