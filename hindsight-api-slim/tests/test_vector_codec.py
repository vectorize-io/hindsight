"""The pgvector binary codec must accept every form a call site may hold.

``set_type_codec`` replaces the codec for the whole connection, so once the binary
codec is registered a caller still passing a rendered ``'[0.1,...]'`` literal would
fail with ``DataError`` at runtime unless the encoder accepts it. These tests pin
that tolerance — it is what makes the call-site conversion an optimisation rather
than a correctness cliff.
"""

from array import array

import numpy as np
import pytest

from hindsight_api.engine.db.vector_codec import decode_vector, encode_vector
from hindsight_api.engine.retain.types import embedding_to_pgvector

VALUES = [0.1, -2.5, 3.0, 4.25, 0.0]


def _forms(values: list[float]) -> dict[str, object]:
    """Every representation an embedding reaches the database as."""
    return {
        "list": list(values),
        "ndarray-f32": np.asarray(values, dtype=np.float32),
        "ndarray-f64": np.asarray(values, dtype=np.float64),
        "array-f": array("f", values),
        "literal": embedding_to_pgvector(np.asarray(values, dtype=np.float32)),
    }


def test_every_input_form_encodes_to_the_same_bytes():
    encoded = {name: encode_vector(v) for name, v in _forms(VALUES).items()}
    assert len(set(encoded.values())) == 1, f"forms disagree: { {k: len(v) for k, v in encoded.items()} }"


def test_wire_format_is_dim_then_float32_big_endian():
    """4-byte header (dim, unused) + 4 bytes per dimension — pgvector's binary layout."""
    raw = encode_vector(VALUES)
    assert len(raw) == 4 + 4 * len(VALUES)
    assert raw[:4] == bytes([0, len(VALUES), 0, 0])


def test_round_trips_through_decode():
    assert np.allclose(decode_vector(encode_vector(VALUES)), np.asarray(VALUES, dtype=np.float32))


def test_binary_is_smaller_than_the_literal_it_replaces():
    """The reason this exists: ~4 bytes per dimension instead of ~12."""
    values = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    assert len(encode_vector(values)) * 3 < len(embedding_to_pgvector(values))


def test_none_passes_through():
    """A NULL embedding must stay NULL rather than become an empty vector."""
    assert encode_vector(None) is None


def test_empty_literal_is_an_empty_vector():
    assert len(encode_vector("[]")) == 4


def test_rejects_a_2d_array():
    """An array of vectors cannot be bound: asyncpg hands the whole list to this
    encoder rather than each element, so failing loudly beats silently storing the
    first row. See the module docstring on `unnest($n::vector[])`."""
    with pytest.raises(ValueError, match="1-d"):
        encode_vector(np.zeros((2, 4), dtype=np.float32))


def test_float64_is_narrowed_the_same_way_the_column_stores_it():
    """float32 is the column's width; a value must land on the same float32 either way."""
    values = np.random.default_rng(1).standard_normal(64)
    assert encode_vector(values) == encode_vector(values.astype(np.float32))
