"""Bind ``vector`` values in pgvector's binary format instead of a text literal.

An embedding rendered as ``'[0.1,0.2,...]'`` costs ~12 bytes per dimension on the
wire and a float parse per dimension on the server. The binary format is 4 bytes per
dimension and no parse at all. Measured on Postgres 18 + pgvector, 2000 rows x 384d,
with the statement shape held identical (COPY into a staging table, then
``INSERT ... SELECT ... RETURNING``):

    COPY text literal -> ::vector    471 ms
    COPY binary vector                22 ms      21.7x

Against what shipped before (a single ``INSERT ... SELECT unnest($n::vector[])`` of
text literals, 242 ms) the end-to-end retain write is ~6.5x faster at that size, and
~1.7x at 50 rows.

**The encoder is deliberately tolerant.** ``set_type_codec`` replaces the codec for
the whole connection, so once this is registered a caller that still passes a rendered
literal would otherwise fail with ``DataError`` at runtime. Accepting ``str`` means a
site that was missed keeps working at exactly its old cost rather than raising —
the conversion of call sites is then an optimisation, not a correctness cliff.

Only PostgreSQL uses this. Oracle 23ai has no pgvector wire format and keeps the
text rendering in ``retain.types.embedding_to_pgvector``.
"""

from __future__ import annotations

import logging
from array import array
from struct import pack
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def encode_vector(value: Any) -> bytes | None:
    """Render an embedding in pgvector's binary wire format: ``dim, 0, float32be...``.

    Accepts every form a caller may hold — the packed ``array('f')`` retain carries, a
    plain float list, a NumPy array, or an already-rendered literal (see the module
    docstring for why the literal is still accepted).
    """
    if value is None:
        return None
    if isinstance(value, str):
        # The slow path, kept only so an unconverted call site cannot fail. Same cost
        # as before this codec existed, plus one parse.
        value = _parse_literal(value)
    elif isinstance(value, array):
        value = np.frombuffer(value, dtype=np.float32) if value.typecode == "f" else np.asarray(value)
    values = np.asarray(value, dtype=">f4")
    if values.ndim != 1:
        raise ValueError(f"expected a 1-d embedding, got {values.ndim} dimensions")
    return pack(">HH", values.shape[0], 0) + values.tobytes()


def _parse_literal(literal: str) -> np.ndarray:
    inner = literal.strip()[1:-1]
    if not inner:
        return np.empty(0, dtype=np.float32)
    return np.array(inner.split(","), dtype=np.float32)


def decode_vector(value: bytes) -> np.ndarray:
    """Decode the same wire format back to float32.

    Nothing in the engine currently selects a raw ``vector`` column — the two read
    sites cast to text explicitly (``embedding::text``) — but a codec must be
    symmetric, and a future reader should get an array rather than bytes.
    """
    return np.frombuffer(value, dtype=">f4", count=-1, offset=4).astype(np.float32)


async def register_vector_codec(conn: Any) -> bool:
    """Install the binary codec on one connection. Returns whether it took.

    Returning False rather than raising is what keeps a database without the pgvector
    extension usable. ``MemoryEngine.initialize`` runs migrations before it creates the
    pool, so normally the type exists by the time this runs; the exception is a process
    started with ``run_migrations=False`` against a database that has not been migrated
    yet. Those connections keep binding text literals, which the tolerant encoder still
    accepts, so the failure mode is "no speedup", not "cannot connect".
    """
    try:
        await conn.set_type_codec(
            "vector",
            schema="public",
            encoder=encode_vector,
            decoder=decode_vector,
            format="binary",
        )
    except Exception as exc:  # noqa: BLE001 - any failure here must degrade, not crash
        logger.debug("pgvector binary codec not registered (%s); binding text literals", exc)
        return False
    return True
