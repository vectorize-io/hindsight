"""Construction and parsing of chunk ids.

A chunk id is the ``(bank_id, document_id, chunk_index)`` triple flattened into one
string, because ``chunks.chunk_id`` is a globally unique primary key that
``memory_units.chunk_id`` references and that the addressed chunk route carries in its
path. Retain rebuilds the id from the triple rather than reading it back, so the
mapping has to be a pure function -- and, since ``chunks`` is keyed on the id alone, an
*injective* one: two different triples that flatten to the same string are two banks
writing the same row.

The naive ``f"{bank_id}_{document_id}_{index}"`` was not injective. Bank ids and
document ids are arbitrary caller-supplied strings, so ``("a", "b_c")`` and
``("a_b", "c")`` both produced ``a_b_c_0`` and the second bank's retain overwrote the
first bank's chunk text (issue #4244). Escaping the separator inside each component
makes the join unambiguous: the id now always splits into exactly three fields.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRef:
    """The ``(bank_id, document_id, chunk_index)`` triple a chunk id encodes."""

    bank_id: str
    document_id: str
    chunk_index: int


# `~` escapes itself and `_`, and is URL-unreserved so an escaped id still travels in
# the `/v1/default/chunks/{chunk_id}` path without further encoding.
_ESCAPE = "~"
_ESCAPED = {"~": "~7E", "_": "~5F"}
_UNESCAPED = {v: k for k, v in _ESCAPED.items()}


def _escape(value: str) -> str:
    return "".join(_ESCAPED.get(ch, ch) for ch in value)


def _unescape(value: str) -> str:
    if _ESCAPE not in value:
        return value
    out: list[str] = []
    i = 0
    while i < len(value):
        token = value[i : i + 3]
        if token in _UNESCAPED:
            out.append(_UNESCAPED[token])
            i += 3
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def build_chunk_id(bank_id: str, document_id: str, chunk_index: int) -> str:
    """The chunk id for this triple. Injective: distinct triples give distinct ids.

    Ids are unchanged from the pre-#4244 format for the common case where neither the
    bank id nor the document id contains ``_`` or ``~``, so existing rows keep working;
    the ones that do contain a separator get a new id the next time their chunk is
    written, which is exactly the case that used to collide.
    """
    return f"{_escape(bank_id)}_{_escape(document_id)}_{chunk_index}"


def parse_chunk_id(chunk_id: str | None) -> ChunkRef | None:
    """Split a chunk id back into ``(bank_id, document_id, chunk_index)``.

    Returns ``None`` when the id is not in that shape, so the caller falls through to a
    SQL lookup rather than guessing.

    Ids written before #4244 do not carry the escaping, and one whose bank or document
    id contains ``_`` is genuinely ambiguous -- there is no recovering which underscore
    was the separator. Those fall back to the historical heuristic: split from the
    RIGHT, which is right whenever the document id has no underscore.
    """
    if not chunk_id:
        return None
    parts = chunk_id.split("_")
    if len(parts) == 3:
        bank_id, document_id, idx_s = parts
    else:
        # Legacy, ambiguous id. Both splits are from the right: the index is the final
        # segment, and a document id with no underscore sits just before it.
        head, _, idx_s = chunk_id.rpartition("_")
        bank_id, _, document_id = head.rpartition("_")
    if not bank_id or not document_id or not idx_s:
        return None
    try:
        index = int(idx_s)
    except ValueError:
        return None
    return ChunkRef(_unescape(bank_id), _unescape(document_id), index)


def resolve_chunk_id_in(chunk_id: str, bank_id: str) -> ChunkRef | None:
    """Split ``chunk_id`` into its triple given the bank that owns it.

    For a bank whose documents live outside SQL there is no ``chunks`` row to read the
    document and index off, and the id is the only thing carrying them. Knowing the bank
    removes the guesswork the bare :func:`parse_chunk_id` has to do about where the bank id
    ends, so this is exact even for the ambiguous ids written before #4244 -- hence the
    legacy, unescaped prefix is tried too (the two coincide when the bank id carries no
    separator).
    """
    for prefix, escaped in ((f"{_escape(bank_id)}_", True), (f"{bank_id}_", False)):
        rest = chunk_id.removeprefix(prefix)
        if rest == chunk_id:
            continue
        document_id, _, idx_s = rest.rpartition("_")
        if not document_id or not idx_s.isdigit():
            continue
        return ChunkRef(bank_id, _unescape(document_id) if escaped else document_id, int(idx_s))
    return None
