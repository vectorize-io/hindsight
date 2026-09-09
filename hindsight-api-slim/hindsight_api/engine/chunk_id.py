"""Construction and parsing for self-describing chunk identifiers."""

from dataclasses import dataclass

_VERSION_PREFIX = "v2:"


@dataclass(frozen=True)
class ChunkAddress:
    bank_id: str
    document_id: str
    chunk_index: int


def build_chunk_id(bank_id: str, document_id: str, chunk_index: int) -> str:
    """Build an unambiguous identifier for a chunk within a bank and document."""
    return f"{_VERSION_PREFIX}{len(bank_id)}:{len(document_id)}:{bank_id}{document_id}_{chunk_index}"


def parse_chunk_id(chunk_id: str | None) -> ChunkAddress | None:
    """Parse current length-prefixed IDs and legacy underscore-delimited IDs."""
    if not chunk_id:
        return None

    if chunk_id.startswith(_VERSION_PREFIX):
        try:
            bank_length_text, document_length_text, payload = chunk_id[len(_VERSION_PREFIX) :].split(":", 2)
            bank_length = int(bank_length_text)
            document_length = int(document_length_text)
            components, separator, chunk_index_text = payload.rpartition("_")
            if bank_length >= 0 and document_length >= 0:
                if separator and len(components) == bank_length + document_length:
                    return ChunkAddress(
                        bank_id=components[:bank_length],
                        document_id=components[bank_length:],
                        chunk_index=int(chunk_index_text),
                    )
        except ValueError:
            pass

    head, separator, chunk_index_text = chunk_id.rpartition("_")
    if not head or not separator or not chunk_index_text:
        return None
    bank_id, separator, document_id = head.rpartition("_")
    if not bank_id or not separator or not document_id:
        return None
    try:
        return ChunkAddress(bank_id=bank_id, document_id=document_id, chunk_index=int(chunk_index_text))
    except ValueError:
        return None
