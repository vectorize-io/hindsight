"""Unit tests for chunk id construction/parsing (issue #4244).

The id flattens ``(bank_id, document_id, chunk_index)`` into the ``chunks`` primary
key, so the mapping must be injective: bank ids and document ids are arbitrary
caller-supplied strings, and two triples that flatten to the same string are two banks
writing the same row.
"""

import pytest

from hindsight_api.engine.chunk_ids import (
    build_chunk_id,
    chunk_index_in,
    document_and_index_in,
    parse_chunk_id,
)


def test_underscore_bank_and_document_do_not_collide():
    """The exact pair from #4244: ``a`` + ``b_c`` and ``a_b`` + ``c``."""
    assert build_chunk_id("a", "b_c", 0) != build_chunk_id("a_b", "c", 0)


@pytest.mark.parametrize(
    ("bank_id", "document_id"),
    [
        ("bank", "doc"),
        ("a", "b_c"),
        ("a_b", "c"),
        ("a_b_c", "d"),
        ("tilde~bank", "doc"),
        ("bank", "doc~5F"),
        ("_", "_"),
        ("bank_1", "session_2_3"),
    ],
)
def test_roundtrip(bank_id, document_id):
    assert parse_chunk_id(build_chunk_id(bank_id, document_id, 7)) == (bank_id, document_id, 7)


def test_ids_are_unique_across_a_grid_of_separator_heavy_pairs():
    parts = ["a", "b", "a_b", "_", "a~b", "~5F"]
    ids = {build_chunk_id(bank, doc, 0) for bank in parts for doc in parts}
    assert len(ids) == len(parts) ** 2


def test_format_unchanged_for_separator_free_ids():
    """Ids already in the database keep working: only components carrying a separator move."""
    assert build_chunk_id("bank123", "doc-abc", 4) == "bank123_doc-abc_4"


def test_legacy_ids_still_parse():
    """Pre-fix ids carry no escaping; the unambiguous ones must still resolve."""
    assert parse_chunk_id("bank123_doc-abc_4") == ("bank123", "doc-abc", 4)
    # Ambiguous legacy shape — resolved the historical way, splitting from the right.
    assert parse_chunk_id("a_b_c_0") == ("a_b", "c", 0)


@pytest.mark.parametrize("value", [None, "", "no-separators", "bank_doc_notanint"])
def test_unparseable_ids_return_none(value):
    assert parse_chunk_id(value) is None


@pytest.mark.parametrize(
    ("bank_id", "document_id"),
    [("bank", "doc"), ("a", "b_c"), ("a_b", "c"), ("bank~1", "doc_2")],
)
def test_chunk_index_in_is_exact_for_the_owning_pair(bank_id, document_id):
    chunk_id = build_chunk_id(bank_id, document_id, 12)
    assert chunk_index_in(chunk_id, bank_id, document_id) == 12
    assert document_and_index_in(chunk_id, bank_id) == (document_id, 12)


def test_anchored_helpers_reject_another_pairs_id():
    chunk_id = build_chunk_id("a", "b_c", 0)
    assert chunk_index_in(chunk_id, "a_b", "c") is None
    assert chunk_index_in(chunk_id, "a", "other") is None


def test_anchored_helpers_still_read_legacy_ids():
    assert chunk_index_in("bank_doc_3", "bank", "doc") == 3
    assert document_and_index_in("bank_doc_3", "bank") == ("doc", 3)
    # A legacy id whose document carries a separator: anchoring on the known pair recovers it
    # exactly, where a bare parse could not.
    assert chunk_index_in("bank_a_b_5", "bank", "a_b") == 5
    assert document_and_index_in("bank_a_b_5", "bank") == ("a_b", 5)
