"""Which attachment did *this* fact come from?

One extraction call sees one chunk, and a chunk that carries a screenshot also
carries the prose around it. Attributing the chunk's attachments to every fact it
produced puts the architecture diagram behind the sentence about paging policy —
the reader is then shown a picture as the evidence for something it does not
show, which is worse than showing nothing.

So the extractor is asked. The prompt numbers the attachments 1..n in the order
they appear in the chunk and each fact answers with `from_attachments`; those
numbers are resolved against that same chunk and stored per unit.

The two halves are tested apart, because they fail apart: the resolution is pure
and deterministic and is asserted directly, while whether the model attributes
*correctly* is a question about the model and goes to the judge.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import (
    attachment_placeholder,
    compute_attachment_hash,
    short_attachment_id,
)
from hindsight_api.engine.retain.fact_extraction import Fact, _attachment_ids_for

FIRST = short_attachment_id("a" * 64)
SECOND = short_attachment_id("b" * 64)
CHUNK = f"before {attachment_placeholder('a' * 64)} between {attachment_placeholder('b' * 64)} after"


def _fact(numbers: list[int] | None) -> Fact:
    return Fact(fact="x", fact_type="world", from_attachments=numbers)


def test_numbers_resolve_to_the_chunks_attachments_in_order():
    assert _attachment_ids_for(_fact([1]), CHUNK) == [FIRST]
    assert _attachment_ids_for(_fact([2]), CHUNK) == [SECOND]
    assert _attachment_ids_for(_fact([2, 1]), CHUNK) == [SECOND, FIRST]


def test_a_fact_the_model_attributed_to_nothing_carries_nothing():
    """The common case: a fact stated in the prose, next to an unrelated image."""
    assert _attachment_ids_for(_fact(None), CHUNK) == []
    assert _attachment_ids_for(_fact([]), CHUNK) == []


def test_a_number_outside_the_chunks_range_is_dropped_rather_than_guessed():
    # A model that answers "3" for a two-attachment chunk has told us nothing
    # usable; attaching the nearest one would be inventing provenance.
    assert _attachment_ids_for(_fact([3]), CHUNK) == []
    assert _attachment_ids_for(_fact([0]), CHUNK) == []
    assert _attachment_ids_for(_fact([-1]), CHUNK) == []
    assert _attachment_ids_for(_fact([1, 9]), CHUNK) == [FIRST]


def test_the_same_attachment_named_twice_yields_one_id():
    assert _attachment_ids_for(_fact([1, 1]), CHUNK) == [FIRST]


def test_numbering_counts_occurrences_because_that_is_what_the_model_was_shown():
    """One part per placeholder reaches the model, repeats included.

    Numbering distinct attachments instead would shift every number after a
    repeat, so a fact attributed to the *third* picture would silently resolve to
    the second one.
    """
    chunk = (
        f"{attachment_placeholder('a' * 64)} again {attachment_placeholder('a' * 64)} "
        f"then {attachment_placeholder('b' * 64)}"
    )
    assert _attachment_ids_for(_fact([1]), chunk) == [FIRST]
    assert _attachment_ids_for(_fact([2]), chunk) == [FIRST]
    assert _attachment_ids_for(_fact([3]), chunk) == [SECOND]
    # Both occurrences of the same attachment are still one edge.
    assert _attachment_ids_for(_fact([1, 2]), chunk) == [FIRST]


def test_a_text_only_chunk_resolves_to_nothing():
    assert _attachment_ids_for(_fact([1]), "no attachments here") == []
