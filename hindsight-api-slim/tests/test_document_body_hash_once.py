"""An oversized document's body is screened and hashed once, not once per slice (#3756).

``collect_sub_batches`` hands every slice of an oversized item the same full
body to write as ``documents.original_text``. The retain path then sanitized and SHA-256'd
that body to derive the document's ``content_hash`` — on every slice. A 45 MB body splits
into ~1,200 slices at the default budget and costs ~0.9s per hash, so ~18 minutes went into
re-deriving one value.

The hash now travels with the screened body. Two things have to stay true for that to be
safe, and both are asserted here: the precomputed value must equal what the retain path
would have computed (it is the document's identity — ownership checks, recovery detection
and delta all compare against it), and the work must actually happen once.
"""

import dataclasses
import hashlib

from hindsight_api.config import HindsightConfig, _get_raw_config
from hindsight_api.engine.memory_engine import _screen_document_body
from hindsight_api.engine.retain.fact_extraction import _sanitize_text
from tests.sub_batch_helpers import collect_screened_bodies, collect_sub_batches

_BODY = (
    "Ada shipped the parser on Tuesday. Grace reviewed it on Wednesday. "
    "The build went green on Thursday, and nobody had to roll it back.\n\n"
) * 400


def _config(memory_defense: dict | None = None) -> HindsightConfig:
    """A real resolved config. ``memory_defense`` is bank-configurable, so the global
    config refuses to serve it — screening only ever runs against a resolved one."""
    return dataclasses.replace(_get_raw_config(), memory_defense=memory_defense)


def _hash_the_way_retain_would(body: str) -> str:
    """Exactly what ``_streaming_retain_batch`` computed before the hash was passed in."""
    return hashlib.sha256((_sanitize_text(body) or "").encode()).hexdigest()


def test_screened_body_hash_matches_what_retain_would_have_computed():
    """The precomputed hash is the same value, not merely a plausible one."""
    screened = _screen_document_body(_BODY, _config())

    assert screened.content_hash == _hash_the_way_retain_would(screened.text)


def test_sub_batches_that_are_not_slices_carry_no_body_override():
    """Only a slice of an oversized item needs the full body written back.

    A packed sub-batch holds its own items whole, so ``documents.original_text`` comes from
    the content itself and there is nothing to override.
    """
    # The budget has to sit between one item and two: above it, so neither item is
    # oversized and sliced (that branch DOES carry an override); below their sum, so they
    # pack into separate sub-batches rather than one. Each of these is 4 tokens.
    small = "Ada shipped it."
    screened = collect_screened_bodies(
        [{"content": small, "document_id": "a"}, {"content": small, "document_id": "b"}],
        6,
        chunk_size=500,
        structured_chunk_size=None,
        config=_config(),
    )

    assert len(screened) > 1, f"the batch did not pack into several sub-batches: {screened}"
    assert all(entry is None for entry in screened)


def test_body_is_screened_and_hashed_once_across_every_slice():
    """One slice's worth of work, however many slices the document produced."""
    contents = [{"content": _BODY, "document_id": "doc-sliced"}]
    split = collect_sub_batches(contents, 200, chunk_size=500, structured_chunk_size=None)
    # The premise of the test: this body really does slice into many sub-batches.
    assert len(split.sub_batches) > 10
    assert all(body == _BODY for body in split.document_body_overrides)

    screened = collect_screened_bodies(contents, 200, chunk_size=500, structured_chunk_size=None, config=_config())

    assert len(screened) == len(split.sub_batches)
    # Every slice gets the identical object, so the redaction ran once and the hash with
    # it — a per-slice implementation would produce equal-but-distinct instances.
    first = screened[0]
    assert first is not None
    assert all(entry is first for entry in screened)


def test_screening_is_per_distinct_body():
    """Two oversized documents in one batch each get their own screened body and hash."""
    other = _BODY.replace("Ada", "Alan")
    screened = collect_screened_bodies(
        [{"content": _BODY, "document_id": "a"}, {"content": other, "document_id": "b"}],
        200,
        chunk_size=500,
        structured_chunk_size=None,
        config=_config(),
    )

    bodies = [entry for entry in screened if entry is not None]
    assert bodies, "neither document was sliced, so nothing was screened"
    distinct = {id(entry) for entry in bodies}
    # Two documents, so exactly two screened instances however many slices each produced.
    assert len(distinct) == 2
    hashes = {entry.content_hash for entry in bodies}
    assert len(hashes) == 2


def test_memory_defense_redaction_is_reflected_in_the_hash():
    """When screening rewrites the body, the hash is of the rewritten text.

    The document row stores the redacted body, so a hash taken before redaction would
    describe a document that was never written — and every ownership check against it
    would then miss.
    """
    config = _config({"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]})
    body = _BODY + "\nThe API key is sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKK.\n"

    screened = _screen_document_body(body, config)

    assert screened.text != body, "the policy should have redacted the secret"
    assert screened.content_hash == _hash_the_way_retain_would(screened.text)
