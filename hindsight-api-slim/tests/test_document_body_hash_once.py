"""An oversized document's body is screened and hashed once, not once per slice (#3756).

The splitter hands every slice of an oversized item the same full
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
import tracemalloc

import pytest

from hindsight_api.config import HindsightConfig, _get_raw_config
from hindsight_api.engine.memory_engine import _screen_document_body
from hindsight_api.engine.retain.fact_extraction import (
    _HASH_WINDOW_CHARS,
    _sanitize_text,
    derive_document_content_hash,
)
from tests.sub_batch_helpers import collect_screened_bodies, collect_sub_batches


def _allocated_mb(fn) -> float:
    """Peak Python bytes ``fn`` allocates, in MB.

    ``tracemalloc``, not RSS. RSS cannot attribute an allocation to the code that made it:
    the allocator maps arenas on first touch and reuses them silently, so the same call reads
    as +400 MB or as +0 MB depending only on what ran before it. That noise is what made
    #3756's original diagnosis wrong, and an RSS-based test of it flaky.
    """
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024 / 1024


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
    subs = collect_sub_batches(contents, 200, chunk_size=500, structured_chunk_size=None)
    # The premise of the test: this body really does slice into many sub-batches.
    assert len(subs) > 10
    assert all(sub.body_override == _BODY for sub in subs)

    screened = collect_screened_bodies(contents, 200, chunk_size=500, structured_chunk_size=None, config=_config())

    assert len(screened) == len(subs)
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


# ---------------------------------------------------------------------------
# The derivation itself, at the sizes the windowing exists for. Everything above runs on a
# body far smaller than one window, so it never crosses a boundary.
#
# Hashing is the last windowed pass over a document body. Token counting was the other, and
# #3788 removed its windowing outright — quicktok's ``count()`` is allocation-free and exact,
# so the workaround stopped earning its keep and ``test_token_counting_windowed.py`` went with
# it. Sanitizing cannot be delegated the same way, so this bound is still hand-rolled and
# still needs its own coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("", id="empty"),
        pytest.param(_BODY, id="under-one-window"),
        # A control character every other char, so the sanitizer deletes on every window and
        # the deletions do not line up with the boundaries.
        pytest.param("a\x00" * (_HASH_WINDOW_CHARS + 5), id="deletions-across-a-boundary"),
        # Multi-byte text sized to put boundaries inside the non-ASCII run.
        pytest.param(
            "Grace \x07évalua. 火曜日に出荷。\U0001f6a2 " * 60_000,
            id="multi-window-non-ascii",
        ),
    ],
)
def test_derived_hash_equals_sanitizing_the_whole_body_first(body):
    """Windowing the sanitize is exact, not approximate.

    ``derive_document_content_hash`` sanitizes and hashes a window at a time so a 45 MB body
    costs a window rather than two more copies of itself (#3756). That is only sound because
    ``_sanitize_text`` deletes a fixed character class with no cross-character context, so
    per-window sanitizing produces the same bytes as sanitizing the whole string. If that ever
    stops holding, this value silently stops matching the rows already written — it is the
    document's identity, so nothing would raise.
    """
    assert derive_document_content_hash(body) == _hash_the_way_retain_would(body)


def test_derived_hash_is_idempotent_on_already_sanitized_text():
    """``handle_document_tracking`` sanitizes for storage, then hashes the sanitized text.

    It gets the same digest as the streaming path, which derives straight off the raw body,
    and the two are compared against each other as an ownership check.
    """
    body = _BODY + "\x00\x07 trailing control characters \x1b"
    sanitized = _sanitize_text(body) or ""

    assert derive_document_content_hash(sanitized) == derive_document_content_hash(body)


def test_deriving_the_hash_does_not_allocate_a_copy_of_the_body():
    """The point of the windowing, and the thing an inlined rewrite would quietly undo.

    Sanitizing the whole body first and then encoding it costs two more full copies of a
    45 MB document to produce 64 hex characters. Bounded by the window instead, so an 8x
    larger body costs the same.

    Both inputs have to span more than one window for that comparison to mean anything. A
    body smaller than a window is its own bound, so it costs proportionally less and the
    test would fail on correct code.
    """
    small = _BODY * 24
    large = _BODY * 192
    assert len(small) > _HASH_WINDOW_CHARS, "the small input must still exceed one window"
    assert len(large) > 8 * _HASH_WINDOW_CHARS, "the large input must span many windows"

    small_cost = _allocated_mb(lambda: derive_document_content_hash(small))
    large_cost = _allocated_mb(lambda: derive_document_content_hash(large))

    assert large_cost <= small_cost * 1.5, f"small={small_cost:.1f} MB, large={large_cost:.1f} MB"


def test_body_is_screened_once_when_other_items_share_the_batch():
    """The one-entry cache must still hit when non-slice sub-batches surround the slices.

    Slices of one document are yielded consecutively, but a sub-batch of packed small items
    carries no body and can land either side of them. That sub-batch is what releases the
    previous document's screened copy, so it must not also evict the entry the following
    slices depend on.
    """
    contents = [
        {"content": "a small first item", "document_id": "small-a"},
        {"content": _BODY, "document_id": "sliced"},
        {"content": "a small last item", "document_id": "small-b"},
    ]

    screened = collect_screened_bodies(contents, 200, chunk_size=500, structured_chunk_size=None, config=_config())

    slices = [entry for entry in screened if entry is not None]
    assert len(slices) > 10, "the middle item should have sliced many times"
    # One redaction + hash for the document, however many slices it produced and whatever
    # else shares the batch — a per-slice implementation yields equal-but-distinct objects.
    assert len({id(entry) for entry in slices}) == 1
