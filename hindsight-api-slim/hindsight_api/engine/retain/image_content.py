"""Canonicalization of multimodal retain content into placeholder text.

A retain item's ``content`` may be an ordered list of text and image blocks. The
rest of the pipeline — ``documents.original_text``, the ``content_hash``
idempotency gate, ``update_mode="append"``, chunk-delta re-extraction,
``reprocess_document`` rebuilding from ``retain_params``, export/import — is built
on the content being *one string*. Rather than thread a second shape through all
of it, the API boundary flattens the blocks into a single canonical body in which
each image is represented by an atomic placeholder::

    To reset the VPN, click the button shown:

    ⟦hs-image:sha256:3f9a...⟧

    ...then reconnect.

The bytes live in file storage, content-addressed by the same hash. They are
resolved back into real image parts only when the extraction prompt is assembled
(see ``fact_extraction``), so the image is seen by the model *in position*,
alongside the prose that refers to it.

Everything here is pure: no I/O, no database, no config. That keeps the mapping
from blocks to canonical text directly testable, which matters because the
mapping must be perfectly deterministic — the same blocks must always produce the
same body, or the ``content_hash`` gate would re-extract an unchanged document.
"""

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

# Delimiters chosen from the Unicode mathematical-brackets block: they survive
# `sanitize_text` (which only strips control characters and surrogates), they are
# not separators the recursive chunker splits on, and they are rare enough in
# real prose that a placeholder is unlikely to collide with authored text.
# A caller cannot forge one regardless -- see `neutralize_placeholders`.
PLACEHOLDER_OPEN = "⟦"
PLACEHOLDER_CLOSE = "⟧"
_PLACEHOLDER_BODY = "hs-image:sha256:"

#: Matches a well-formed image placeholder and captures its hash.
PLACEHOLDER_RE = re.compile(
    re.escape(PLACEHOLDER_OPEN)
    + re.escape(_PLACEHOLDER_BODY)
    + r"(?P<hash>[0-9a-f]{64})"
    + re.escape(PLACEHOLDER_CLOSE)
)

#: Matches anything *shaped* like a placeholder, well-formed or not. Used to scrub
#: caller-supplied text so authored content can never impersonate a real image
#: reference (which would otherwise let one document cite another's image).
_PLACEHOLDER_LOOKALIKE_RE = re.compile(
    re.escape(PLACEHOLDER_OPEN)
    + re.escape(_PLACEHOLDER_BODY)
    + r"[^"
    + re.escape(PLACEHOLDER_CLOSE)
    + r"]*"
    + re.escape(PLACEHOLDER_CLOSE)
)


def image_placeholder(image_hash: str) -> str:
    """Render the atomic placeholder token standing in for ``image_hash``."""
    return f"{PLACEHOLDER_OPEN}{_PLACEHOLDER_BODY}{image_hash}{PLACEHOLDER_CLOSE}"


def compute_image_hash(data: bytes) -> str:
    """Content-address image bytes. Identical images dedupe on this hash."""
    return hashlib.sha256(data).hexdigest()


def neutralize_placeholders(text: str) -> str:
    """Strip placeholder-shaped substrings from caller-authored text.

    Only the canonicalizer may mint a placeholder. Without this, a caller could
    write the token by hand in a text block and have extraction resolve it to an
    image the document never carried -- including one stored for a different
    document in the same bank.
    """
    return _PLACEHOLDER_LOOKALIKE_RE.sub("", text)


def iter_placeholder_hashes(text: str) -> Iterator[str]:
    """Yield the image hashes referenced by ``text``, in order, with repeats."""
    for match in PLACEHOLDER_RE.finditer(text):
        yield match.group("hash")


def contains_image(text: str) -> bool:
    """Whether ``text`` references at least one image."""
    return PLACEHOLDER_RE.search(text) is not None


@dataclass(frozen=True)
class RetainImage:
    """One decoded image from a multimodal retain item."""

    image_hash: str
    media_type: str
    data: bytes
    #: Index of the block this image came from, kept so the first appearance of an
    #: image in a document can be recorded for provenance.
    block_index: int

    @property
    def byte_size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class CanonicalContent:
    """A multimodal item flattened to text plus the images it references."""

    text: str
    #: Deduplicated by hash, in first-appearance order. The same image used twice
    #: in one document yields two placeholders but one entry here, so it is stored
    #: and recorded once.
    images: tuple[RetainImage, ...]

    @property
    def has_images(self) -> bool:
        return bool(self.images)


@dataclass(frozen=True)
class RetainText:
    """One text block from a multimodal retain item."""

    text: str


#: One element of a multimodal item's content, in the order the caller wrote it.
ContentBlock = RetainText | RetainImage


def _pad_to_blank_line(body: str) -> str:
    """Close ``body`` with a paragraph break, without doubling an existing one.

    Placeholders sit alone on their own paragraph so the recursive chunker's
    most-preferred separator ("\\n\\n") falls either side of one. An image then
    lands at a natural chunk boundary instead of being split away from the
    sentence that introduces it.
    """
    if not body:
        return body
    return f"{body.rstrip(chr(10))}\n\n"


def canonicalize(blocks: Sequence[ContentBlock]) -> CanonicalContent:
    """Flatten ordered text/image blocks into the canonical body plus its images.

    Image blocks must already be decoded and hashed; this decides only where each
    placeholder lands. A single text block canonicalizes to exactly its own text,
    so ``[{"type": "text", "text": X}]`` and the plain string ``X`` produce an
    identical body -- and therefore an identical ``content_hash``.
    """
    body = ""
    images: list[RetainImage] = []
    seen: set[str] = set()
    after_image = False

    for block in blocks:
        if isinstance(block, RetainText):
            if after_image:
                body = _pad_to_blank_line(body)
            body += neutralize_placeholders(block.text)
            after_image = False
        else:
            body = _pad_to_blank_line(body) + image_placeholder(block.image_hash)
            after_image = True
            if block.image_hash not in seen:
                seen.add(block.image_hash)
                images.append(block)

    return CanonicalContent(text=body, images=tuple(images))
