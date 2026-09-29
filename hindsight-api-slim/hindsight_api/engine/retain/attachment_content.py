"""Canonicalization of multimodal retain content into placeholder text.

A retain item's ``content`` may be an ordered list of text and image blocks. The
rest of the pipeline — ``documents.original_text``, the ``content_hash``
idempotency gate, ``update_mode="append"``, chunk-delta re-extraction,
``reprocess_document`` rebuilding from ``retain_params``, export/import — is built
on the content being *one string*. Rather than thread a second shape through all
of it, the API boundary flattens the blocks into a single canonical body in which
each image is represented by an atomic placeholder::

    To reset the VPN, click the button shown:

    ⟦hs-att:c414cd0e204d⟧

    ...then reconnect.

The bytes live in file storage, content-addressed by the full digest the
placeholder's short id prefixes. They are
resolved back into real image parts only when the extraction prompt is assembled
(see ``fact_extraction``), so the image is seen by the model *in position*,
alongside the prose that refers to it.

Everything here is pure: no I/O, no database, no config. That keeps the mapping
from blocks to canonical text directly testable, which matters because the
mapping must be perfectly deterministic — the same blocks must always produce the
same body, or the ``content_hash`` gate would re-extract an unchanged document.
"""

import base64
import binascii
import hashlib
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator

# typing.TypeAliasType is 3.12+; the backport ships with pydantic, which needs it.
from typing_extensions import TypeAliasType

# MIME syntax is the only local format restriction. Provider capabilities evolve
# faster than an allowlist here could, so unsupported media is left for the
# configured model to reject with its own actionable error.
_MEDIA_TYPE_RE = re.compile(r"^[\w.+-]+/[\w.+-]+$")


class Base64AttachmentSource(BaseModel):
    """Inline attachment bytes, base64-encoded.

    The only source type in this version. ``url`` (server-side fetch) and
    ``blob_id`` (pre-uploaded handle) are the natural next ones, which is why this
    is modelled as a discriminated union on ``type`` rather than as bare fields.
    """

    type: Literal["base64"] = "base64"
    media_type: str = Field(
        description=(
            "MIME type of the attachment, e.g. 'image/png' or 'application/pdf'. Any well-formed "
            "type is accepted; whether the model can read it is the model's answer to give, and a "
            "provider that rejects it fails the retain with its own error."
        )
    )
    data: str = Field(description="Base64-encoded bytes (no data: URI prefix).")

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, v: str) -> str:
        if not _MEDIA_TYPE_RE.match(v):
            raise ValueError(f"media_type must look like 'type/subtype', got {v!r}")
        return v

    def decode(self) -> bytes:
        """Decode on use, so model construction does not copy large payloads.

        The API calls this once at its validation boundary, keeping malformed
        base64 a request error without paying the decode cost twice.
        """
        try:
            return base64.b64decode(self.data, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"attachment source data is not valid base64: {e}") from e


class TextContentBlock(BaseModel):
    """A run of text within a multimodal item, in the position the caller wrote it."""

    type: Literal["text"]
    text: str


class ImageContentBlock(BaseModel):
    """An image within a multimodal item, in the position the caller wrote it."""

    type: Literal["image"]
    source: Base64AttachmentSource


class FileContentBlock(BaseModel):
    """A non-image attachment — a PDF, a spreadsheet — in its input position.

    This stays distinct from ``image`` because providers use different request
    parts for images and documents; retaining the caller's kind avoids guessing.
    """

    type: Literal["file"]
    source: Base64AttachmentSource
    filename: str | None = Field(
        default=None,
        description="Original filename, passed to providers that show one to the model (e.g. OpenAI).",
    )


#: One element of a multimodal ``content`` array. Discriminated on ``type`` so a
#: malformed block reports which variant it failed against instead of dumping
#: every variant's errors.
ContentBlockItem = Annotated[TextContentBlock | ImageContentBlock | FileContentBlock, Field(discriminator="type")]


def _require_content(v: str | list[ContentBlockItem]) -> str | list[ContentBlockItem]:
    if isinstance(v, str):
        if not v.strip():
            raise ValueError("content cannot be empty")
        return v

    if not v:
        raise ValueError("content cannot be empty")
    # Match the string form: a block list with only blank text still carries
    # no content, while an attachment-only list is a valid multimodal input.
    if not any(isinstance(block, (ImageContentBlock, FileContentBlock)) for block in v) and not any(
        block.text.strip() for block in v if isinstance(block, TextContentBlock)
    ):
        raise ValueError("content cannot be empty")
    return v


#: The raw content to retain or extract from: a plain string or an ordered list of
#: content blocks. A named alias rather than an inline union so retain and dry-run
#: share one ``Content`` schema (inline, the client generators minted ``Content1``
#: for the second use). It was first a ``RootModel``, which gave the same schema but
#: turned ``MemoryItem.content`` into a wrapper object — breaking every caller that
#: reads it as the string or list it has always been. The alias keeps the value raw.
Content = TypeAliasType(
    "Content",
    Annotated[
        str | list[ContentBlockItem],
        AfterValidator(_require_content),
        Field(
            title="Content",
            description=(
                "The raw content to retain or extract from. Either a plain string or an ordered list of content blocks."
            ),
        ),
    ],
)


class ContentValidationError(ValueError):
    """Raised when multimodal content fails structural or resource limits."""


# Delimiters chosen from the Unicode mathematical-brackets block: they survive
# `sanitize_text` (which only strips control characters and surrogates), they are
# not separators the recursive chunker splits on, and they are rare enough in
# real prose that a placeholder is unlikely to collide with authored text.
# A caller cannot forge one regardless -- see `neutralize_placeholders`.
PLACEHOLDER_OPEN = "⟦"
PLACEHOLDER_CLOSE = "⟧"
_PLACEHOLDER_BODY = "hs-att:"

#: How much of the sha256 the placeholder carries. The token is text that sits in
#: the document body and in every chunk of it -- so it is stored, it is indexed by
#: BM25, and it spends chunk budget that could have been prose. The full 64-hex
#: digest cost 82 characters a reference; a 12-hex prefix costs 22, which matters
#: once an article carries ten screenshots.
#:
#: 48 bits needs roughly 16 million attachments in ONE bank before a 1% chance of two
#: sharing a prefix, and a collision is not silent: ``attachments`` carries a
#: unique index on (bank_id, short_id), so the second image fails its insert
#: loudly instead of having its placeholder quietly resolve to the first one.
SHORT_ID_LENGTH = 12

#: Matches a well-formed image placeholder and captures its short id.
PLACEHOLDER_RE = re.compile(
    re.escape(PLACEHOLDER_OPEN)
    + re.escape(_PLACEHOLDER_BODY)
    + r"(?P<attachment_id>[0-9a-f]{"
    + str(SHORT_ID_LENGTH)
    + r"})"
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


def short_attachment_id(attachment_hash: str) -> str:
    """The prefix of ``attachment_hash`` that identifies an image inside document text."""
    return attachment_hash[:SHORT_ID_LENGTH]


def attachment_placeholder(attachment_hash: str) -> str:
    """Render the atomic placeholder token standing in for ``attachment_hash``.

    Accepts either the full digest or an already-shortened id, so a caller that
    holds one of them does not have to know which.
    """
    return f"{PLACEHOLDER_OPEN}{_PLACEHOLDER_BODY}{short_attachment_id(attachment_hash)}{PLACEHOLDER_CLOSE}"


def compute_attachment_hash(data: bytes) -> str:
    """Content-address attachment bytes. Identical attachments dedupe on this hash."""
    return hashlib.sha256(data).hexdigest()


def contains_placeholder_like(text: str) -> bool:
    """Whether ``text`` holds anything shaped like a placeholder, valid or not.

    A cheap pre-check so the retain path only pays for resolving a document's own
    attachments when the caller actually wrote something that looks like one.
    """
    return _PLACEHOLDER_LOOKALIKE_RE.search(text) is not None


def neutralize_placeholders(text: str, allowed_ids: "Collection[str] | None" = None) -> str:
    """Strip placeholder-shaped substrings from caller-authored text.

    Only the canonicalizer may mint a placeholder. Without this, a caller could
    write the token by hand and have extraction resolve it to an attachment the
    document never carried -- anything stored in the same bank.

    ``allowed_ids`` exempts attachments the document *already* has. That is not a
    hole in the rule, it is the rule stated properly: re-sending a document's own
    body is an ordinary edit, and the control plane does exactly that when someone
    changes an article's wording. Without the exemption such an edit silently
    deleted every screenshot in the article.
    """
    allowed = set(allowed_ids or ())

    def _keep_or_drop(match: "re.Match[str]") -> str:
        token = match.group(0)
        well_formed = PLACEHOLDER_RE.fullmatch(token)
        if well_formed and well_formed.group("attachment_id") in allowed:
            return token
        return ""

    return _PLACEHOLDER_LOOKALIKE_RE.sub(_keep_or_drop, text)


def iter_placeholder_ids(text: str) -> Iterator[str]:
    """Yield the short image ids referenced by ``text``, in order, with repeats."""
    for match in PLACEHOLDER_RE.finditer(text):
        yield match.group("attachment_id")


def describe_placeholders(text: str, attachments: "Mapping[str, LoadedAttachment] | None" = None) -> str:
    """Replace every placeholder with a short human-readable note.

    For text that becomes a *memory* rather than a prompt. The ``chunks`` and
    ``verbatim`` extraction modes copy chunk text straight into ``fact_text``, so
    without this a recalled fact would read "click the button shown:
    ⟦hs-att:c414cd0e204d⟧" — a content hash presented to a user as though it were
    knowledge. The note keeps the position and says what was there; the machine
    -readable handle travels beside the memory in ``attachments``, not inside its
    text.

    ``attachments`` is optional and only sharpens the note ("[image: image/png]"
    rather than "[attachment]"); the substitution happens either way.
    """

    def _describe(match: "re.Match[str]") -> str:
        record = (attachments or {}).get(match.group("attachment_id"))
        if record is None:
            return "[attachment]"
        if record.filename:
            return f"[{record.kind}: {record.filename}]"
        return f"[{record.kind}: {record.media_type}]"

    return PLACEHOLDER_RE.sub(_describe, text)


def contains_attachment(text: str) -> bool:
    """Whether ``text`` references at least one attachment."""
    return PLACEHOLDER_RE.search(text) is not None


@dataclass(frozen=True)
class RetainAttachment:
    """One decoded attachment from a multimodal retain item."""

    attachment_hash: str
    media_type: str
    data: bytes
    #: Index of the block this attachment came from, kept so the first appearance
    #: of an attachment in a document can be recorded for provenance.
    block_index: int
    #: "image" or "file" — the caller's own distinction, carried through so the
    #: per-provider conversion never has to infer it from the media type.
    kind: Literal["image", "file"] = "image"
    filename: str | None = None

    @property
    def byte_size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class AttachmentOccurrence:
    """One attachment appearance, without duplicating its potentially large bytes."""

    block_index: int
    kind: Literal["image", "file"]
    media_type: str


@dataclass(frozen=True)
class CanonicalContent:
    """A multimodal item flattened to text plus the attachments it references."""

    text: str
    #: Deduplicated by hash, in first-appearance order. The same attachment used
    #: twice in one document yields two placeholders but one entry here, so it is
    #: stored and recorded once.
    attachments: tuple[RetainAttachment, ...]
    #: Every newly supplied attachment occurrence in placeholder order. Existing
    #: placeholders preserved through ``allowed_ids`` have no occurrence entry.
    #: Unlike ``attachments``, this preserves duplicate appearances of the same
    #: attachment at different block positions for accurate attribution.
    occurrences: tuple[AttachmentOccurrence, ...] = ()

    @property
    def has_attachments(self) -> bool:
        return bool(self.attachments)


@dataclass(frozen=True)
class RetainText:
    """One text block from a multimodal retain item."""

    text: str


#: One element of a multimodal item's canonical content blocks.
CanonicalBlock = RetainText | RetainAttachment


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


@dataclass(frozen=True)
class LoadedAttachment:
    """Attachment bytes fetched back out of storage, ready to put in a prompt."""

    media_type: str
    data: bytes
    kind: str = "image"
    filename: str | None = None

    def as_data_uri(self) -> str:
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode()}"


def _prompt_part(attachment: LoadedAttachment) -> dict[str, object]:
    """The canonical (OpenAI-shaped) content part for one attachment.

    Images become ``image_url``; everything else becomes a ``file`` part. Each
    provider converts from these two shapes — see the ``_to_*`` helpers in the
    provider modules — so the vocabulary is decided once, here.
    """
    if attachment.kind == "image":
        return {"type": "image_url", "image_url": {"url": attachment.as_data_uri()}}
    file_part: dict[str, object] = {"file_data": attachment.as_data_uri()}
    if attachment.filename:
        file_part["filename"] = attachment.filename
    return {"type": "file", "file": file_part}


def build_prompt_parts(text: str, attachments: Mapping[str, LoadedAttachment]) -> list[dict[str, object]] | str:
    """Turn placeholder text back into an interleaved multimodal user message.

    Returns OpenAI-style content parts — the canonical wire shape the providers
    convert from — with each image sitting exactly where its placeholder stood,
    so the model reads the screenshot in the same position as the reader of the
    original article did. That positioning is the whole feature: an image
    appended at the end of the prompt loses the sentence that gives it meaning.

    Returns the plain string unchanged when there is nothing to interleave, so a
    text-only chunk produces byte-identical requests to before.

    A placeholder with no entry in ``attachments`` degrades to a short textual note
    rather than raising. The bytes can legitimately be gone — reclaimed, or a
    storage backend swapped underneath an old document — and extracting the
    surrounding prose is far better than failing the whole retain.
    """
    if not PLACEHOLDER_RE.search(text):
        return text

    parts: list[dict[str, object]] = []
    pending: list[str] = []

    def _flush_text() -> None:
        if pending:
            joined = "".join(pending)
            pending.clear()
            if joined.strip():
                parts.append({"type": "text", "text": joined})

    cursor = 0
    for match in PLACEHOLDER_RE.finditer(text):
        pending.append(text[cursor : match.start()])
        attachment = attachments.get(match.group("attachment_id"))
        if attachment is None:
            pending.append("[attachment unavailable]")
        else:
            _flush_text()
            parts.append(_prompt_part(attachment))
        cursor = match.end()
    pending.append(text[cursor:])
    _flush_text()

    return parts


def canonicalize(blocks: Sequence[CanonicalBlock], allowed_ids: "Collection[str] | None" = None) -> CanonicalContent:
    """Flatten ordered text/attachment blocks into the canonical body plus its attachments.

    Image blocks must already be decoded and hashed; this decides only where each
    placeholder lands. A single text block canonicalizes to exactly its own text,
    so ``[{"type": "text", "text": X}]`` and the plain string ``X`` produce an
    identical body -- and therefore an identical ``content_hash``.
    """
    body = ""
    attachments: list[RetainAttachment] = []
    seen: set[str] = set()
    occurrences: list[AttachmentOccurrence] = []
    after_attachment = False

    for block in blocks:
        if isinstance(block, RetainText):
            if after_attachment:
                body = _pad_to_blank_line(body)
            body += neutralize_placeholders(block.text, allowed_ids)
            after_attachment = False
        else:
            body = _pad_to_blank_line(body) + attachment_placeholder(block.attachment_hash)
            after_attachment = True
            occurrences.append(
                AttachmentOccurrence(
                    block_index=block.block_index,
                    kind=block.kind,
                    media_type=block.media_type,
                )
            )
            if block.attachment_hash not in seen:
                seen.add(block.attachment_hash)
                attachments.append(block)

    return CanonicalContent(
        text=body,
        attachments=tuple(attachments),
        occurrences=tuple(occurrences),
    )


def render_chunk_placeholders(
    chunk_text: str,
    occurrences: Sequence[AttachmentOccurrence],
) -> str:
    """Render each placeholder in chunk_text into human-readable '[Block #...: ...]' format.

    Replaces each placeholder token in order using the corresponding occurrence in ``occurrences``.
    """
    if not occurrences or not contains_attachment(chunk_text):
        return chunk_text

    occ_iter = iter(occurrences)

    def _replace(match: re.Match[str]) -> str:
        try:
            occ = next(occ_iter)
            return f"[Block #{occ.block_index}: {occ.kind} ({occ.media_type})]"
        except StopIteration:
            return match.group(0)

    return PLACEHOLDER_RE.sub(_replace, chunk_text)


def occurrences_by_chunk(
    chunk_texts: Sequence[str],
    occurrences: Sequence[AttachmentOccurrence],
) -> list[list[AttachmentOccurrence]]:
    """Partition ordered attachment appearances using each chunk's placeholders."""
    offset = 0
    result: list[list[AttachmentOccurrence]] = []
    for text in chunk_texts:
        count = sum(1 for _ in iter_placeholder_ids(text))
        result.append(list(occurrences[offset : offset + count]))
        offset += count
    return result


def select_occurrences(
    numbers: Sequence[int],
    occurrences: Sequence[AttachmentOccurrence],
) -> list[AttachmentOccurrence]:
    """Resolve one-based prompt attachment numbers to distinct input blocks.

    Unlike retained fact edges, which deduplicate by attachment id, dry-run output
    preserves two positions containing identical bytes because its public contract
    reports the original ``block_index`` values.
    """
    selected: list[AttachmentOccurrence] = []
    seen_blocks: set[int] = set()
    for number in numbers:
        if 1 <= number <= len(occurrences):
            occurrence = occurrences[number - 1]
            if occurrence.block_index not in seen_blocks:
                seen_blocks.add(occurrence.block_index)
                selected.append(occurrence)
    return selected


def validate_and_canonicalize_content(
    content: str | Sequence[ContentBlockItem],
    max_attachment_count: int,
    max_attachment_size_bytes: int,
    max_attachment_size_mb: int,
    allowed_attachment_ids: Collection[str] | None = None,
    path_prefix: str = "content",
    attachment_limit_path: str | None = None,
) -> CanonicalContent:
    """Validate multimodal content blocks and canonicalize them to text with placeholders.

    Raises ContentValidationError if attachment limits or base64 decoding fail.
    """
    if isinstance(content, str):
        return CanonicalContent(
            text=neutralize_placeholders(content, allowed_attachment_ids),
            attachments=(),
            occurrences=(),
        )

    blocks: list[CanonicalBlock] = []
    attachment_count = 0
    attachment_limit_path = attachment_limit_path or path_prefix

    for block_index, block in enumerate(content):
        if isinstance(block, TextContentBlock):
            blocks.append(RetainText(block.text))
            continue

        attachment_count += 1
        if attachment_count > max_attachment_count:
            raise ContentValidationError(
                f"{attachment_limit_path} carries more than {max_attachment_count} attachments. "
                "Split the content across several items."
            )
        try:
            data = block.source.decode()
        except ValueError as e:
            raise ContentValidationError(f"{path_prefix}[{block_index}]: {e}") from e
        if not data:
            raise ContentValidationError(f"{path_prefix}[{block_index}]: attachment source data is empty")
        if len(data) > max_attachment_size_bytes:
            raise ContentValidationError(
                f"{path_prefix}[{block_index}]: attachment is "
                f"{len(data) / (1024 * 1024):.1f}MB, exceeding the "
                f"{max_attachment_size_mb}MB limit for a single attachment."
            )
        blocks.append(
            RetainAttachment(
                attachment_hash=compute_attachment_hash(data),
                media_type=block.source.media_type,
                data=data,
                block_index=block_index,
                kind=block.type,
                filename=getattr(block, "filename", None),
            )
        )

    return canonicalize(blocks, allowed_attachment_ids)
