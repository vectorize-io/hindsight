"""Content-addressed persistence for attachments retained inline with document content.

Bytes go to the existing :class:`~hindsight_api.engine.storage.base.FileStorage`
abstraction — the same one uploaded files use — so a deployment on S3, GCS or
Azure keeps attachment bytes out of the database without any new backend. The
``attachments`` row records what the bytes are and where they live, keyed by
``(bank_id, attachment_hash)`` and resolved by the ``short_id`` prefix that document
text carries.

Both writes are idempotent, because the key *is* the content hash. That is what
makes it safe to persist at the API ingress, before retain has decided whether it
will commit: re-retaining an unchanged document rewrites the same bytes to the
same key, and a retain that later fails leaves a blob that the next retain of the
same attachment simply reuses.

Resolving the other way — from a placeholder in some chunk's text back to bytes —
is what recall provenance needs, and is served by :func:`load_bank_attachments`.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..memory_engine import fq_table
from .attachment_content import LoadedAttachment, RetainAttachment, short_attachment_id

logger = logging.getLogger(__name__)


def attachment_storage_key(bank_id: str, attachment_hash: str) -> str:
    """Where an attachment's bytes live, derived entirely from its content hash.

    Mirrors the ``banks/{bank_id}/...`` layout the file-retain and export paths
    already use. Bank-scoped rather than global so a bank's blobs can be swept as
    a unit and one bank can never read another's bytes by guessing a key.
    """
    return f"banks/{bank_id}/attachments/sha256-{attachment_hash}"


@dataclass(frozen=True)
class StoredAttachment:
    """An attachment that is committed to file storage and recorded for a bank."""

    attachment_hash: str
    #: The prefix of ``attachment_hash`` that document text references. Resolution
    #: looks up by this, because it is what a placeholder can afford to carry.
    short_id: str
    media_type: str
    byte_size: int
    storage_key: str
    #: "attachment" or "file", as the caller wrote it. Decides the prompt part shape.
    kind: str = "attachment"
    #: The name this attachment was given *in the document being read*. Not a
    #: property of the blob — it lives on `document_attachments` — so it is None
    #: whenever the reader had no document in hand (the byte-serving endpoint,
    #: the reclaim sweep).
    filename: str | None = None


async def store_images(
    file_storage,
    conn,
    bank_id: str,
    attachments: Sequence[RetainAttachment],
) -> list[StoredAttachment]:
    """Persist ``attachments`` for ``bank_id``, skipping any already stored.

    Returns one :class:`StoredAttachment` per input attachment, in the order given. Callers
    hand this the deduplicated list from
    :func:`~hindsight_api.engine.retain.attachment_content.canonicalize`, so an attachment
    used twice in one document is stored once.
    """
    if not attachments:
        return []

    already_stored = await _existing_hashes(conn, bank_id, [attachment.attachment_hash for attachment in attachments])

    stored: list[StoredAttachment] = []
    for attachment in attachments:
        key = attachment_storage_key(bank_id, attachment.attachment_hash)
        record = StoredAttachment(
            attachment_hash=attachment.attachment_hash,
            short_id=short_attachment_id(attachment.attachment_hash),
            media_type=attachment.media_type,
            byte_size=attachment.byte_size,
            storage_key=key,
            kind=attachment.kind,
            filename=attachment.filename,
        )
        stored.append(record)
        if attachment.attachment_hash in already_stored:
            # The bytes are addressed by their own hash, so an existing row
            # guarantees identical content. Re-uploading would be pure cost.
            continue
        await file_storage.store(
            file_data=attachment.data,
            key=key,
            metadata={"content_type": attachment.media_type, "bank_id": bank_id},
        )

    await _record_attachments(
        conn, bank_id, [record for record in stored if record.attachment_hash not in already_stored]
    )
    return stored


async def load_bank_attachments(
    conn,
    bank_id: str,
    attachment_ids: Sequence[str],
    document_id: str | None = None,
) -> dict[str, StoredAttachment]:
    """Resolve short attachment ids — as found in a document or chunk's text — to records.

    Keyed by ``short_id``, which is what a placeholder carries. Ids with no row
    are simply absent from the result. That is not an error: a document retained
    before its blob was reclaimed still names the id in its text, and a caller
    rendering provenance should show the fact without the attachment rather than fail.

    ``document_id`` fills in ``filename``, which lives on the document edge rather
    than the blob — the same bytes can be attached under a different name in
    another document. Callers with no document in hand (serving the bytes,
    sweeping for orphans) omit it and get ``filename=None``, which neither needs.
    """
    if not attachment_ids:
        return {}

    # LEFT JOIN, and on a NULL document_id it matches nothing: a caller without a
    # document still gets every blob, just with no name for it.
    rows = await conn.fetch(
        f"""
        SELECT ba.attachment_hash, ba.short_id, ba.media_type, ba.byte_size, ba.storage_key,
               ba.kind, da.filename
        FROM {fq_table("attachments")} ba
        LEFT JOIN {fq_table("document_attachments")} da
          ON da.bank_id = ba.bank_id
         AND da.attachment_hash = ba.attachment_hash
         AND da.document_id = $3
        WHERE ba.bank_id = $1 AND ba.short_id = ANY($2::text[])
        """,
        bank_id,
        list(dict.fromkeys(attachment_ids)),
        document_id,
    )
    return {
        row["short_id"]: StoredAttachment(
            attachment_hash=row["attachment_hash"],
            short_id=row["short_id"],
            media_type=row["media_type"],
            byte_size=row["byte_size"],
            storage_key=row["storage_key"],
            kind=row["kind"],
            filename=row["filename"],
        )
        for row in rows
    }


class RetainAttachmentLoader:
    """Fetches attachment bytes back for extraction, cached for one retain operation.

    Extraction runs many chunks concurrently, and a document commonly repeats one
    attachment (a product screenshot referenced from several sections), so the same
    blob would otherwise be pulled from S3 once per chunk. The cache is bounded by
    total bytes rather than entry count because the entries are attachments: a hundred
    thumbnails and a hundred full-page screenshots are three orders of magnitude
    apart, and only the byte count predicts the memory the retain holds.

    Eviction is "stop admitting", not LRU. A retain's working set is one chunk's
    attachments; past the budget the loader keeps serving correctly and simply stops
    growing, which is the behaviour worth having here — a cache miss costs a
    fetch, while an unbounded cache costs the worker.
    """

    def __init__(
        self,
        file_storage,
        backend,
        bank_id: str,
        *,
        filenames: dict[str, str] | None = None,
        max_cached_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._file_storage = file_storage
        self._backend = backend
        self._bank_id = bank_id
        # Short id -> the name the caller gave it in this retain. The blob row no
        # longer carries a filename (it belongs to the reference, not the bytes),
        # and the document edge does not exist yet while extraction is running —
        # the document row is written later in the same retain. So the names come
        # from the request, which is where they were anyway.
        self._filenames = filenames or {}
        self._max_cached_bytes = max_cached_bytes
        self._cache: dict[str, LoadedAttachment] = {}
        self._cached_bytes = 0
        self._lock = asyncio.Lock()

    async def load(self, attachment_ids: Sequence[str]) -> dict[str, LoadedAttachment]:
        """Resolve short attachment ids to bytes. Ids that cannot be resolved are omitted."""
        wanted = list(dict.fromkeys(attachment_ids))
        if not wanted:
            return {}

        resolved = {i: self._cache[i] for i in wanted if i in self._cache}
        missing = [i for i in wanted if i not in resolved]
        if not missing:
            return resolved

        async with self._backend.acquire() as conn:
            records = await load_bank_attachments(conn, self._bank_id, missing)

        for attachment_id in missing:
            record = records.get(attachment_id)
            if record is None:
                # No row: the attachment was never stored for this bank, or its row was
                # reclaimed. The placeholder degrades to a note in the prompt.
                logger.warning(
                    "No attachments row for %s in bank %s; extracting without it", attachment_id, self._bank_id
                )
                continue
            try:
                data = await self._file_storage.retrieve(record.storage_key)
            except FileNotFoundError:
                logger.warning(
                    "attachments row for %s in bank %s points at missing key %s; extracting without it",
                    attachment_id,
                    self._bank_id,
                    record.storage_key,
                )
                continue
            loaded = LoadedAttachment(
                media_type=record.media_type,
                data=data,
                kind=record.kind,
                filename=self._filenames.get(attachment_id) or record.filename,
            )
            resolved[attachment_id] = loaded
            async with self._lock:
                if attachment_id not in self._cache and self._cached_bytes + len(data) <= self._max_cached_bytes:
                    self._cache[attachment_id] = loaded
                    self._cached_bytes += len(data)

        return resolved


async def _existing_hashes(conn, bank_id: str, attachment_hashes: Sequence[str]) -> set[str]:
    rows = await conn.fetch(
        f"SELECT attachment_hash FROM {fq_table('attachments')} WHERE bank_id = $1 AND attachment_hash = ANY($2::text[])",
        bank_id,
        list(dict.fromkeys(attachment_hashes)),
    )
    return {row["attachment_hash"] for row in rows}


async def _record_attachments(conn, bank_id: str, attachments: Sequence[StoredAttachment]) -> None:
    if not attachments:
        return
    # DO NOTHING rather than DO UPDATE: the PK is the content hash, so a conflict
    # means an identical attachment, and there is nothing to update. It also makes two
    # concurrent retains of the same attachment a no-op for the loser instead of a
    # deadlock-prone write.
    #
    # The conflict target is deliberately the PK and NOT (bank_id, short_id). A
    # short-id clash between two *different* attachments must not be swallowed — it
    # would leave the second attachment unrecorded and its placeholder resolving to the
    # first. The unique index on (bank_id, short_id) therefore raises instead, and
    # the retain fails where an operator can see it.
    await conn.executemany(
        f"""
        INSERT INTO {fq_table("attachments")}
            (bank_id, attachment_hash, short_id, media_type, byte_size, storage_key, kind)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (bank_id, attachment_hash) DO NOTHING
        """,
        [
            (
                bank_id,
                attachment.attachment_hash,
                attachment.short_id,
                attachment.media_type,
                attachment.byte_size,
                attachment.storage_key,
                attachment.kind,
            )
            for attachment in attachments
        ],
    )
