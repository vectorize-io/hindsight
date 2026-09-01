"""Content-addressed persistence for images retained inline with document content.

Bytes go to the existing :class:`~hindsight_api.engine.storage.base.FileStorage`
abstraction — the same one uploaded files use — so a deployment on S3, GCS or
Azure keeps image bytes out of the database without any new backend. The
``bank_images`` row records what the bytes are and where they live, keyed by
``(bank_id, image_hash)``.

Both writes are idempotent, because the key *is* the content hash. That is what
makes it safe to persist at the API ingress, before retain has decided whether it
will commit: re-retaining an unchanged document rewrites the same bytes to the
same key, and a retain that later fails leaves a blob that the next retain of the
same image simply reuses.

Resolving the other way — from a placeholder in some chunk's text back to bytes —
is what recall provenance needs, and is served by :func:`load_bank_images`.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..memory_engine import fq_table
from .image_content import RetainImage

logger = logging.getLogger(__name__)


def image_storage_key(bank_id: str, image_hash: str) -> str:
    """Where an image's bytes live, derived entirely from its content hash.

    Mirrors the ``banks/{bank_id}/...`` layout the file-retain and export paths
    already use. Bank-scoped rather than global so a bank's blobs can be swept as
    a unit and one bank can never read another's bytes by guessing a key.
    """
    return f"banks/{bank_id}/images/sha256-{image_hash}"


@dataclass(frozen=True)
class StoredImage:
    """An image that is committed to file storage and recorded for a bank."""

    image_hash: str
    media_type: str
    byte_size: int
    storage_key: str


async def store_images(
    file_storage,
    conn,
    bank_id: str,
    images: Sequence[RetainImage],
) -> list[StoredImage]:
    """Persist ``images`` for ``bank_id``, skipping any already stored.

    Returns one :class:`StoredImage` per input image, in the order given. Callers
    hand this the deduplicated list from
    :func:`~hindsight_api.engine.retain.image_content.canonicalize`, so an image
    used twice in one document is stored once.
    """
    if not images:
        return []

    already_stored = await _existing_hashes(conn, bank_id, [image.image_hash for image in images])

    stored: list[StoredImage] = []
    for image in images:
        key = image_storage_key(bank_id, image.image_hash)
        record = StoredImage(
            image_hash=image.image_hash,
            media_type=image.media_type,
            byte_size=image.byte_size,
            storage_key=key,
        )
        stored.append(record)
        if image.image_hash in already_stored:
            # The bytes are addressed by their own hash, so an existing row
            # guarantees identical content. Re-uploading would be pure cost.
            continue
        await file_storage.store(
            file_data=image.data,
            key=key,
            metadata={"content_type": image.media_type, "bank_id": bank_id},
        )

    await _record_images(conn, bank_id, [record for record in stored if record.image_hash not in already_stored])
    return stored


async def load_bank_images(conn, bank_id: str, image_hashes: Sequence[str]) -> dict[str, StoredImage]:
    """Resolve image hashes — as found in a document or chunk's text — to their records.

    Hashes with no row are simply absent from the result. That is not an error: a
    document retained before its blob was reclaimed still names the hash in its
    text, and a caller rendering provenance should show the fact without the image
    rather than fail.
    """
    if not image_hashes:
        return {}

    rows = await conn.fetch(
        f"""
        SELECT image_hash, media_type, byte_size, storage_key
        FROM {fq_table("bank_images")}
        WHERE bank_id = $1 AND image_hash = ANY($2::text[])
        """,
        bank_id,
        list(dict.fromkeys(image_hashes)),
    )
    return {
        row["image_hash"]: StoredImage(
            image_hash=row["image_hash"],
            media_type=row["media_type"],
            byte_size=row["byte_size"],
            storage_key=row["storage_key"],
        )
        for row in rows
    }


async def _existing_hashes(conn, bank_id: str, image_hashes: Sequence[str]) -> set[str]:
    rows = await conn.fetch(
        f"SELECT image_hash FROM {fq_table('bank_images')} WHERE bank_id = $1 AND image_hash = ANY($2::text[])",
        bank_id,
        list(dict.fromkeys(image_hashes)),
    )
    return {row["image_hash"] for row in rows}


async def _record_images(conn, bank_id: str, images: Sequence[StoredImage]) -> None:
    if not images:
        return
    # DO NOTHING rather than DO UPDATE: the PK is the content hash, so a conflict
    # means an identical image, and there is nothing to update. It also makes two
    # concurrent retains of the same image a no-op for the loser instead of a
    # deadlock-prone write.
    await conn.executemany(
        f"""
        INSERT INTO {fq_table("bank_images")} (bank_id, image_hash, media_type, byte_size, storage_key)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (bank_id, image_hash) DO NOTHING
        """,
        [(bank_id, image.image_hash, image.media_type, image.byte_size, image.storage_key) for image in images],
    )
