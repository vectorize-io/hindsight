"""Disk-backed transfer archive shared by export and import paths."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class TransferArchive:
    """Seekable temporary ZIP that can also be streamed and safely removed."""

    path: str
    size_bytes: int

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.iter_bytes()

    def cleanup(self) -> None:
        """Remove the archive, allowing cleanup to be retried safely."""
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    async def iter_bytes(self, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        try:
            with open(self.path, "rb") as source:
                while chunk := source.read(chunk_size):
                    yield chunk
        finally:
            self.cleanup()
