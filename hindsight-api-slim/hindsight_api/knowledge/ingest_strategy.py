"""Ingestion strategies: which extension pre-processes which file type.

Knowledge banks ingest and search, as memory banks retain and recall. An ingestion strategy is
named, like a retain strategy, and maps file types (an extension such as ``.zip``, or a content
type such as ``application/zip``) to the extension class that handles them, with its settings.
Strategies are defined on the server, ``HINDSIGHT_API_INGEST_STRATEGIES`` (JSON)::

    {
      "game-assets": {
        ".unitypackage": {"extension": "acme.unity:Unpack"},
        ".pdf": {"extension": "acme.ocr:OcrPdf", "settings": {"dpi": 300}}
      }
    }

A strategy applies only when it is chosen: ``strategy`` on the ingested item, else the server's
``HINDSIGHT_API_INGEST_DEFAULT_STRATEGY``. A file whose type the chosen strategy does not map is
chunked by the knowledge bank itself, as with no strategy.

For each file it handles, an extension returns either:

- **files**: raw files, each ingested as its own document of the same bank (``<document_id>/<name>``,
  an upsert like any ingest) and chunked by the knowledge bank, or by the strategy again if it
  maps their type (up to ``MAX_DEPTH`` levels); or
- **passages**: already chunked, embedded as they are.

Extensions run on the worker, inside the document's ingest, with its retries. A ``ValueError``
fails the document for good; any other error is retried. ``version`` is part of the document's
content hash: bump it and ingesting an unchanged file processes it again.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..extensions.loader import resolve_class_path

#: How many levels of files-within-files a strategy may unpack (a zip of zips of ...).
MAX_DEPTH = 3


@dataclass
class IngestFile:
    """The file an extension is given."""

    bank_id: str
    document_id: str
    filename: str
    content_type: str
    data: bytes
    tags: list[str]


@dataclass
class OutputFile:
    """A raw file an extension splits its input into: ingested as its own document."""

    #: Unique within the input; the document id is ``<input document_id>/<name>``.
    name: str
    data: bytes
    #: Its file name, which decides how it is chunked (``report.pdf``). Defaults to ``name``.
    filename: str | None = None
    content_type: str = "application/octet-stream"
    #: Added to the input document's tags.
    tags: list[str] = field(default_factory=list)


@dataclass
class OutputPassage:
    """One already-chunked passage to embed and search, in order. Exactly one of ``text`` / ``data``."""

    modality: Literal["text", "image", "audio", "video"]
    text: str | None = None
    #: Media bytes (an image, an audio or video clip), stored and streamed back as the passage's media.
    data: bytes | None = None
    mime: str | None = None
    page: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass
class IngestOutput:
    """What an extension made of a file: raw files, or passages (not both)."""

    files: list[OutputFile] | None = None
    passages: list[OutputPassage] | None = None

    def __post_init__(self) -> None:
        if (self.files is None) == (self.passages is None):
            raise ValueError("an ingest extension's output holds either files or passages")


class IngestExtension(ABC):
    """Pre-processes the files a strategy maps to it. Built once per strategy entry, with its settings."""

    #: Part of every document's content hash it handles: bump it when the output changes.
    version: str = "1"

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    @abstractmethod
    async def ingest(self, file: IngestFile) -> IngestOutput:
        """Turn ``file`` into raw files or passages."""


@dataclass(frozen=True)
class Handler:
    """One strategy entry: the extension class (``module.path:ClassName``) and its settings."""

    extension: str
    settings: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.extension}|{json.dumps(self.settings, sort_keys=True)}"


@dataclass(frozen=True)
class IngestStrategy:
    name: str
    #: File type (``.zip`` or ``application/zip``, lower case) -> its handler.
    handlers: dict[str, Handler]

    def handler_for(self, content_type: str | None, filename: str | None) -> Handler | None:
        """The extension's file name first (browsers send generic types), then the content type."""
        ext = Path(filename or "").suffix.lower()
        kind = (content_type or "").split(";")[0].strip().lower()
        return self.handlers.get(ext) or self.handlers.get(kind)


def parse_strategies(raw: dict[str, Any] | None) -> dict[str, IngestStrategy]:
    """``{name: {file_type: {"extension": "module:Class", "settings": {...}}}}``."""
    strategies = {}
    for name, mapping in (raw or {}).items():
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"ingestion strategy '{name}' must map file types to extensions")
        handlers = {}
        for file_type, entry in mapping.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("extension"), str):
                raise ValueError(f"ingestion strategy '{name}': '{file_type}' needs an 'extension' (module:Class)")
            settings = entry.get("settings") or {}
            if not isinstance(settings, dict):
                raise ValueError(f"ingestion strategy '{name}': the settings of '{file_type}' must be an object")
            handlers[file_type.lower()] = Handler(extension=entry["extension"], settings=settings)
        strategies[name] = IngestStrategy(name=name, handlers=handlers)
    return strategies


_extensions: dict[str, IngestExtension] = {}
# A threading lock, not an asyncio one: the cache is process-wide and the critical section never awaits.
_extensions_lock = threading.Lock()


def load_extension(handler: Handler) -> IngestExtension:
    """The extension for one strategy entry, built once per class and settings."""
    with _extensions_lock:
        extension = _extensions.get(handler.key)
        if extension is None:
            extension = _extensions[handler.key] = resolve_class_path(handler.extension, IngestExtension)(
                handler.settings
            )
        return extension
