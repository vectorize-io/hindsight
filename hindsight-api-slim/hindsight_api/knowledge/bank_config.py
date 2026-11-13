"""Per-bank ingest settings for knowledge banks.

A knowledge bank may override how its files are split into passages, and nothing else: models,
providers, keys, vector sizes, search weights and the server's limits stay the server's. The
overrides live where a memory bank's do, in the bank row's ``config`` JSONB, under the server
field's name (``knowledge_<field>``), so they never collide with a memory-bank setting and the
memory-bank config resolver (which keeps only its own configurable fields) never reads them.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..config import get_config
from .media import MediaType

#: Where an override is stored in ``banks.config``: the server field it shadows.
STORED_PREFIX = "knowledge_"


class KnowledgeBankSettings(BaseModel):
    """The ingest settings a knowledge bank splits its files with: its overrides, else the server's."""

    model_config = ConfigDict(frozen=True)

    text_passage_tokens: int
    pdf_page_pixels: int
    pdf_bands: bool
    image_tokens: int
    audio_window_seconds: int
    video_window_seconds: int
    media_overlap_seconds: int
    max_media_seconds: int
    max_pdf_pages: int
    disabled_media_types: list[MediaType]


class KnowledgeBankOverrides(BaseModel):
    """What a bank overrides; a field left None follows the server. Strict, so ``"5"`` or ``true``
    is not taken for a number."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text_passage_tokens: int | None = Field(default=None, ge=16, le=8192)
    pdf_page_pixels: int | None = Field(default=None, ge=256, le=8192)
    pdf_bands: bool | None = None
    # EmbeddingGemma 2's vision-token range.
    image_tokens: int | None = Field(default=None, ge=70, le=1120)
    audio_window_seconds: int | None = Field(default=None, ge=1, le=3600)
    video_window_seconds: int | None = Field(default=None, ge=1, le=3600)
    media_overlap_seconds: int | None = Field(default=None, ge=0, le=3599)
    # Upper bounds are the server's own limits, checked in ``validate_overrides``: a bank may ingest
    # less than the server allows, never more.
    max_media_seconds: int | None = Field(default=None, ge=1)
    max_pdf_pages: int | None = Field(default=None, ge=1)
    disabled_media_types: list[MediaType] | None = None

    @model_validator(mode="after")
    def _unique_media_types(self) -> KnowledgeBankOverrides:
        if self.disabled_media_types is not None:
            self.disabled_media_types = sorted(set(self.disabled_media_types))
        return self


OVERRIDABLE_FIELDS: tuple[str, ...] = tuple(KnowledgeBankOverrides.model_fields)


def server_settings() -> KnowledgeBankSettings:
    config = get_config()
    return KnowledgeBankSettings(
        **{
            name: getattr(config, STORED_PREFIX + name) for name in OVERRIDABLE_FIELDS if name != "disabled_media_types"
        },
        disabled_media_types=sorted(config.knowledge_disabled_media_types),
    )


def resolve(overrides: KnowledgeBankOverrides) -> KnowledgeBankSettings:
    server = server_settings()
    update: dict[str, Any] = overrides.model_dump(exclude_none=True)
    # A media type the server refuses stays refused, and a limit the server lowered after the bank
    # set its own is the server's again: a bank only ever narrows these.
    if "disabled_media_types" in update:
        update["disabled_media_types"] = sorted(set(update["disabled_media_types"]) | set(server.disabled_media_types))
    for limit in ("max_media_seconds", "max_pdf_pages"):
        if limit in update:
            update[limit] = min(update[limit], getattr(server, limit))
    return server.model_copy(update=update)


def stored_overrides(raw: Any) -> KnowledgeBankOverrides:
    """A bank's overrides from its ``banks.config`` value (JSONB, or its JSON text)."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    mine = {
        name: raw[STORED_PREFIX + name]
        for name in OVERRIDABLE_FIELDS
        if raw and raw.get(STORED_PREFIX + name) is not None
    }
    return KnowledgeBankOverrides.model_validate(mine)


def validate_overrides(current: KnowledgeBankOverrides, updates: dict[str, Any]) -> KnowledgeBankOverrides:
    """The bank's overrides once ``updates`` apply (a None value clears that one). Raises ``ValueError``
    naming what is wrong: a field that is not a per-bank setting, or a value out of range."""
    unknown = sorted(set(updates) - set(OVERRIDABLE_FIELDS))
    if unknown:
        raise ValueError(
            f"not per-bank knowledge settings: {', '.join(unknown)} (a bank may set: {', '.join(OVERRIDABLE_FIELDS)})"
        )
    merged = {**current.model_dump(exclude_none=True), **updates}
    try:
        overrides = KnowledgeBankOverrides.model_validate({k: v for k, v in merged.items() if v is not None})
    except ValidationError as exc:
        raise ValueError(
            "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_url=False))
        ) from exc
    server = server_settings()
    for limit in ("max_media_seconds", "max_pdf_pages"):
        value = getattr(overrides, limit)
        if value is not None and value > getattr(server, limit):
            raise ValueError(f"{limit}: may not exceed the server's limit of {getattr(server, limit)}")
    # Checked on the values the bank will split with, so overriding one side is checked against the other.
    # Only when the bank sets one of them: the server's own values are the operator's to get right.
    settings = resolve(overrides)
    window = min(settings.audio_window_seconds, settings.video_window_seconds)
    sets_media = {"audio_window_seconds", "video_window_seconds", "media_overlap_seconds"} & set(
        overrides.model_dump(exclude_none=True)
    )
    if sets_media and settings.media_overlap_seconds >= window:
        raise ValueError(
            f"media_overlap_seconds ({settings.media_overlap_seconds}) must be less than the audio and video "
            f"windows ({settings.audio_window_seconds}s, {settings.video_window_seconds}s)"
        )
    return overrides


def to_stored(overrides: KnowledgeBankOverrides) -> dict[str, Any]:
    return {STORED_PREFIX + k: v for k, v in overrides.model_dump(exclude_none=True).items()}
