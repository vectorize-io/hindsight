"""Precision-aware embedding and BM25 date enrichment regressions."""

from datetime import UTC, datetime

import pytest

from hindsight_api.engine.memories.base import build_text_signals_from_parts
from hindsight_api.engine.retain.embedding_processing import augment_texts_with_dates
from hindsight_api.engine.retain.types import ExtractedFact
from hindsight_api.engine.temporal_precision import OCCURRENCE_PRECISION_METADATA_KEY
from hindsight_api.engine.transfer.importer import _to_extracted_fact
from hindsight_api.engine.transfer.schema import TransferFact


def _format_month(value: datetime) -> str:
    return value.strftime("%B %Y")


@pytest.mark.parametrize(
    ("precision", "start", "expected", "forbidden"),
    [
        ("year", datetime(2026, 1, 1, tzinfo=UTC), "happened in 2026", "January"),
        ("month", datetime(2026, 8, 1, tzinfo=UTC), "happened in August 2026", "August 1"),
        ("day", datetime(2026, 8, 30, tzinfo=UTC), "happened in August 2026", "August 30"),
    ],
)
def test_embedding_enrichment_preserves_occurrence_precision(precision, start, expected, forbidden):
    fact = ExtractedFact(
        fact_text="User attended the summit",
        fact_type="world",
        occurred_start=start,
        occurred_end=start,
        metadata={OCCURRENCE_PRECISION_METADATA_KEY: precision},
    )

    augmented = augment_texts_with_dates([fact], _format_month)[0]

    assert expected in augmented
    assert forbidden not in augmented


def test_embedding_enrichment_does_not_apply_occurrence_precision_to_mention_fallback():
    fact = ExtractedFact(
        fact_text="The user prefers local models",
        fact_type="world",
        mentioned_at=datetime(2026, 8, 30, tzinfo=UTC),
        metadata={OCCURRENCE_PRECISION_METADATA_KEY: "year"},
    )

    augmented = augment_texts_with_dates([fact], _format_month)[0]

    assert "happened in August 2026" in augmented
    assert "happened in 2026" not in augmented


def test_embedding_enrichment_recovers_legacy_coarse_when_for_facts_only():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    world = ExtractedFact(
        fact_text="Summit talk | When: 2026 | Involving: user",
        fact_type="world",
        occurred_start=start,
        occurred_end=start,
    )
    observation = ExtractedFact(
        fact_text="Observation | When: 2026 | Involving: user",
        fact_type="observation",
        occurred_start=start,
        occurred_end=start,
    )

    world_augmented, observation_augmented = augment_texts_with_dates([world, observation], _format_month)

    assert "happened in 2026" in world_augmented
    assert "happened in January 2026" in observation_augmented


@pytest.mark.parametrize(
    ("precision", "start", "end", "expected", "forbidden"),
    [
        ("year", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), "user 2026", "January"),
        (
            "month",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 1, tzinfo=UTC),
            "user August 2026",
            "August 1",
        ),
        (
            "day",
            datetime(2026, 8, 30, tzinfo=UTC),
            datetime(2026, 8, 30, tzinfo=UTC),
            "user August 30 2026",
            "August 1",
        ),
        (
            "range",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 30, tzinfo=UTC),
            "user August 1 2026 August 30 2026",
            "January",
        ),
    ],
)
def test_bm25_text_signals_preserve_occurrence_precision(precision, start, end, expected, forbidden):
    signals = build_text_signals_from_parts(
        entity_names=["user"],
        fact_text="User attended the summit",
        fact_type="world",
        metadata={OCCURRENCE_PRECISION_METADATA_KEY: precision},
        occurred_start=start,
        occurred_end=end,
    )

    assert signals == expected
    assert forbidden not in signals


def test_bm25_text_signals_do_not_use_mention_time_for_undated_facts():
    signals = build_text_signals_from_parts(
        entity_names=["user"],
        fact_text="The user prefers local models",
        fact_type="world",
        metadata={OCCURRENCE_PRECISION_METADATA_KEY: "year"},
        occurred_start=None,
        occurred_end=None,
    )

    assert signals == "user"


def test_transfer_round_trip_reuses_precision_for_regenerated_embedding_and_signals():
    occurred = datetime(2026, 1, 1, tzinfo=UTC)
    transferred = TransferFact(
        text="Summit talk | When: 2026 | Involving: user",
        fact_type="world",
        occurred_start=occurred,
        occurred_end=occurred,
        mentioned_at=datetime(2026, 8, 30, tzinfo=UTC),
        metadata={OCCURRENCE_PRECISION_METADATA_KEY: "year"},
        entities=["user"],
    )
    restored_transfer = TransferFact.model_validate_json(transferred.model_dump_json())
    extracted = _to_extracted_fact(restored_transfer)

    augmented = augment_texts_with_dates([extracted], _format_month)[0]
    signals = build_text_signals_from_parts(
        entity_names=extracted.entities,
        fact_text=extracted.fact_text,
        fact_type=extracted.fact_type,
        metadata=extracted.metadata,
        occurred_start=extracted.occurred_start,
        occurred_end=extracted.occurred_end,
    )

    assert extracted.metadata == {OCCURRENCE_PRECISION_METADATA_KEY: "year"}
    assert "happened in 2026" in augmented
    assert "January" not in augmented
    assert signals == "user 2026"
