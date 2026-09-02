"""Tests for what reflect tool results carry into the agent's context.

Reflect tool results are dropped into the model's context verbatim, so every
field in them is spent context. `_drop_unread_fields` removes the retrieval
plumbing the agent never reads. The risk of removing fields is that something
downstream quietly needed one, so these tests pin BOTH directions: the heavy
fields go, and the fields the loop actually depends on stay.
"""

from __future__ import annotations

import unittest

from hindsight_api.engine.reflect.tools import _UNREAD_RESULT_FIELDS, _drop_unread_fields
from hindsight_api.engine.response_models import ChunkInfo


def _dumped_observation() -> dict:
    """A serialized result carrying every field the trim targets."""
    return {
        "id": "obs-1",
        "text": "The deploy runs at 09:00 UTC.",
        "occurred_start": "2026-08-01T00:00:00Z",
        "tags": ["ops"],
        "source_fact_ids": ["f1", "f2"],
        "scores": {"semantic": 0.71, "reranker": 0.93, "final": 1.04},
        "metadata": {"ingest_batch": "b-17"},
        "entities": ["Robert Smith"],
        "chunk_id": "chunk-9",
        "document_id": "doc-3",
    }


class DropUnreadFieldsTests(unittest.TestCase):
    def test_every_unread_field_is_removed(self):
        trimmed = _drop_unread_fields(_dumped_observation())
        for field in _UNREAD_RESULT_FIELDS:
            self.assertNotIn(field, trimmed, f"{field} should not reach the agent")

    def test_the_fields_the_loop_depends_on_survive(self):
        """The direction that actually breaks things.

        Citations key on ``id``; ``based_on`` persists id/text/type/context; the
        expand tool takes memory_ids. Dropping any of these would silently
        degrade answers rather than raise.

        ``entities`` survives too: it carries canonical entity *names* (not
        ids), which are semantic signal the surface text may lack ("Bob" in the
        text vs canonical "Robert Smith"). Reflect's recalls don't populate it
        yet, but the trim must not eat the names once ``include_entities`` is
        turned on.
        """
        trimmed = _drop_unread_fields(_dumped_observation())
        for field in ("id", "text", "occurred_start", "tags", "source_fact_ids", "entities"):
            self.assertIn(field, trimmed, f"{field} is load-bearing and must survive the trim")

    def test_missing_fields_are_not_an_error(self):
        """Results legitimately omit these — `_prune_nulls` runs first."""
        self.assertEqual(_drop_unread_fields({"id": "obs-1"}), {"id": "obs-1"})

    def test_trim_is_idempotent(self):
        once = _drop_unread_fields(_dumped_observation())
        self.assertEqual(_drop_unread_fields(dict(once)), once)


class ChunkEnvelopeTests(unittest.TestCase):
    def test_chunk_info_carries_no_unread_fields(self):
        """Pins why ``chunks`` is left untrimmed in tool_recall.

        ChunkInfo holds only chunk_text / chunk_index / truncated, so trimming
        it would be a no-op and its absence is not an inconsistency. If a future
        field lands here that IS plumbing, this test fails and the decision gets
        revisited instead of silently going stale.
        """
        overlap = set(ChunkInfo.model_fields) & set(_UNREAD_RESULT_FIELDS)
        self.assertEqual(
            overlap,
            set(),
            f"ChunkInfo gained trimmable field(s) {overlap}; tool_recall's chunks should now be trimmed too",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
