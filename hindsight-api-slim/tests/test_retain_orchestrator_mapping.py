"""Unit tests for retain orchestrator mapping and embeddings length guarantee.

Regression coverage for issue #1037: a silent length mismatch between the
extracted facts and the generated embeddings caused
`_map_results_to_contents` to raise IndexError during batch_retain.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.retain.orchestrator import (
    _map_results_to_contents,
    _process_extracted_facts,
    _remap_causal_relations,
)
from hindsight_api.engine.retain.types import CausalRelation, ExtractedFact, ProcessedFact, RetainContent


def _make_processed_fact(content_index: int, text: str = "fact") -> ProcessedFact:
    return ProcessedFact(
        fact_text=text,
        fact_type="world",
        embedding=[0.0, 0.0, 0.0],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=datetime(2026, 1, 1),
        context="",
        metadata={},
        content_index=content_index,
    )


def _make_content(text: str = "x") -> RetainContent:
    return RetainContent(content=text)


def _make_extracted_fact(text: str, chunk_index: int, causal_targets: list[int] | None = None) -> ExtractedFact:
    return ExtractedFact(
        fact_text=text,
        fact_type="world",
        chunk_index=chunk_index,
        mentioned_at=datetime(2026, 1, 1),
        causal_relations=[
            CausalRelation(relation_type="caused_by", target_fact_index=target) for target in causal_targets or []
        ],
    )


class TestMapResultsToContents:
    def test_groups_unit_ids_by_content_index(self):
        contents = [_make_content("a"), _make_content("b"), _make_content("c")]
        processed = [
            _make_processed_fact(0, "a1"),
            _make_processed_fact(0, "a2"),
            _make_processed_fact(2, "c1"),
        ]
        unit_ids = ["u-a1", "u-a2", "u-c1"]

        result = _map_results_to_contents(contents, processed, unit_ids)

        assert result == [["u-a1", "u-a2"], [], ["u-c1"]]

    def test_handles_out_of_range_content_index(self):
        contents = [_make_content("a"), _make_content("b")]
        processed = [
            _make_processed_fact(-1, "f1"),
            _make_processed_fact(99, "f2"),
        ]
        unit_ids = ["u1", "u2"]

        result = _map_results_to_contents(contents, processed, unit_ids)

        assert result == [["u1"], ["u2"]]

    def test_empty_inputs(self):
        assert _map_results_to_contents([], [], []) == []

    def test_length_mismatch_raises(self):
        # Regression for #1037: previously the function silently overran unit_ids.
        contents = [_make_content("a")]
        processed = [_make_processed_fact(0), _make_processed_fact(0)]
        unit_ids = ["u1"]  # one fewer than processed_facts

        with pytest.raises(ValueError, match="length mismatch"):
            _map_results_to_contents(contents, processed, unit_ids)

    def test_unit_ids_assigned_by_processed_fact_position(self):
        # Even if processed_facts are interleaved across contents, each unit_id
        # must follow its corresponding processed_fact (positional alignment).
        contents = [_make_content("a"), _make_content("b")]
        processed = [
            _make_processed_fact(1, "b1"),
            _make_processed_fact(0, "a1"),
            _make_processed_fact(1, "b2"),
        ]
        unit_ids = ["u-b1", "u-a1", "u-b2"]

        result = _map_results_to_contents(contents, processed, unit_ids)

        assert result == [["u-a1"], ["u-b1", "u-b2"]]


class TestProcessExtractedFacts:
    def test_filters_extracted_and_processed_facts_in_lockstep(self):
        extracted = [
            _make_extracted_fact("Alice joined Acme", 10),
            _make_extracted_fact("...", 11),
            _make_extracted_fact("Bob leads the ML team", 12),
        ]
        embeddings = [[10.0], [11.0], [12.0]]

        result = _process_extracted_facts(extracted, embeddings)

        assert [fact.fact_text for fact in result.extracted_facts] == ["Alice joined Acme", "Bob leads the ML team"]
        assert [fact.chunk_index for fact in result.extracted_facts] == [10, 12]
        assert [fact.fact_text for fact in result.processed_facts] == ["Alice joined Acme", "Bob leads the ML team"]
        assert [fact.embedding for fact in result.processed_facts] == [[10.0], [12.0]]
        assert result.retained_index_by_original == [0, None, 1]

    def test_remaps_canonical_relations_and_drops_rejected_targets(self):
        extracted = [
            _make_extracted_fact("Initial event", 10),
            _make_extracted_fact("...", 11),
            _make_extracted_fact("Consequence", 12, [0, 1]),
            _make_extracted_fact("Later consequence", 13, [1, 2]),
        ]

        result = _process_extracted_facts(extracted, [[0.0], [1.0], [2.0], [3.0]])

        assert [
            [relation.target_fact_index for relation in fact.causal_relations] for fact in result.processed_facts
        ] == [
            [],
            [0],
            [1],
        ]
        assert [fact.causal_relations for fact in result.extracted_facts] == [
            fact.causal_relations for fact in result.processed_facts
        ]

    def test_remaps_transfer_relation_matrix_with_original_source_ordinals(self):
        relations = [
            [],
            [],
            [
                CausalRelation(relation_type="causes", target_fact_index=0),
                CausalRelation(relation_type="prevents", target_fact_index=1),
            ],
            [CausalRelation(relation_type="enables", target_fact_index=2)],
        ]

        remapped = _remap_causal_relations(relations, [0, None, 1, 2])

        assert [
            [(relation.relation_type, relation.target_fact_index) for relation in fact_relations]
            for fact_relations in remapped
        ] == [[], [("causes", 0)], [("enables", 1)]]

    def test_rejects_fact_embedding_length_mismatch(self):
        extracted = [_make_extracted_fact("one", 1), _make_extracted_fact("two", 2)]

        with pytest.raises(ValueError, match="length mismatch"):
            _process_extracted_facts(extracted, [[1.0]])


class TestEmbeddingSingleValidation:
    def test_generate_embedding_preserves_validation_runtime_error(self):
        backend = MagicMock()
        backend.dimension = 3
        backend.encode_documents.return_value = [[]]

        with pytest.raises(RuntimeError, match="embedding 0 has dimension 0; expected 3"):
            embedding_utils.generate_embedding(backend, "a")

    def test_generate_embedding_raises_when_backend_returns_wrong_count(self):
        backend = MagicMock()
        backend.dimension = 3
        backend.encode_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with pytest.raises(RuntimeError, match="returned 2 vectors for 1 input text"):
            embedding_utils.generate_embedding(backend, "a")

    def test_generate_embedding_uses_query_encoder_when_requested(self):
        backend = MagicMock()
        backend.dimension = 2
        backend.encode_query.return_value = [[0.1, 0.2]]

        result = embedding_utils.generate_embedding(backend, "a", input_type="query")

        assert result == [0.1, 0.2]
        backend.encode_query.assert_called_once_with(["a"])
        backend.encode_documents.assert_not_called()


class TestEmbeddingsBatchLengthGuarantee:
    def test_raises_when_backend_returns_fewer_embeddings(self):
        # Regression for #1037: backends that silently truncate must not pass
        # through — `zip(extracted_facts, embeddings)` would otherwise drop
        # facts and break unit_id alignment downstream.
        backend = MagicMock()
        backend.encode_documents.return_value = [[0.1, 0.2]]  # only 1 vector for 3 inputs

        with pytest.raises(RuntimeError, match="returned 1 vectors for 3 input texts"):
            asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b", "c"]))

    def test_raises_when_backend_returns_more_embeddings(self):
        backend = MagicMock()
        backend.encode_documents.return_value = [[0.1], [0.2], [0.3]]

        with pytest.raises(RuntimeError, match="returned 3 vectors for 2 input texts"):
            asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b"]))

    def test_passes_through_aligned_embeddings(self):
        backend = MagicMock()
        backend.dimension = 1
        backend.encode_documents.return_value = [[0.1], [0.2]]

        result = asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b"]))

        assert result == [[0.1], [0.2]]

    def test_raises_when_backend_returns_empty_embedding_vector(self):
        backend = MagicMock()
        backend.dimension = 3
        backend.encode_documents.return_value = [[0.1, 0.2, 0.3], []]

        with pytest.raises(RuntimeError, match="embedding 1 has dimension 0; expected 3"):
            asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b"]))

    def test_raises_when_backend_returns_wrong_embedding_dimension(self):
        backend = MagicMock()
        backend.dimension = 3
        backend.encode_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5]]

        with pytest.raises(RuntimeError, match="embedding 1 has dimension 2; expected 3"):
            asyncio.run(embedding_utils.generate_embeddings_batch(backend, ["a", "b"]))
