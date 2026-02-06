"""Unit tests for HindsightStorage."""

from unittest.mock import MagicMock, patch

from hindsight_crewai import HindsightStorage, configure, reset_config
from hindsight_crewai.errors import HindsightError

import pytest


def _passthrough(fn, *args, **kwargs):
    """Replace call_sync with direct call for testing."""
    return fn(*args, **kwargs)


class TestHindsightStorage:
    def setup_method(self):
        reset_config()
        configure(hindsight_api_url="http://localhost:8888")

    def teardown_method(self):
        reset_config()

    def _make_storage(self, **kwargs):
        """Create a storage instance with a mocked client."""
        storage = HindsightStorage(bank_id="test-bank", **kwargs)
        mock_client = MagicMock()
        storage._local.client = mock_client
        storage._created_banks.add("test-bank")
        return storage, mock_client

    def _make_recall_result(self, text="Memory text", type_="world", **kwargs):
        """Create a mock RecallResult."""
        r = MagicMock()
        r.text = text
        r.type = type_
        r.context = kwargs.get("context")
        r.occurred_start = kwargs.get("occurred_start")
        r.document_id = kwargs.get("document_id")
        r.metadata = kwargs.get("metadata")
        r.tags = kwargs.get("tags")
        return r

    # --- save() tests ---

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_save_calls_retain(self, _mock_cs):
        storage, mock_client = self._make_storage()

        storage.save("Task output text", metadata={"task": "research"}, agent="Researcher")

        mock_client.retain.assert_called_once()
        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["content"] == "Task output text"
        assert call_kwargs["metadata"]["source"] == "crewai"
        assert call_kwargs["metadata"]["agent"] == "Researcher"
        assert call_kwargs["metadata"]["task"] == "research"

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_save_stringifies_metadata_values(self, _mock_cs):
        storage, mock_client = self._make_storage()

        storage.save("text", metadata={"count": 42, "active": True})

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["metadata"]["count"] == "42"
        assert call_kwargs["metadata"]["active"] == "True"

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_save_without_metadata(self, _mock_cs):
        storage, mock_client = self._make_storage()

        storage.save("text")

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "crewai"}
        assert call_kwargs["context"] == "crewai:task_output:unknown"

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_save_raises_hindsight_error_on_failure(self, _mock_cs):
        storage, mock_client = self._make_storage()
        mock_client.retain.side_effect = RuntimeError("connection refused")

        with pytest.raises(HindsightError, match="Failed to store memory"):
            storage.save("text")

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_save_passes_tags(self, _mock_cs):
        storage, mock_client = self._make_storage(tags=["env:prod"])

        storage.save("text")

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["tags"] == ["env:prod"]

    # --- search() tests ---

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_search_calls_recall(self, _mock_cs):
        storage, mock_client = self._make_storage()

        mock_response = MagicMock()
        mock_response.results = [self._make_recall_result()]
        mock_client.recall.return_value = mock_response

        results = storage.search("programming preferences", limit=5)

        mock_client.recall.assert_called_once()
        assert len(results) == 1
        assert results[0]["context"] == "Memory text"
        assert "score" in results[0]
        assert results[0]["metadata"]["type"] == "world"

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_search_respects_limit(self, _mock_cs):
        storage, mock_client = self._make_storage()

        mock_response = MagicMock()
        mock_response.results = [self._make_recall_result(text=f"Memory {i}") for i in range(10)]
        mock_client.recall.return_value = mock_response

        results = storage.search("test", limit=3)
        assert len(results) == 3

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_search_returns_empty_on_no_results(self, _mock_cs):
        storage, mock_client = self._make_storage()

        mock_response = MagicMock()
        mock_response.results = []
        mock_client.recall.return_value = mock_response

        results = storage.search("nonexistent topic")
        assert results == []

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_search_includes_rich_metadata(self, _mock_cs):
        storage, mock_client = self._make_storage()

        r = self._make_recall_result(
            context="conversation",
            occurred_start="2024-01-01",
            document_id="doc-1",
            metadata={"key": "value"},
            tags=["tag1"],
        )
        mock_response = MagicMock()
        mock_response.results = [r]
        mock_client.recall.return_value = mock_response

        results = storage.search("test")
        meta = results[0]["metadata"]
        assert meta["source_context"] == "conversation"
        assert meta["occurred_start"] == "2024-01-01"
        assert meta["document_id"] == "doc-1"
        assert meta["key"] == "value"
        assert meta["tags"] == ["tag1"]

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_search_passes_recall_tags(self, _mock_cs):
        storage, mock_client = self._make_storage(recall_tags=["scope:global"], recall_tags_match="all")

        mock_response = MagicMock()
        mock_response.results = []
        mock_client.recall.return_value = mock_response

        storage.search("test")

        call_kwargs = mock_client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_search_raises_hindsight_error_on_failure(self, _mock_cs):
        storage, mock_client = self._make_storage()
        mock_client.recall.side_effect = RuntimeError("timeout")

        with pytest.raises(HindsightError, match="Failed to search memories"):
            storage.search("test")

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_search_synthetic_scores_descend(self, _mock_cs):
        storage, mock_client = self._make_storage()

        mock_response = MagicMock()
        mock_response.results = [self._make_recall_result(text=f"Memory {i}") for i in range(5)]
        mock_client.recall.return_value = mock_response

        results = storage.search("test", limit=5, score_threshold=0.0)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 1.0

    # --- reset() tests ---

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_reset_deletes_bank(self, _mock_cs):
        storage, mock_client = self._make_storage()

        storage.reset()

        mock_client.delete_bank.assert_called_once_with("test-bank")

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_reset_is_best_effort(self, _mock_cs):
        storage, mock_client = self._make_storage()
        mock_client.delete_bank.side_effect = RuntimeError("not found")

        # Should not raise
        storage.reset()

    # --- per-agent banks ---

    def test_per_agent_banks_resolves_bank_id(self):
        storage = HindsightStorage(bank_id="crew", per_agent_banks=True)
        assert storage._resolve_bank_id("Researcher") == "crew-researcher"
        assert storage._resolve_bank_id("Data Analyst") == "crew-data-analyst"
        assert storage._resolve_bank_id(None) == "crew"

    def test_custom_bank_resolver(self):
        resolver = lambda base, agent: f"custom-{agent}" if agent else base
        storage = HindsightStorage(bank_id="crew", bank_resolver=resolver)
        assert storage._resolve_bank_id("Alice") == "custom-Alice"
        assert storage._resolve_bank_id(None) == "crew"

    @patch("hindsight_crewai.storage.call_sync", side_effect=_passthrough)
    def test_save_uses_per_agent_bank(self, _mock_cs):
        storage = HindsightStorage(bank_id="crew", per_agent_banks=True)
        mock_client = MagicMock()
        storage._local.client = mock_client
        storage._created_banks.add("crew-researcher")

        storage.save("output", agent="Researcher")

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["bank_id"] == "crew-researcher"

    # --- config resolution ---

    def test_constructor_overrides_config(self):
        configure(budget="low", max_tokens=1024)
        storage = HindsightStorage(bank_id="test", budget="high", max_tokens=8192)
        assert storage._budget == "high"
        assert storage._max_tokens == 8192

    def test_falls_back_to_config(self):
        configure(budget="high", max_tokens=2048, verbose=True)
        storage = HindsightStorage(bank_id="test")
        assert storage._budget == "high"
        assert storage._max_tokens == 2048
        assert storage._verbose is True

    def test_falls_back_to_defaults_without_config(self):
        reset_config()
        storage = HindsightStorage(bank_id="test")
        assert storage._api_url == "http://localhost:8888"
        assert storage._budget == "mid"
        assert storage._max_tokens == 4096
