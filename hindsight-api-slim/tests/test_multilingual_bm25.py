"""Tests for multilingual BM25 + LLM output language wiring.

Covers:
- ``HINDSIGHT_API_LLM_OUTPUT_LANGUAGE`` directive injection across all three
  LLM-generating pipelines: retain (fact extraction), consolidation
  (observations), and reflect (response synthesis).
- The new alembic migration's structural shape (chains off the right head).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.consolidation.prompts import build_batch_consolidation_prompt
from hindsight_api.engine.prompt_utils import output_language_directive
from hindsight_api.engine.reflect.prompts import build_final_system_prompt
from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema
from hindsight_api.engine.search import retrieval as retrieval_mod
from hindsight_api.engine.search.retrieval import tokenize_query
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect


def _baseline_config() -> MagicMock:
    """Mock config with the minimal fields needed by _build_extraction_prompt_and_schema."""
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    return config


# ---------------------------------------------------------------------------
# Shared directive helper
# ---------------------------------------------------------------------------


def test_output_language_directive_empty_when_unset():
    assert output_language_directive(None) == ""
    assert output_language_directive("") == ""


def test_output_language_directive_mentions_language_three_times():
    directive = output_language_directive("Japanese")
    # All three references are needed so the LLM applies the constraint to
    # source translation, fact text, and the final response equally.
    assert directive.count("Japanese") == 3
    assert "Respond exclusively in Japanese" in directive
    assert "Translate any source content into Japanese" in directive


# ---------------------------------------------------------------------------
# Retain (fact extraction)
# ---------------------------------------------------------------------------


def test_retain_unset_does_not_inject_directive():
    config = _baseline_config()
    config.llm_output_language = None

    prompt, _ = _build_extraction_prompt_and_schema(config)

    assert "Respond exclusively in" not in prompt
    assert "Translate any source content" not in prompt


def test_retain_injects_directive():
    config = _baseline_config()
    config.llm_output_language = "Japanese"

    prompt, _ = _build_extraction_prompt_and_schema(config)

    assert "Respond exclusively in Japanese" in prompt
    assert "Translate any source content into Japanese" in prompt


def test_retain_directive_appears_after_base_prompt():
    """The directive is appended at the end so mode-specific guidelines are
    still respected — the LLM reads them, then applies the language constraint."""
    config = _baseline_config()
    config.llm_output_language = "Spanish"

    prompt, _ = _build_extraction_prompt_and_schema(config)

    directive_idx = prompt.find("Respond exclusively in Spanish")
    assert directive_idx > 0
    # A non-trivial extraction prompt body precedes the directive.
    assert directive_idx > 100


def test_retain_works_with_custom_mode():
    """Custom extraction mode + llm_output_language: directive must still appear."""
    config = _baseline_config()
    config.retain_extraction_mode = "custom"
    config.retain_custom_instructions = "Extract only product mentions."
    config.llm_output_language = "French"

    prompt, _ = _build_extraction_prompt_and_schema(config)

    assert "Extract only product mentions." in prompt
    assert "Respond exclusively in French" in prompt


# ---------------------------------------------------------------------------
# Consolidation (observations)
# ---------------------------------------------------------------------------


def test_consolidation_unset_does_not_inject_directive():
    prompt = build_batch_consolidation_prompt(llm_output_language=None)
    assert "Respond exclusively in" not in prompt


def test_consolidation_injects_directive():
    prompt = build_batch_consolidation_prompt(llm_output_language="Chinese")
    assert "Respond exclusively in Chinese" in prompt
    assert "Translate any source content into Chinese" in prompt


def test_consolidation_directive_does_not_break_format_placeholders():
    """The consolidation prompt is later passed through str.format(facts_text=..., observations_text=...).
    The appended directive must not introduce stray { / } that would raise KeyError."""
    prompt = build_batch_consolidation_prompt(llm_output_language="Japanese")
    # str.format must succeed with the expected placeholders.
    prompt.format(facts_text="X", observations_text="Y")


# ---------------------------------------------------------------------------
# Reflect (response synthesis)
# ---------------------------------------------------------------------------


def test_reflect_unset_does_not_inject_directive():
    prompt = build_final_system_prompt(mission=None, llm_output_language=None)
    assert "Respond exclusively in" not in prompt


def test_reflect_injects_directive():
    prompt = build_final_system_prompt(mission=None, llm_output_language="Korean")
    assert "Respond exclusively in Korean" in prompt


def test_reflect_preserves_mission_alongside_directive():
    prompt = build_final_system_prompt(mission="Act as a financial analyst.", llm_output_language="Spanish")
    assert "financial analyst" in prompt
    assert "Respond exclusively in Spanish" in prompt


# ---------------------------------------------------------------------------
# Migration shape regression test
# ---------------------------------------------------------------------------


def test_configurable_bm25_language_migration_chains_off_head():
    """The new migration must descend from the head it was authored against.

    Tests that re-pointing the migration's down_revision wouldn't go
    unnoticed — it would silently break the chain on a fresh DB.
    """
    versions_dir = Path(__file__).resolve().parent.parent / "hindsight_api" / "alembic" / "versions"
    target = versions_dir / "p4q5r6s7t8u9_configurable_bm25_language.py"
    assert target.exists(), "configurable_bm25_language migration file is missing"

    src = target.read_text()
    assert 'revision: str = "p4q5r6s7t8u9"' in src
    assert 'down_revision: str | Sequence[str] | None = "86f7a033d372"' in src


# ---------------------------------------------------------------------------
# BM25 query term cap
# ---------------------------------------------------------------------------


def test_postgresql_native_bm25_caps_raw_terms_preserving_order():
    query = "Alpha beta alpha, gamma delta beta epsilon"
    tokens = tokenize_query(query)

    assert PostgreSQLDialect().prepare_bm25_text(tokens, query, max_query_terms=3) == "alpha | beta | alpha"


def test_postgresql_native_bm25_zero_cap_keeps_existing_unlimited_behavior():
    query = "Alpha beta alpha"
    tokens = tokenize_query(query)

    assert PostgreSQLDialect().prepare_bm25_text(tokens, query, max_query_terms=0) == "alpha | beta | alpha"


def test_postgresql_extension_bm25_keeps_raw_query_text():
    query = "Alpha beta alpha, gamma delta beta epsilon"
    tokens = tokenize_query(query)

    assert (
        PostgreSQLDialect().prepare_bm25_text(tokens, query, text_search_extension="vchord", max_query_terms=3) == query
    )


@pytest.mark.asyncio
async def test_combined_retrieval_uses_default_bm25_cap_for_legacy_config(monkeypatch):
    class FakeDialect:
        max_query_terms: int | None = None

        def build_semantic_arm(self, **kwargs):
            return "SELECT 'semantic' AS source"

        def build_bm25_arm(self, **kwargs):
            return "SELECT 'bm25' AS source"

        def prepare_bm25_text(self, tokens, query_text, *, text_search_extension="native", max_query_terms=None):
            self.max_query_terms = max_query_terms
            return " | ".join(tokens)

    class FakeConn:
        backend_type = "postgresql"

        async def fetch(self, query, *params):
            return []

    fake_dialect = FakeDialect()
    legacy_config = SimpleNamespace(
        semantic_min_similarity=0.0,
        bm25_min_score=0.0,
        text_search_extension="native",
        text_search_extension_native_language="english",
    )
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: legacy_config)
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: fake_dialect)

    result = await retrieval_mod.retrieve_semantic_bm25_combined(
        FakeConn(),
        "[0.0]",
        "alpha beta",
        "bank-1",
        ["observation"],
        5,
    )

    assert result == {"observation": ([], [])}
    assert fake_dialect.max_query_terms == 0
