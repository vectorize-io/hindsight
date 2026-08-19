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
    from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt

    prompt = build_consolidation_system_prompt(llm_output_language=None)
    assert "Respond exclusively in" not in prompt


def test_consolidation_unset_requires_source_language():
    """With no configured language, consolidation must still be told to keep each
    observation in the language of its own source facts (#3166) — otherwise the
    all-English prompt makes multilingual models drift to English."""
    from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt

    prompt = build_consolidation_system_prompt(llm_output_language=None)
    assert "## LANGUAGE" in prompt
    assert "language of its own source facts" in prompt
    # The three cases the rule has to settle, not just the happy path.
    assert "Per observation, not per batch" in prompt
    assert "compose the merged observation from scratch in the new facts' language" in prompt
    assert "Proper nouns, identifiers, and units stay verbatim" in prompt


def test_consolidation_injects_directive():
    from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt

    prompt = build_consolidation_system_prompt(llm_output_language="Chinese")
    assert "Respond exclusively in Chinese" in prompt
    assert "Translate any source content into Chinese" in prompt


def test_consolidation_directive_replaces_source_language_rule():
    """An explicit output language wins outright: the source-language default is
    dropped rather than left to contradict "translate everything into X"."""
    from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt

    prompt = build_consolidation_system_prompt(llm_output_language="Chinese")
    assert "## LANGUAGE" not in prompt
    assert "language of its own source facts" not in prompt


def test_consolidation_directive_does_not_break_format_placeholders():
    """build_consolidation_system_prompt appends the language directive and then runs
    str.format() internally (the cached OUTPUT examples use doubled braces). A directive
    that introduced a stray { / } would raise KeyError here — so a successful build with
    a language set is the regression guard."""
    from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt

    prompt = build_consolidation_system_prompt(llm_output_language="Japanese")
    assert "Respond exclusively in Japanese" in prompt


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

    result = await retrieval_mod.retrieve_semantic_bm25_combined_sql(
        FakeConn(),
        "[0.0]",
        "alpha beta",
        "bank-1",
        ["observation"],
        5,
    )

    assert result == {"observation": retrieval_mod.SemanticBm25Result(semantic=[], bm25=[], graph_seeds=None)}
    # A config predating the field falls back to the current default cap.
    assert fake_dialect.max_query_terms == retrieval_mod.DEFAULT_BM25_MAX_QUERY_TERMS


@pytest.mark.parametrize("selective", [True, False])
@pytest.mark.asyncio
async def test_selective_terms_flag_gates_the_pg_stats_lookup(monkeypatch, selective):
    """A long native query consults pg_stats only when bm25_selective_terms is on;
    opting out caps by position without the catalog read."""

    class FakeDialect:
        received_tokens: list[str] | None = None

        def build_semantic_arm(self, **kwargs):
            return "SELECT 'semantic' AS source"

        def build_bm25_arm(self, **kwargs):
            return "SELECT 'bm25' AS source"

        def prepare_bm25_text(self, tokens, query_text, *, text_search_extension="native", max_query_terms=None):
            self.received_tokens = list(tokens)
            return " | ".join(tokens)

    class FakeConn:
        backend_type = "postgresql"

        async def fetch(self, query, *params):
            return []

    selected: list[bool] = []

    async def fake_select(conn, tokens, **kwargs):
        selected.append(True)
        return tokens[:5]

    fake_dialect = FakeDialect()
    config = SimpleNamespace(
        semantic_min_similarity=0.0,
        bm25_min_score=0.0,
        text_search_extension="native",
        text_search_extension_native_language="english",
        bm25_max_query_terms=16,
        bm25_selective_terms=selective,
    )
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: config)
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: fake_dialect)
    monkeypatch.setattr(retrieval_mod, "get_current_schema", lambda: "public")
    monkeypatch.setattr(retrieval_mod, "select_selective_bm25_tokens", fake_select)

    long_query = " ".join(f"term{i}" for i in range(20))  # 20 tokens, over the cap
    await retrieval_mod.retrieve_semantic_bm25_combined_sql(
        FakeConn(), "[0.0]", long_query, "bank-1", ["observation"], 5
    )

    if selective:
        assert selected == [True]
        assert len(fake_dialect.received_tokens) == 5  # selection applied
    else:
        assert selected == []  # pg_stats never consulted
        assert len(fake_dialect.received_tokens) == 20  # capping left to prepare_bm25_text


@pytest.mark.asyncio
async def test_combined_retrieval_reuses_raw_semantic_pool_for_graph_seeds(monkeypatch):
    class FakeDialect:
        def build_semantic_arm(self, **kwargs):
            return "SELECT 'semantic' AS source"

    class FakeConn:
        backend_type = "postgresql"

        async def fetch(self, query, *params):
            return [
                {"id": "best", "text": "best", "fact_type": "world", "source": "semantic", "similarity": 0.9},
                {"id": "graph", "text": "graph", "fact_type": "world", "source": "semantic", "similarity": 0.6},
                {"id": "weak", "text": "weak", "fact_type": "world", "source": "semantic", "similarity": 0.2},
            ]

    config = SimpleNamespace(semantic_min_similarity=0.1, bm25_min_score=0.0)
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: config)
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: FakeDialect())

    result = await retrieval_mod.retrieve_semantic_bm25_combined_sql(
        FakeConn(),
        "[0.0]",
        "",
        "bank-1",
        ["world"],
        1,
        graph_seed_min_similarity=0.5,
    )

    assert [candidate.id for candidate in result["world"].semantic] == ["best"]
    assert [candidate.id for candidate in result["world"].graph_seeds or []] == ["best", "graph"]


@pytest.mark.asyncio
async def test_combined_retrieval_keeps_graph_query_when_semantic_threshold_is_stricter(monkeypatch):
    class FakeDialect:
        def build_semantic_arm(self, **kwargs):
            return "SELECT 'semantic' AS source"

    class FakeConn:
        backend_type = "postgresql"

        async def fetch(self, query, *params):
            return []

    config = SimpleNamespace(semantic_min_similarity=0.7, bm25_min_score=0.0)
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: config)
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: FakeDialect())

    result = await retrieval_mod.retrieve_semantic_bm25_combined_sql(
        FakeConn(),
        "[0.0]",
        "",
        "bank-1",
        ["world"],
        10,
        graph_seed_min_similarity=0.3,
    )

    assert result["world"].graph_seeds is None
