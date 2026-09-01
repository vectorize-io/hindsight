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
from hindsight_api.engine.reflect.prompts import (
    build_final_prompt,
    build_final_system_prompt,
    build_reduce_prompt,
)
from hindsight_api.engine.retain.fact_extraction import (
    _DEFAULT_LANGUAGE_RULE as _RETAIN_DEFAULT_LANGUAGE_RULE,
)
from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema
from hindsight_api.engine.search import bm25_term_selection as bm25_mod
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


@pytest.mark.parametrize("mode", ["concise", "verbose", "verbatim", "custom"])
def test_retain_directive_replaces_source_language_rule(mode):
    """An explicit output language wins outright: the source-language default is
    dropped rather than left to contradict "translate everything into X".

    Retain's default rule is phrased far more forcefully than the appended
    directive ("STRICTLY FORBIDDEN from translating" vs "Respond exclusively
    in X") and comes first, so leaving both in place makes the model keep
    emitting source-language facts and silently no-ops the setting. Mirrors
    ``test_consolidation_directive_replaces_source_language_rule``.
    """
    config = _baseline_config()
    config.retain_extraction_mode = mode
    config.retain_custom_instructions = "Extract only product mentions." if mode == "custom" else None
    config.llm_output_language = "English"

    prompt, _ = _build_extraction_prompt_and_schema(config)

    assert _RETAIN_DEFAULT_LANGUAGE_RULE not in prompt
    assert output_language_directive("English") in prompt


@pytest.mark.parametrize("mode", ["concise", "verbose", "verbatim", "custom"])
def test_retain_unset_requires_source_language(mode):
    """With no configured language, retain must still be told to keep the output
    in the input's language (#181) - otherwise the all-English extraction prompt
    makes multilingual models drift. Removing the rule outright would regress that,
    so this pins the default half of the mutual exclusion.
    """
    config = _baseline_config()
    config.retain_extraction_mode = mode
    config.retain_custom_instructions = "Extract only product mentions." if mode == "custom" else None
    config.llm_output_language = None

    prompt, _ = _build_extraction_prompt_and_schema(config)

    assert _RETAIN_DEFAULT_LANGUAGE_RULE in prompt
    assert "Respond exclusively in" not in prompt


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
    from hindsight_api.engine.consolidation.prompts import (
        _DEFAULT_LANGUAGE_RULE,
        build_consolidation_system_prompt,
    )

    prompt = build_consolidation_system_prompt(llm_output_language="Chinese")
    assert _DEFAULT_LANGUAGE_RULE not in prompt
    assert output_language_directive("Chinese") in prompt


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
    assert build_final_prompt("q", [], {"name": "Bank"}) == build_final_prompt(
        "q", [], {"name": "Bank"}, llm_output_language=None
    )


def test_reflect_injects_directive():
    """Reflect's directive rides on the USER prompt, not the system prompt.

    It has to be the last thing the model reads: the question and the retrieved data
    follow the system prompt, and a directive stranded behind them loses (#3776 — see
    ``build_final_system_prompt`` for the measurement).
    """
    prompt = build_final_prompt("질문", [], {"name": "Bank"}, llm_output_language="Korean")
    assert "Respond exclusively in Korean" in prompt
    assert prompt.rstrip().endswith("must be in Korean."), "the directive must come last"


def test_reflect_reduce_prompt_injects_directive():
    """The split-synthesis path writes the answer too, so it carries the directive as well.

    ``_forced_final_synthesis`` picks between ``build_final_prompt`` and
    ``build_reduce_prompt`` on whether the retrieved data fits one chunk. A configured
    output language that only worked below that threshold would be the same silent no-op
    in a different disguise.
    """
    prompt = build_reduce_prompt("질문", ["- a claim"], {"name": "Bank"}, llm_output_language="Korean")
    assert "Respond exclusively in Korean" in prompt
    assert prompt.rstrip().endswith("must be in Korean."), "the directive must come last"


def test_reflect_unset_does_not_inject_directive_into_the_user_prompt():
    prompt = build_final_prompt("question", [], {"name": "Bank"}, llm_output_language=None)
    assert "Respond exclusively in" not in prompt


def test_reflect_preserves_mission_alongside_directive():
    system_prompt = build_final_system_prompt(mission="Act as a financial analyst.", llm_output_language="Spanish")
    prompt = build_final_prompt("q", [], {"name": "Bank"}, llm_output_language="Spanish")
    assert "financial analyst" in system_prompt
    assert "Respond exclusively in Spanish" in prompt


def test_reflect_directive_replaces_source_language_rule():
    """Reflect follows retain and consolidation: an explicit output language drops the
    answer-in-the-question's-language default instead of contradicting it.

    The rule defers to "a directive above" and nothing ever put one there, so before this
    it never took precedence and a configured language was silently no-opped. Dropping the
    rule is only half of it — the directive also has to reach the model *after* the
    question; ``test_reflect_injects_directive`` pins that half.
    """
    from hindsight_api.engine.reflect.prompts import _FINAL_LANGUAGE_RULE

    system_prompt = build_final_system_prompt(mission=None, llm_output_language="Korean")
    prompt = build_final_prompt("질문", [], {"name": "Bank"}, llm_output_language="Korean")

    assert _FINAL_LANGUAGE_RULE not in system_prompt
    assert output_language_directive("Korean") not in system_prompt, "the directive belongs on the user prompt"
    assert output_language_directive("Korean") in prompt


def test_reflect_unset_requires_source_language():
    """With no configured language the default rule must still be there — without it
    weaker models drift to English on a non-English question."""
    from hindsight_api.engine.reflect.prompts import _FINAL_LANGUAGE_RULE

    prompt = build_final_system_prompt(mission=None, llm_output_language=None)

    assert _FINAL_LANGUAGE_RULE in prompt
    assert "Respond exclusively in" not in prompt


def test_reflect_agent_loop_directive_replaces_source_language_rule():
    """The done() path — not just forced synthesis — honours the configured language.

    Most reflect answers are written by the tool-calling model under
    ``build_system_prompt_for_tools``; ``build_final_system_prompt`` only reaches the
    model on the forced-synthesis fallback. Fixing the latter alone left a Chinese
    question answered in Chinese on every run that completed normally.
    """
    from hindsight_api.engine.reflect.prompts import (
        _TOOLS_LANGUAGE_RULE,
        build_agent_user_prompt,
        build_system_prompt_for_tools,
    )

    system_prompt = build_system_prompt_for_tools({"name": "Bank"}, llm_output_language="Korean")
    user_prompt = build_agent_user_prompt("질문", llm_output_language="Korean")

    assert _TOOLS_LANGUAGE_RULE not in system_prompt
    assert output_language_directive("Korean") not in system_prompt, "the directive belongs on the user message"
    assert user_prompt == "질문" + output_language_directive("Korean")


def test_reflect_agent_loop_unset_requires_source_language():
    from hindsight_api.engine.reflect.prompts import (
        _TOOLS_LANGUAGE_RULE,
        build_agent_user_prompt,
        build_system_prompt_for_tools,
    )

    system_prompt = build_system_prompt_for_tools({"name": "Bank"}, llm_output_language=None)

    assert _TOOLS_LANGUAGE_RULE in system_prompt
    assert "Respond exclusively in" not in system_prompt
    assert build_agent_user_prompt("question", llm_output_language=None) == "question"


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

    for ext in ("vchord", "pg_search", "pg_textsearch", "pgroonga"):
        assert (
            PostgreSQLDialect().prepare_bm25_text(tokens, query, text_search_extension=ext, max_query_terms=3) == query
        )


def test_postgresql_pgroonga_bm25_arm_uses_pgroonga_tokenize():
    arm = PostgreSQLDialect().build_bm25_arm(
        table="memory_units",
        cols="id, text",
        fact_type="world",
        bank_id_param="$2",
        limit_param="$3",
        text_param="$4",
        text_search_extension="pgroonga",
    )
    assert "pgroonga_tokenize($4, 'tokenizer', 'TokenBigram', 'normalizer', 'NormalizerNFKC150')" in arm
    assert "string_agg(pgroonga_query_escape(elem->>'value'), ' OR ')" in arm


@pytest.mark.asyncio
async def test_combined_retrieval_rejects_config_missing_bm25_cap(monkeypatch):
    """A config object lacking the BM25 cap fields fails loudly instead of defaulting.

    This used to read ``getattr(config, "bm25_max_query_terms", DEFAULT_...)`` and
    silently fall back, on the theory that a config could predate the field. It
    cannot: ``HindsightConfig`` is a dataclass that always defines both fields, so
    only a stub like the one below can produce that state — and when one does reach
    here it is a wrong-config bug, not a legacy config. Silently substituting a
    global default for a resolved value is exactly what made #3584 undiagnosable.
    """

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

    with pytest.raises(AttributeError, match="bm25_max_query_terms"):
        await retrieval_mod.retrieve_semantic_bm25_combined_sql(
            FakeConn(),
            "[0.0]",
            "alpha beta",
            "bank-1",
            ["observation"],
            5,
        )

    # It raised while resolving the cap, so the dialect was never handed a bogus one.
    assert fake_dialect.max_query_terms is None


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
        text_search_extension_pg_search_function_schema="paradedb",
        bm25_max_query_terms=16,
        bm25_selective_terms=selective,
    )
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: config)
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: fake_dialect)
    monkeypatch.setattr(bm25_mod, "get_current_schema", lambda: "public")
    monkeypatch.setattr(bm25_mod, "select_selective_bm25_tokens", fake_select)

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
