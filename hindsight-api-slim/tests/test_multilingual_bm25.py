"""Tests for multilingual BM25 + LLM output language wiring.

Covers:
- ``HINDSIGHT_API_LLM_OUTPUT_LANGUAGE`` directive injection across all three
  LLM-generating pipelines: retain (fact extraction), consolidation
  (observations), and reflect (response synthesis).
- The new alembic migration's structural shape (chains off the right head).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from hindsight_api.engine.consolidation.prompts import build_batch_consolidation_prompt
from hindsight_api.engine.prompt_utils import output_language_directive
from hindsight_api.engine.reflect.prompts import build_final_system_prompt
from hindsight_api.engine.retain.fact_extraction import _build_extraction_prompt_and_schema


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
