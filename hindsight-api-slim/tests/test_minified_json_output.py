"""The JSON-output extraction/consolidation prompts can request MINIFIED JSON.

Models pretty-print structured output by default (line breaks + indentation),
which is ~40% of the output tokens. Since these outputs are machine-parsed
(json.loads / hand-parsed) and never shown to a user, the prompts can append a
formatting-only directive telling the model to emit compact JSON. Gated by the
global HINDSIGHT_API_MINIFY_LLM_JSON_OUTPUT flag, which is OPT-IN (default off).
These tests assert the directive is absent by default, present when enabled, and
that prompt assembly still renders either way.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.consolidation.prompts import (
    build_batch_consolidation_prompt,
    build_consolidation_system_prompt,
)
from hindsight_api.engine.prompt_utils import minified_json_directive
from hindsight_api.engine.reflect.prompts import STRUCTURED_DELTA_SYSTEM_PROMPT
from hindsight_api.engine.retain.fact_extraction import (
    _build_extraction_prompt_and_schema,
)


def _config(mode: str) -> MagicMock:
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    return config


@pytest.fixture
def minify_on(monkeypatch):
    """Opt into the minify flag (default is off) and restore the cache after."""
    monkeypatch.setenv("HINDSIGHT_API_MINIFY_LLM_JSON_OUTPUT", "true")
    clear_config_cache()  # re-read env
    yield
    clear_config_cache()  # let following tests re-read default (off)


@pytest.fixture
def default_config():
    """Ensure config reflects the (default-off) env, not a stale cached value."""
    clear_config_cache()
    yield
    clear_config_cache()


class TestMinifyDirectiveHelper:
    def test_off_by_default_returns_empty(self, default_config):
        assert minified_json_directive("keep X the same") == ""

    def test_enabled_produces_directive(self, minify_on):
        d = minified_json_directive("keep X the same")
        assert "minified" in d.lower()
        assert "keep X the same" in d
        # brace-free so it survives downstream str.format()
        assert "{" not in d and "}" not in d
        d.format()  # must not raise

    def test_extra_caveat_appended(self, minify_on):
        d = minified_json_directive("keep X the same", extra=" ESCAPE_NOTE")
        assert "ESCAPE_NOTE" in d


class TestRetainMinified:
    def test_off_by_default(self, default_config):
        prompt, _ = _build_extraction_prompt_and_schema(_config("concise"))
        assert "minified" not in prompt.lower()

    def test_every_llm_extraction_mode_when_enabled(self, minify_on):
        # "chunks" skips the LLM entirely (no prompt); the rest are LLM modes.
        for mode in ("concise", "verbose", "custom", "verbatim"):
            prompt, _ = _build_extraction_prompt_and_schema(_config(mode))
            assert "minified" in prompt.lower(), f"mode {mode} missing minify directive"

    def test_directive_is_formatting_only(self, minify_on):
        prompt, _ = _build_extraction_prompt_and_schema(_config("concise"))
        assert "same facts" in prompt.lower()


class TestConsolidationMinified:
    def test_off_by_default(self, default_config):
        assert "minified" not in build_consolidation_system_prompt().lower()

    def test_system_prompt_when_enabled_and_renders(self, minify_on):
        # build_consolidation_system_prompt calls template.format() internally;
        # a stray brace in the directive would raise here.
        rendered = build_consolidation_system_prompt()
        assert "minified" in rendered.lower()
        # brace-escaped examples still unescape correctly ({{ -> {)
        assert '{"creates": []' in rendered

    def test_batch_prompt_when_enabled_and_renders(self, minify_on):
        prompt = build_batch_consolidation_prompt(observations_mission="track facts")
        rendered = prompt.format(facts_text="<facts>", observations_text="<obs>")
        assert "minified" in rendered.lower()


class TestStructuredDeltaMinified:
    def test_helper_directive_used_by_delta(self, minify_on):
        # The mm-delta directive is appended at the call site via the helper;
        # verify the helper carries the ops-preserving phrase + escaping caveat.
        d = minified_json_directive(
            "produce exactly the same operations",
            extra=" Newlines inside a text/items string value must still be escaped as \\n.",
        )
        assert "minified" in d.lower()
        assert "same operations" in d
        assert "escaped as \\n" in d

    def test_delta_constant_keeps_string_escaping_rules(self):
        # The minify directive lives at the call site, not the constant; the
        # constant's in-string escaping rules must remain intact.
        assert "JSON STRING RULES" in STRUCTURED_DELTA_SYSTEM_PROMPT
