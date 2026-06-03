"""Narrator resolution + injection (issue #1680).

The "Narrator: {name}" line in fact extraction is stamped into the who-dimension
of every first-person fact and the observations derived from them. When the bank
``name`` is just the bank_id (the auto-create default), that string is a routing
key, not a speaker, and priming extraction with it pollutes stored fact text.

These are pure unit tests — no LLM, no DB. They pin the suppression decision and
that the Narrator line is present/absent accordingly.
"""

from datetime import datetime

from hindsight_api.engine.retain.fact_extraction import _build_user_message
from hindsight_api.engine.retain.orchestrator import _resolve_narrator


class TestResolveNarrator:
    def test_suppressed_when_name_equals_bank_id(self):
        """Auto-create default (name == bank_id) → no narrator, routing key not injected."""
        bank_id = "my-agent::channel-456::user-789"
        assert _resolve_narrator(bank_id, bank_id) is None

    def test_explicit_name_passes_through(self):
        """A human-readable name decoupled from bank_id is used as the narrator."""
        assert _resolve_narrator("Aria", "my-agent::channel-456::user-789") == "Aria"

    def test_short_name_distinct_from_bank_id(self):
        assert _resolve_narrator("Aria", "Aria-bank-1") == "Aria"


class TestNarratorInjection:
    def _msg(self, agent_name, context="agent log"):
        return _build_user_message(
            chunk="I shipped the fix.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2024, 6, 1),
            context=context,
            metadata=None,
            agent_name=agent_name,
        )

    def test_no_narrator_line_when_suppressed(self):
        """agent_name=None (the suppressed case) → no Narrator line, no leaked string."""
        msg = self._msg(None)
        assert "Narrator:" not in msg

    def test_narrator_line_present_for_named_agent(self):
        msg = self._msg("Aria")
        assert "Narrator: Aria" in msg

    def test_context_precedence_clause_only_when_context_set(self):
        """The 'Context above takes precedence' clause appears only with a context."""
        with_context = self._msg("Aria", context="chat with a customer")
        assert "Context above takes precedence" in with_context

        without_context = self._msg("Aria", context="")
        assert "Narrator: Aria" in without_context  # base narrator still present
        assert "Context above takes precedence" not in without_context

    def test_routing_key_never_reaches_prompt_via_resolution(self):
        """End-to-end of the fix: resolve then build → routing key absent from prompt."""
        bank_id = "my-agent::channel-456::user-789"
        msg = self._msg(_resolve_narrator(bank_id, bank_id))
        assert bank_id not in msg
        assert "Narrator:" not in msg
