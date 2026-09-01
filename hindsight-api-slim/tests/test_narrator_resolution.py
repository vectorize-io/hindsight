"""The bank's display ``name`` never becomes the extraction narrator (issues #1680, #3962).

The "Narrator: {name}" line primes fact extraction to read first-person statements
as the narrator's own actions, and that name is stamped into the who-dimension of
every fact it produces — and into the observations later consolidated from them.

Retain used to derive that narrator from ``banks.name``. But ``name`` is a display
label: nothing but the bank selector reads it, and callers set it to whatever
identifies the bank to *them* — a routing key on auto-create (#1680), or a project
label like ``AuditProject_0825`` (#3962). Either way the string ends up verbatim in
stored fact text that never mentioned it. Retain now passes no narrator at all; a
caller who wants to name the speaker says so in the item's ``context``.

``extract_facts_from_text`` still accepts an explicit ``agent_name`` (the dry-run
endpoint's deprecated override), so the injection itself is still exercised here.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.retain.fact_extraction import _build_user_message


class TestNarratorInjection:
    """Pure unit tests of the prompt line — no LLM, no DB."""

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

    def test_no_narrator_line_without_a_name(self):
        """What retain now always passes → no Narrator line, nothing to leak."""
        assert "Narrator:" not in self._msg(None)

    def test_narrator_line_present_for_named_agent(self):
        assert "Narrator: Aria" in self._msg("Aria")

    def test_context_precedence_clause_only_when_context_set(self):
        """The 'Context above takes precedence' clause appears only with a context."""
        with_context = self._msg("Aria", context="chat with a customer")
        assert "Context above takes precedence" in with_context

        without_context = self._msg("Aria", context="")
        assert "Narrator: Aria" in without_context  # base narrator still present
        assert "Context above takes precedence" not in without_context


@pytest.fixture(autouse=True)
def _fast_retain_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


class _NarratorSpy:
    """Records the ``agent_name`` retain hands to fact extraction."""

    def __init__(self) -> None:
        self.narrators: list[object] = []

    def install(self, monkeypatch) -> None:
        original = fact_extraction.extract_facts_from_contents

        async def _spy(contents, llm_config, agent_name, *args, **kwargs):
            self.narrators.append(agent_name)
            return await original(contents, llm_config, agent_name, *args, **kwargs)

        monkeypatch.setattr(fact_extraction, "extract_facts_from_contents", _spy)


@pytest.mark.asyncio
async def test_retain_passes_no_narrator_for_a_named_bank(memory, request_context, monkeypatch):
    """#3962: a bank named after a project must not prime extraction with that name."""
    bank_id = f"test_narrator_named_bank_{datetime.now(timezone.utc).timestamp()}"
    project_name = "AuditProject_0825"
    spy = _NarratorSpy()
    try:
        await memory.update_bank(bank_id, name=project_name, request_context=request_context)
        spy.install(monkeypatch)

        await memory.retain_async(
            bank_id=bank_id,
            content="Dispatch scheduled a truck for the user for next Wednesday morning.",
            request_context=request_context,
        )

        assert spy.narrators, "fact extraction was never called — the test proves nothing"
        assert all(n is None for n in spy.narrators), f"retain primed extraction with a narrator: {spy.narrators}"

        units = await memory.list_memory_units(bank_id, limit=100, request_context=request_context)
        stored = "\n".join(str(u) for u in units["items"])
        assert project_name not in stored, "the bank's display name leaked into stored memory"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
