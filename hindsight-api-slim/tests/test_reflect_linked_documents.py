"""Which reflects read the knowledge banks a memory bank links: an interactive one, never a refresh.

A mental model refresh that read the documents would fold them into the bank's own knowledge, so
its recall tool must leave them out, and with them their pictures.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.models import RequestContext
from tests.test_reflect_decision_model_switch import engine  # noqa: F401  (fixture)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "reads_documents"),
    [("reflect", True), ("refresh_mental_model", False), ("dry_run_refresh_mental_model", False)],
)
async def test_only_an_interactive_reflect_reads_linked_documents(engine, monkeypatch, label, reads_documents):  # noqa: F811
    from hindsight_api.engine.reflect.models import ReflectAgentResult

    engine._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(
            return_value=SimpleNamespace(
                llm_gemini_safety_settings=None,
                reflect_use_decision_model=False,
                mental_model_refresh_use_decision_model=False,
            )
        ),
        get_bank_config=AsyncMock(return_value={}),
    )
    recalled: list[tuple[bool, int]] = []

    async def fake_tool_recall(*args, include_documents: bool = False, kb_images: int = 0, **kwargs):
        recalled.append((include_documents, kb_images))
        return {}

    async def fake_run_reflect_agent(**kwargs):
        await kwargs["recall_fn"]("q", 100, 100)
        return ReflectAgentResult(text="ok")

    monkeypatch.setattr("hindsight_api.engine.memory_engine.tool_recall", fake_tool_recall)
    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_run_reflect_agent)
    await engine.reflect_async(
        bank_id="bank-1",
        query="test",
        request_context=RequestContext(),
        exclude_mental_models=True,
        _operation_label=label,
    )
    # The engine fixture's mock LLM reads images, so an interactive reflect asks for the server's cap.
    from hindsight_api.config import get_config

    assert recalled == [(reads_documents, get_config().reflect_kb_search_max_images if reads_documents else 0)]
