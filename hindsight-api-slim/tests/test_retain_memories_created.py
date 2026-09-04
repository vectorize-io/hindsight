"""The synchronous retain response says how many memories it created (#4065).

Extraction finding nothing to remember in a piece of content is a legitimate
outcome — the document is stored, it just has no memory units, and recall can
never reach it. The endpoint answered 200 either way and carried no count, so a
caller could only discover the difference later, by recalling and finding
nothing. Reported over enough content at once that looks like a storage engine
silently dropping writes; it is not.
"""

import uuid

import pytest

pytestmark = pytest.mark.asyncio


def _bank() -> str:
    return f"memories_created_{uuid.uuid4().hex[:8]}"


async def _retain(api_client, bank_id: str, content: str, document_id: str) -> dict:
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": content, "document_id": document_id}], "async": False},
    )
    assert response.status_code == 200, response.text
    return response.json()


async def _document_unit_count(api_client, bank_id: str, document_id: str) -> int:
    response = await api_client.get(f"/v1/default/banks/{bank_id}/documents/{document_id}")
    assert response.status_code == 200, response.text
    return response.json()["memory_unit_count"]


async def test_sync_retain_reports_the_units_it_created(api_client):
    """A retain that stores memories reports the same count the document owns."""
    bank_id = _bank()
    body = await _retain(
        api_client,
        bank_id,
        "Alice is a machine learning researcher at Stanford.",
        "doc-with-facts",
    )

    assert body["memories_created"] >= 1
    assert body["memories_created"] == await _document_unit_count(api_client, bank_id, "doc-with-facts")

    await api_client.delete(f"/v1/default/banks/{bank_id}")


async def test_sync_retain_reports_zero_when_extraction_finds_nothing(api_client, monkeypatch):
    """Nothing extracted is still a 200 — ``memories_created`` is what says so."""
    from hindsight_api.engine.providers import mock_llm

    monkeypatch.setattr(mock_llm.MockLLM, "_build_mock_facts", staticmethod(lambda messages: {"facts": []}))

    bank_id = _bank()
    body = await _retain(api_client, bank_id, "Independent concurrent marker CONCURRENT_INDEP_7.", "doc-no-facts")

    assert body["success"] is True
    assert body["memories_created"] == 0
    # The document is stored regardless — that is the shape the issue observed
    # from outside: a 200, a document, and nothing recallable behind it.
    assert await _document_unit_count(api_client, bank_id, "doc-no-facts") == 0

    await api_client.delete(f"/v1/default/banks/{bank_id}")
