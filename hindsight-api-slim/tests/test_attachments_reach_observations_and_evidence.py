"""An observation, and the evidence behind a reflect answer, carry the attachments behind them.

Two gaps, one symptom: a knowledge-base article's screenshot stopped reaching the client
once the answer came from somewhere other than the raw fact extracted from it.

* An observation is extracted from no attachment, so it had none — yet reflect prefers
  observations over raw facts. It now shows the attachments of the facts it was
  consolidated from, read through its sources when it is rendered.
* ``based_on`` rebuilt each cited memory from the agent's tool output, which is trimmed of
  provenance the agent never reads. It now carries the memory as stored — document, chunk,
  tags, metadata — and its attachments.

The fact with the image comes from a `chunks`-mode retain, so its attachment edge is
deterministic (see ``test_attachment_read_surfaces``). The observation is seeded straight
into ``memory_units``, because what is under test is how it is read, not how consolidation
chose its sources.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.reflect.models import ReflectAgentResult, ToolCall
from hindsight_api.engine.retain import embedding_processing
from hindsight_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id

# Seeds an observation with an INSERT into `memory_units`, which a store that owns its rows
# never reads. The store-owned path is covered in test_attachments_store_owned.py.
pytestmark = pytest.mark.memory_backend_incompatible

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_ID = short_attachment_id(compute_attachment_hash(PNG_BYTES))
DOCUMENT_ID = "sso-article"
OBSERVATION_TEXT = "SSO is configured on the Admin > Identity > SSO page."


async def _retain(api_client, bank_id: str, content, document_id: str, tags: list[str]) -> None:
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": content, "document_id": document_id, "tags": tags}], "async": False},
    )
    assert response.status_code == 200, response.text


async def _seed_observation(memory, bank_id: str, source_ids: list[str]) -> str:
    observation_id = uuid.uuid4()
    embedding = await embedding_processing.generate_embeddings_batch(memory.embeddings, [OBSERVATION_TEXT])
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, embedding, event_date, source_memory_ids, proof_count,
                created_at, updated_at
            ) VALUES ($1, $2, $3, 'observation', $4::vector, NOW(), $5, $6, NOW(), NOW())
            """,
            observation_id,
            bank_id,
            OBSERVATION_TEXT,
            str(embedding[0]),
            [uuid.UUID(s) for s in source_ids],
            len(source_ids),
        )
    return str(observation_id)


@pytest.fixture
async def bank(api_client, memory):
    """One fact drawn from a screenshot, one plain-text fact, and an observation over each."""
    bank_id = f"obs-att-{uuid.uuid4().hex[:8]}"
    assert (await api_client.put(f"/v1/default/banks/{bank_id}", json={})).status_code == 200
    config = await api_client.patch(
        f"/v1/default/banks/{bank_id}/config", json={"updates": {"retain_extraction_mode": "chunks"}}
    )
    assert config.status_code == 200, config.text
    await _retain(
        api_client,
        bank_id,
        [
            {"type": "text", "text": "Open the SSO settings page shown below."},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG_BYTES).decode()},
            },
        ],
        DOCUMENT_ID,
        ["kb"],
    )
    await _retain(api_client, bank_id, "A customer asked where the SSO page is.", "ticket-1", ["ticket"])

    listed = (await api_client.get(f"/v1/default/banks/{bank_id}/memories/list")).json()["items"]
    # Raw facts only: the mock consolidates on retain, and its observations now carry attachments too.
    facts = [m for m in listed if m["fact_type"] != "observation"]
    image_fact = next(m for m in facts if m.get("attachments"))
    text_fact = next(m for m in facts if m.get("document_id") == "ticket-1")
    return {
        "bank_id": bank_id,
        "image_fact": image_fact["id"],
        "observation": await _seed_observation(memory, bank_id, [image_fact["id"], text_fact["id"]]),
        "text_only_observation": await _seed_observation(memory, bank_id, [text_fact["id"]]),
    }


def _ids(attachments) -> list[str]:
    return [a["id"] for a in attachments or []]


@pytest.mark.asyncio
async def test_an_observation_shows_its_source_facts_attachments_on_list_and_get(api_client, bank):
    base = f"/v1/default/banks/{bank['bank_id']}"

    listed = {m["id"]: m for m in (await api_client.get(f"{base}/memories/list")).json()["items"]}
    detail = (await api_client.get(f"{base}/memories/{bank['observation']}")).json()

    assert _ids(listed[bank["observation"]].get("attachments")) == [PNG_ID]
    assert _ids(detail.get("attachments")) == [PNG_ID]
    # Its URL serves the bytes: the handle is the same one the source fact reports.
    assert detail["attachments"][0]["url"].endswith(f"/attachments/{PNG_ID}")


@pytest.mark.asyncio
async def test_an_observation_over_plain_text_facts_shows_none(api_client, bank):
    """The field stays absent, not empty, when no source was drawn from an attachment."""
    base = f"/v1/default/banks/{bank['bank_id']}"

    listed = {m["id"]: m for m in (await api_client.get(f"{base}/memories/list")).json()["items"]}

    assert listed[bank["text_only_observation"]].get("attachments") is None


@pytest.mark.asyncio
async def test_a_recalled_observation_shows_its_source_facts_attachments(api_client, bank):
    response = await api_client.post(
        f"/v1/default/banks/{bank['bank_id']}/memories/recall",
        json={"query": "Where is the SSO page configured?", "types": ["observation"]},
    )

    assert response.status_code == 200, response.text
    results = {r["id"]: r for r in response.json()["results"]}
    assert bank["observation"] in results, "the seeded observation was not recalled"
    assert _ids(results[bank["observation"]].get("attachments")) == [PNG_ID]


@pytest.mark.asyncio
async def test_based_on_carries_each_cited_memory_as_stored_with_its_attachments(api_client, bank, monkeypatch):
    """The agent saw trimmed tool output; the caller gets the memory's provenance and attachments.

    The agent is stubbed with the tool output shape reflect's tools really produce — no
    ``document_id``, ``chunk_id`` or ``metadata`` — so everything asserted beyond that has
    to come from the stored memory.
    """

    async def fake_agent(**kwargs):
        return ReflectAgentResult(
            text="Use the Admin > Identity > SSO page.",
            tool_trace=[
                ToolCall(
                    tool="recall",
                    input={"query": "sso"},
                    output={"memories": [{"id": bank["image_fact"], "text": "SSO page", "fact_type": "world"}]},
                    duration_ms=1,
                ),
                ToolCall(
                    tool="search_observations",
                    input={"query": "sso"},
                    output={
                        "observations": [
                            {"id": bank["observation"], "text": OBSERVATION_TEXT, "fact_type": "observation"}
                        ]
                    },
                    duration_ms=1,
                ),
            ],
            used_memory_ids=[bank["image_fact"]],
            used_observation_ids=[bank["observation"]],
        )

    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_agent)

    response = await api_client.post(
        f"/v1/default/banks/{bank['bank_id']}/reflect",
        json={"query": "Where do I configure SSO?", "include": {"facts": {}}, "exclude_mental_models": True},
    )

    assert response.status_code == 200, response.text
    memories = {m["id"]: m for m in response.json()["based_on"]["memories"]}
    fact = memories[bank["image_fact"]]
    assert fact["document_id"] == DOCUMENT_ID
    assert fact["tags"] == ["kb"]
    assert fact["chunk_id"]
    assert _ids(fact.get("attachments")) == [PNG_ID]
    assert _ids(memories[bank["observation"]].get("attachments")) == [PNG_ID]
