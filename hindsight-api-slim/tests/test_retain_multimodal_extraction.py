"""The image actually reaches the extraction call, in position.

Phase 1's tests prove the bytes are stored; these prove they come back out and
land in the prompt beside the prose that introduces them. That round trip is the
feature — storing an image nothing ever looks at would be worse than useless.

Deterministic throughout: the MockLLM records the exact messages it was handed,
so the assertions are on message structure rather than on what a model made of
the picture. What a real vision model extracts is a separate, non-deterministic
question, judged in test_multimodal_extraction_llm.py.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _image_block(data: bytes = PNG_BYTES) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(data).decode()},
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


async def _retain(client, bank_id: str, content, **fields):
    return await client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": content, **fields}], "async": False},
    )


def _retain_messages(memory) -> list[list[dict]]:
    """Every extraction call's message list, from the mock provider's record."""
    provider = memory._retain_llm_config._provider_impl
    return [call["messages"] for call in provider.get_mock_calls() if call["scope"] == "retain_extract_facts"]


@pytest.mark.asyncio
async def test_the_image_reaches_the_model_between_its_surrounding_prose(api_client, memory):
    bank_id = f"vis-{uuid.uuid4().hex[:8]}"
    memory._retain_llm_config._provider_impl.clear_mock_calls()

    response = await _retain(
        api_client,
        bank_id,
        [
            _text_block("To reset the VPN, click the button shown:"),
            _image_block(),
            _text_block("...then reconnect."),
        ],
        document_id="vpn",
    )
    assert response.status_code == 200, response.text

    messages = _retain_messages(memory)
    assert messages, "extraction never ran"
    user_content = messages[0][-1]["content"]

    assert isinstance(user_content, list), "the user message stayed a plain string; the image was dropped"
    kinds = [part["type"] for part in user_content]
    assert kinds == ["text", "image_url", "text"]
    assert "click the button shown" in user_content[0]["text"]
    assert "then reconnect" in user_content[2]["text"]


@pytest.mark.asyncio
async def test_the_bytes_in_the_prompt_are_the_bytes_that_were_retained(api_client, memory):
    """A full round trip: request -> content-addressed storage -> extraction prompt."""
    bank_id = f"vis-{uuid.uuid4().hex[:8]}"
    memory._retain_llm_config._provider_impl.clear_mock_calls()

    assert (
        await _retain(api_client, bank_id, [_text_block("see:"), _image_block()], document_id="d")
    ).status_code == 200

    user_content = _retain_messages(memory)[0][-1]["content"]
    image_part = next(part for part in user_content if part["type"] == "image_url")
    payload = image_part["image_url"]["url"].split(",", 1)[1]

    assert base64.b64decode(payload) == PNG_BYTES
    assert compute_attachment_hash(base64.b64decode(payload)) == compute_attachment_hash(PNG_BYTES)


@pytest.mark.asyncio
async def test_a_text_only_retain_still_sends_a_plain_string(api_client, memory):
    """The multimodal path must not perturb ordinary retains."""
    bank_id = f"vis-{uuid.uuid4().hex[:8]}"
    memory._retain_llm_config._provider_impl.clear_mock_calls()

    assert (await _retain(api_client, bank_id, "Alice joined the AI team", document_id="t")).status_code == 200

    assert isinstance(_retain_messages(memory)[0][-1]["content"], str)


@pytest.mark.asyncio
async def test_a_retain_llm_without_vision_is_refused_before_anything_is_written(api_client, memory, monkeypatch):
    """Failing loudly beats a document that looks retained with its images gone."""
    bank_id = f"vis-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(type(memory._retain_llm_config), "supports_vision", lambda self: False)

    response = await _retain(api_client, bank_id, [_text_block("see:"), _image_block()], document_id="d")

    assert response.status_code == 422
    assert "cannot read images" in response.json()["detail"]

    # Nothing was stored: not the blob row, not the document.
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM attachments WHERE bank_id = $1", bank_id) == 0
        assert await conn.fetchval("SELECT count(*) FROM documents WHERE bank_id = $1", bank_id) == 0


@pytest.mark.asyncio
async def test_an_llm_of_unknown_vision_capability_is_also_refused(api_client, memory, monkeypatch):
    """A gateway's unknown model must not silently swallow images."""
    bank_id = f"vis-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(type(memory._retain_llm_config), "supports_vision", lambda self: None)

    response = await _retain(api_client, bank_id, [_image_block()], document_id="d")

    assert response.status_code == 422
    assert "HINDSIGHT_API_LLM_VISION=true" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_text_only_retain_is_unaffected_by_a_non_vision_llm(api_client, memory, monkeypatch):
    """The gate applies to images, not to retain in general."""
    bank_id = f"vis-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(type(memory._retain_llm_config), "supports_vision", lambda self: False)

    assert (await _retain(api_client, bank_id, "plain text", document_id="t")).status_code == 200
