"""The maintained wrapper forwards multimodal content blocks unaltered.

`content` widened from `str` to `str | list[ContentBlock]` so an image can sit
inline where it appears. The wrapper is what most consumers call, so a wrapper
that still coerced or dropped the list form would strip images for every Python
user while the generated SDK happily supported them.

The TypeScript wrapper has the matching test in
`tests/retain_content_blocks.test.ts` — these two surfaces must stay in step.
"""

from unittest.mock import MagicMock

from hindsight_client import Hindsight

IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
}


def _capture_retain(monkeypatch, client, captured):
    async def fake_retain(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(success=True)

    monkeypatch.setattr(client._memory_api, "retain_memories", fake_retain)


def test_retain_forwards_a_block_list_verbatim(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_retain(monkeypatch, client, captured)

    content = [
        {"type": "text", "text": "click the button shown:"},
        IMAGE_BLOCK,
        {"type": "text", "text": "...then reconnect."},
    ]

    client.retain("test-bank", content)

    # `content` is a generated anyOf wrapper; the payload is its actual_instance.
    sent = captured["request"].items[0].content.actual_instance
    assert sent == content, "the wrapper altered or dropped the content blocks"


def test_retain_still_forwards_a_plain_string(monkeypatch):
    """Widening the type must not disturb every existing caller."""
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_retain(monkeypatch, client, captured)

    client.retain("test-bank", "Alice joined the AI team")

    assert captured["request"].items[0].content.actual_instance == "Alice joined the AI team"


def test_retain_batch_forwards_block_lists(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_retain(monkeypatch, client, captured)

    client.retain_batch(
        "test-bank",
        [
            {"content": [{"type": "text", "text": "see:"}, IMAGE_BLOCK]},
            {"content": "plain text item"},
        ],
    )

    items = captured["request"].items
    assert items[0].content.actual_instance == [{"type": "text", "text": "see:"}, IMAGE_BLOCK]
    assert items[1].content.actual_instance == "plain text item"
