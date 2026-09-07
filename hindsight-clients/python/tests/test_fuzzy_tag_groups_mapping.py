"""The maintained wrapper forwards ``tag_groups`` — including fuzzy leaves — to the SDK.

Mirrors the TypeScript wrapper's ``fuzzy_tag_groups_mapping`` regression tests. Two things
are pinned here:

* ``tag_groups`` reaches the request object at all. The wrapper used to import a
  ``RecallRequestTagGroupsInner`` model the generator never emits, so every caller passing
  ``tag_groups`` got a ModuleNotFoundError instead of a filtered recall.
* the leaf's ``resolve`` field survives the dict -> model conversion, so ``fuzzy`` matching
  (#4026) is reachable through the wrapper rather than only over raw HTTP.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_client import Hindsight

FUZZY_GROUP = {"tags": ["typsecript"], "match": "any_strict", "resolve": "fuzzy"}


def _capture(monkeypatch, client, api_attr, method, captured):
    async def fake(bank_id, request_obj, **kwargs):
        captured["bank_id"] = bank_id
        captured["request"] = request_obj
        return MagicMock()

    monkeypatch.setattr(getattr(client, api_attr), method, fake)


@pytest.mark.parametrize(
    "api_attr,method,call",
    [
        ("_memory_api", "recall_memories", "recall"),
        ("_memory_api", "reflect", "reflect"),
    ],
)
def test_wrapper_forwards_fuzzy_tag_groups(monkeypatch, api_attr, method, call):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, api_attr, method, captured)

    getattr(client, call)("bank-1", "what language is the parser in", tag_groups=[FUZZY_GROUP])

    sent = captured["request"].to_dict()["tag_groups"]
    assert sent == [FUZZY_GROUP], "tag_groups must reach the request with `resolve` intact"
