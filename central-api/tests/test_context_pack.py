"""Tests for context-pack builder."""

import json

from app.memory.context_pack import ContextPack


def test_context_pack_creation():
    """Test ContextPack creation and serialization."""
    pack = ContextPack(
        tenant_id="test",
        content={"key": "value"},
        version=1,
        tag="test",
    )
    assert pack.tenant_id == "test"
    assert pack.version == 1
    assert pack.tag == "test"
    assert pack.content_hash
    assert pack.created_at


def test_context_pack_hash_deterministic():
    """Test context-pack hash is deterministic."""
    content = {"a": 1, "b": 2}
    pack1 = ContextPack("test", content, version=1, tag="t1")
    pack2 = ContextPack("test", content, version=1, tag="t2")
    assert pack1.content_hash == pack2.content_hash


def test_context_pack_to_dict():
    """Test ContextPack serialization to dict."""
    content = {"key": "value"}
    pack = ContextPack("test", content, version=1, tag="tag1")
    d = pack.to_dict()
    assert d["tenant_id"] == "test"
    assert d["version"] == 1
    assert d["tag"] == "tag1"
    assert d["content"] == content


def test_context_pack_to_json():
    """Test ContextPack serialization to JSON."""
    pack = ContextPack("test", {"key": "value"}, version=1, tag="tag1")
    j = pack.to_json()
    parsed = json.loads(j)
    assert parsed["tenant_id"] == "test"
    assert parsed["version"] == 1
