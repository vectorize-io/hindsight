"""
Test that RetainRequest correctly serializes the async field.

Regression test for a bug where the client passed async_=True (invalid kwarg)
instead of var_async=True, causing async mode to be silently ignored.
"""

from hindsight_client_api.models.memory_item import MemoryItem
from hindsight_client_api.models.retain_request import RetainRequest


def _make_item():
    return MemoryItem(content="test content")


def test_retain_request_async_true_serialized():
    """var_async=True must appear as 'async': True in the serialized dict."""
    req = RetainRequest(items=[_make_item()], var_async=True)
    d = req.to_dict()
    assert d["async"] is True


def test_retain_request_async_false_serialized():
    """var_async=False (default) must appear as 'async': False."""
    req = RetainRequest(items=[_make_item()], var_async=False)
    d = req.to_dict()
    assert d["async"] is False


def test_retain_request_default_is_sync():
    """Omitting var_async should default to synchronous (async=False)."""
    req = RetainRequest(items=[_make_item()])
    d = req.to_dict()
    assert d["async"] is False


def test_retain_request_async_json_roundtrip():
    """async=True must survive a JSON serialization roundtrip."""
    req = RetainRequest(items=[_make_item()], var_async=True)
    json_str = req.to_json()
    restored = RetainRequest.from_json(json_str)
    assert restored.var_async is True
