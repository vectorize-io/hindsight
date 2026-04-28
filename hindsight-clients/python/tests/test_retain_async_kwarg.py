"""
Test that retain() and aretain() forward retain_async to the batch methods.

Prevents silent regressions where retain_async is accepted in the signature
but dropped before reaching the API request (similar to the bug fixed in #709).
"""

from unittest.mock import AsyncMock, MagicMock

from hindsight_client import Hindsight


def _make_client():
    return Hindsight(base_url="http://localhost:8888")


async def test_aretain_forwards_retain_async_default():
    """aretain() should forward retain_async=False by default."""
    client = _make_client()
    client.aretain_batch = AsyncMock()

    await client.aretain("bank", "content")
    assert client.aretain_batch.call_args.kwargs["retain_async"] is False


async def test_aretain_forwards_retain_async_true():
    """aretain(retain_async=True) should forward it to aretain_batch()."""
    client = _make_client()
    client.aretain_batch = AsyncMock()

    await client.aretain("bank", "content", retain_async=True)
    assert client.aretain_batch.call_args.kwargs["retain_async"] is True


def test_retain_forwards_retain_async_default():
    """retain() should forward retain_async=False by default."""
    client = _make_client()
    client.retain_batch = MagicMock()

    client.retain("bank", "content")
    assert client.retain_batch.call_args.kwargs["retain_async"] is False


def test_retain_forwards_retain_async_true():
    """retain(retain_async=True) should forward it to retain_batch()."""
    client = _make_client()
    client.retain_batch = MagicMock()

    client.retain("bank", "content", retain_async=True)
    assert client.retain_batch.call_args.kwargs["retain_async"] is True
