"""
Test that retain helpers forward async and idempotency options.

Prevents silent regressions where retain_async is accepted in the signature
but dropped before reaching the API request (similar to the bug fixed in #709),
or where operation_id is unavailable through the high-level client surface.
"""

import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_client import Hindsight

OPERATION_ID = "123e4567-e89b-12d3-a456-426614174000"


def _make_client():
    return Hindsight(base_url="http://localhost:8888")


def _captured_request(client):
    return client._memory_api.retain_memories.call_args.args[1]


class _LegacyBatchOverrides(Hindsight):
    """Models subclasses written before operation_id was added."""

    def retain_batch(
        self,
        bank_id,
        items,
        document_id=None,
        document_tags=None,
        retain_async=False,
    ):
        return MagicMock()

    async def aretain_batch(
        self,
        bank_id,
        items,
        document_id=None,
        document_tags=None,
        retain_async=False,
    ):
        return MagicMock()


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


async def test_aretain_forwards_operation_id():
    """aretain() should forward a caller-supplied idempotency key."""
    client = _make_client()
    client.aretain_batch = AsyncMock()

    await client.aretain("bank", "content", retain_async=True, operation_id=OPERATION_ID)
    assert client.aretain_batch.call_args.kwargs["operation_id"] == OPERATION_ID


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


def test_retain_forwards_operation_id():
    """retain() should forward a caller-supplied idempotency key."""
    client = _make_client()
    client.retain_batch = MagicMock()

    client.retain("bank", "content", retain_async=True, operation_id=OPERATION_ID)
    assert client.retain_batch.call_args.kwargs["operation_id"] == OPERATION_ID


def test_retain_batch_forwards_operation_id():
    """retain_batch() should forward operation_id to the async implementation."""
    client = _make_client()
    client.aretain_batch = AsyncMock()

    client.retain_batch("bank", [{"content": "content"}], retain_async=True, operation_id=OPERATION_ID)
    assert client.aretain_batch.call_args.kwargs["operation_id"] == OPERATION_ID


async def test_aretain_batch_sets_operation_id_on_request():
    """aretain_batch() should put a supplied UUID on the generated request."""
    client = _make_client()
    client._memory_api.retain_memories = AsyncMock()

    await client.aretain_batch("bank", [{"content": "content"}], retain_async=True, operation_id=OPERATION_ID)

    request = _captured_request(client)
    assert request.operation_id == OPERATION_ID
    assert request.to_dict()["operation_id"] == OPERATION_ID


async def test_aretain_batch_omits_operation_id_by_default():
    """The legacy request shape should not gain an operation_id null field."""
    client = _make_client()
    client._memory_api.retain_memories = AsyncMock()

    await client.aretain_batch("bank", [{"content": "content"}])

    assert "operation_id" not in _captured_request(client).to_dict()


async def test_aretain_batch_omits_operation_id_for_sync_retain():
    """A supplied idempotency key should stay off synchronous requests."""
    client = _make_client()
    client._memory_api.retain_memories = AsyncMock()

    await client.aretain_batch("bank", [{"content": "content"}], operation_id=OPERATION_ID)

    assert "operation_id" not in _captured_request(client).to_dict()


def test_retain_preserves_legacy_batch_override_for_default_call():
    """Default retain() calls should not pass new kwargs to old overrides."""
    client = _LegacyBatchOverrides(base_url="http://localhost:8888")

    Hindsight.retain(client, "bank", "content")


def test_retain_batch_preserves_legacy_async_override_for_default_call():
    """Default retain_batch() calls should keep old async overrides callable."""
    client = _LegacyBatchOverrides(base_url="http://localhost:8888")

    Hindsight.retain_batch(client, "bank", [{"content": "content"}])


async def test_aretain_preserves_legacy_batch_override_for_default_call():
    """Default aretain() calls should not pass new kwargs to old overrides."""
    client = _LegacyBatchOverrides(base_url="http://localhost:8888")

    await Hindsight.aretain(client, "bank", "content")


def test_retain_warns_once_when_operation_id_dropped_on_sync():
    """A supplied operation_id on a sync retain should warn exactly once.

    Runs the full retain -> retain_batch -> aretain_batch delegation so a
    duplicated warning across the chain would be caught.
    """
    client = _make_client()
    client._memory_api.retain_memories = AsyncMock()

    with pytest.warns(UserWarning, match="operation_id is ignored for synchronous retain") as records:
        client.retain("bank", "content", operation_id=OPERATION_ID)

    assert len(records) == 1


async def test_aretain_batch_warns_when_operation_id_dropped_on_sync():
    """A supplied operation_id on a sync aretain_batch should warn."""
    client = _make_client()
    client._memory_api.retain_memories = AsyncMock()

    with pytest.warns(UserWarning, match="operation_id is ignored for synchronous retain"):
        await client.aretain_batch("bank", [{"content": "content"}], operation_id=OPERATION_ID)


async def test_aretain_does_not_warn_when_operation_id_used_on_async():
    """A supplied operation_id on an async retain must not warn."""
    client = _make_client()
    client.aretain_batch = AsyncMock()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await client.aretain("bank", "content", retain_async=True, operation_id=OPERATION_ID)


def test_retain_does_not_warn_by_default():
    """A plain sync retain without operation_id must not warn."""
    client = _make_client()
    client.retain_batch = MagicMock()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client.retain("bank", "content")
