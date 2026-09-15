"""Bank-scoped file storage: the tenant-scoped key prefix, and what deletes the files under it."""

import uuid

import pytest

from hindsight_api.engine.storage import bank_storage_prefix


@pytest.mark.parametrize("bank_id", [".", "..", "a/b", "a%2Fb", "user.name", "ü b"])
def test_every_bank_id_is_one_key_segment_an_object_store_accepts(bank_id):
    import obstore as obs
    from obstore.store import MemoryStore

    prefix = bank_storage_prefix(bank_id)
    segments = prefix.rstrip("/").split("/")
    assert len(segments) == 4 and segments[2] == "banks"
    assert segments[3] not in (".", "..")
    obs.put(MemoryStore(), f"{prefix}f", b"x")  # raises on a path it cannot parse


def test_distinct_bank_ids_never_share_a_prefix():
    ids = [".", "..", "%2E", "a.b", "a%2Eb", "a/b", "a%2Fb"]
    assert len({bank_storage_prefix(b) for b in ids}) == len(ids)


def test_an_empty_bank_id_is_refused():
    with pytest.raises(ValueError):
        bank_storage_prefix("")


@pytest.mark.asyncio
async def test_object_store_delete_prefix_stops_at_the_prefix():
    import obstore as obs
    from obstore.store import MemoryStore

    from hindsight_api.engine.storage.base import delete_object_store_prefix

    store = MemoryStore()
    for key in ("t/banks/a/1", "t/banks/a/x/2", "t/banks/ab/3"):
        obs.put(store, key, b"x")

    assert await delete_object_store_prefix(store, "t/banks/a/") == 2
    assert [meta["path"] for page in obs.list(store) for meta in page] == ["t/banks/ab/3"]


@pytest.mark.asyncio
async def test_postgres_delete_prefix_treats_like_wildcards_literally(memory):
    storage = memory._file_storage
    await storage.store(file_data=b"x", key="t/banks/a%2F_/1")
    await storage.store(file_data=b"x", key="t/banks/aXX2FZ/1")

    assert await storage.delete_prefix("t/banks/a%2F_/") == 1
    assert await storage.exists("t/banks/aXX2FZ/1")
    await storage.delete("t/banks/aXX2FZ/1")


@pytest.mark.asyncio
async def test_deleting_a_document_deletes_its_uploaded_original(api_client, memory):
    bank_id = f"files-{uuid.uuid4().hex[:8]}"
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "Alice joined Acme.", "document_id": "doc"}], "async": False},
    )
    assert response.status_code == 200, response.text
    key = f"{bank_storage_prefix(bank_id)}files/{uuid.uuid4()}/notes.txt"
    await memory._file_storage.store(file_data=b"Alice joined Acme.", key=key)
    # Forge what a file retain leaves behind without running a parser: the key is
    # only reachable through this column, and no API sets it directly.
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        await conn.execute("UPDATE documents SET file_storage_key = $2 WHERE bank_id = $1", bank_id, key)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc")
    assert response.status_code == 200, response.text

    assert not await memory._file_storage.exists(key)


@pytest.mark.asyncio
async def test_deleting_a_bank_deletes_everything_under_its_prefix(api_client, memory):
    bank_id = f"files-{uuid.uuid4().hex[:8]}"
    other_bank = f"{bank_id}-other"
    for bank in (bank_id, other_bank):
        response = await api_client.put(f"/v1/default/banks/{bank}", json={})
        assert response.status_code in (200, 201), response.text
    export_key = f"{bank_storage_prefix(bank_id)}exports/{uuid.uuid4()}/transfer.zip"
    other_key = f"{bank_storage_prefix(other_bank)}exports/{uuid.uuid4()}/transfer.zip"
    for key in (export_key, other_key):
        await memory._file_storage.store(file_data=b"zip", key=key)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}")
    assert response.status_code == 200, response.text

    assert not await memory._file_storage.exists(export_key)
    # Same name plus a suffix: a prefix without its trailing "/" would sweep it too.
    assert await memory._file_storage.exists(other_key)
    await memory._file_storage.delete(other_key)


@pytest.mark.asyncio
async def test_a_file_under_another_tenant_is_not_downloadable_through_a_same_named_bank(api_client, memory):
    """Object stores share one bucket across tenants: authorizing the bank id alone is not enough."""
    bank_id = f"files-{uuid.uuid4().hex[:8]}"
    response = await api_client.put(f"/v1/default/banks/{bank_id}", json={})
    assert response.status_code in (200, 201), response.text
    foreign_key = f"tenants/someone-else/banks/{bank_id}/exports/{uuid.uuid4()}/transfer.zip"
    await memory._file_storage.store(file_data=b"zip", key=foreign_key)

    try:
        response = await api_client.get(f"/v1/default/files/download/{foreign_key}")
        assert response.status_code == 404
    finally:
        await memory._file_storage.delete(foreign_key)
