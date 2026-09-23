"""HTTP API support for authored mental-model content."""

import uuid

import pytest


@pytest.mark.asyncio
async def test_create_mental_model_with_content_skips_refresh(api_client):
    bank_id = f"test-mm-content-create-{uuid.uuid4().hex[:8]}"
    await api_client.put(f"/v1/default/banks/{bank_id}", json={})
    authored = "## Preferences\n\n- Prefer typed APIs\n"

    try:
        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/mental-models",
            json={
                "id": "team-prefs",
                "name": "Team Preferences",
                "source_query": "What are the team's preferences?",
                "content": authored,
                "trigger": {"refresh_after_consolidation": False},
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mental_model_id"] == "team-prefs"
        assert body["operation_id"] is None

        # The point of authored content: nothing is queued to overwrite it. A null
        # operation_id could also come from a response that simply forgot to fill it
        # in, so check the queue itself.
        operations = await api_client.get(
            f"/v1/default/banks/{bank_id}/operations",
            params={"type": "refresh_mental_model"},
        )
        assert operations.status_code == 200, operations.text
        assert operations.json()["total"] == 0, "authored content must not schedule a create-time refresh"

        get_response = await api_client.get(
            f"/v1/default/banks/{bank_id}/mental-models/team-prefs",
            params={"detail": "content"},
        )
        assert get_response.status_code == 200, get_response.text
        model = get_response.json()
        assert model["content"] != "", "authored content must not be left as the empty placeholder"
        assert "Prefer typed APIs" in model["content"]
    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}")


@pytest.mark.asyncio
async def test_create_mental_model_without_content_still_schedules_refresh(api_client):
    bank_id = f"test-mm-content-async-{uuid.uuid4().hex[:8]}"
    await api_client.put(f"/v1/default/banks/{bank_id}", json={})

    try:
        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/mental-models",
            json={
                "name": "Async Preferences",
                "source_query": "What are the team's preferences?",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body.get("mental_model_id")
        assert body.get("operation_id")

        # The complement of the authored-content case: omitting content still queues
        # the create-time refresh that generates it.
        operations = await api_client.get(
            f"/v1/default/banks/{bank_id}/operations",
            params={"type": "refresh_mental_model"},
        )
        assert operations.status_code == 200, operations.text
        assert operations.json()["total"] == 1, "omitting content must schedule the create-time refresh"

        get_response = await api_client.get(
            f"/v1/default/banks/{bank_id}/mental-models/{body['mental_model_id']}",
            params={"detail": "content"},
        )
        assert get_response.status_code == 200, get_response.text
        assert get_response.json()["content"] == "", "omitting content leaves the placeholder the refresh fills"
    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}")


@pytest.mark.asyncio
async def test_update_mental_model_with_content(api_client):
    bank_id = f"test-mm-content-update-{uuid.uuid4().hex[:8]}"
    await api_client.put(f"/v1/default/banks/{bank_id}", json={})

    try:
        create_response = await api_client.post(
            f"/v1/default/banks/{bank_id}/mental-models",
            json={
                "id": "team-prefs",
                "name": "Team Preferences",
                "source_query": "What are the team's preferences?",
                "content": "## Preferences\n\n- Prefer typed APIs\n",
                "trigger": {"refresh_after_consolidation": False},
            },
        )
        assert create_response.status_code == 200, create_response.text

        update_response = await api_client.patch(
            f"/v1/default/banks/{bank_id}/mental-models/team-prefs",
            json={"content": "## Preferences\n\n- Prefer typed APIs\n- Prefer small PRs\n"},
        )
        assert update_response.status_code == 200, update_response.text
        updated = update_response.json()
        assert "Prefer small PRs" in updated["content"]
        assert updated["name"] == "Team Preferences"
    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}")


@pytest.mark.asyncio
async def test_create_and_update_reject_blank_content(api_client):
    bank_id = f"test-mm-content-blank-{uuid.uuid4().hex[:8]}"
    await api_client.put(f"/v1/default/banks/{bank_id}", json={})

    try:
        create_response = await api_client.post(
            f"/v1/default/banks/{bank_id}/mental-models",
            json={
                "name": "Blank Content",
                "source_query": "What is blank?",
                "content": "   ",
            },
        )
        assert create_response.status_code == 422, create_response.text

        seeded = await api_client.post(
            f"/v1/default/banks/{bank_id}/mental-models",
            json={
                "id": "seeded",
                "name": "Seeded",
                "source_query": "What is seeded?",
                "content": "seeded content",
                "trigger": {"refresh_after_consolidation": False},
            },
        )
        assert seeded.status_code == 200, seeded.text

        update_response = await api_client.patch(
            f"/v1/default/banks/{bank_id}/mental-models/seeded",
            json={"content": "\n\t"},
        )
        assert update_response.status_code == 422, update_response.text
    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}")
