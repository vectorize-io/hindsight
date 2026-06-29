"""End-to-end curator test through the real async pipeline (no direct curate call).

Creates a 'people' folder with a mission, retains a document, and lets the real
triggers run (folder-create curation + retain → consolidation → curation). Asserts
the resulting memory, observation, and the curator-built pages.
"""

import uuid

import httpx
import pytest

from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import MemoryEngine

pytestmark = pytest.mark.hs_llm_core


async def test_people_folder_builds_one_page_per_person(memory_real_llm: MemoryEngine, request_context):
    memory = memory_real_llm
    bank_id = f"test-people-{uuid.uuid4().hex[:8]}"
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        base = f"/v1/default/banks/{bank_id}"

        # 1. Folder with a mission (this alone schedules a curation, but there are
        #    no memories yet).
        r = await client.post(
            f"{base}/knowledge-base/folders",
            json={"name": "people", "mission": "track all people"},
        )
        assert r.status_code == 201, r.text

        # 2. Retain a document → extraction + consolidation + curation, all async.
        r = await client.post(
            f"{base}/memories",
            json={"items": [{"content": "marco loves anna", "document_id": "doc1"}]},
        )
        assert r.status_code in (200, 201, 202), r.text
        await memory.wait_for_background_tasks()

        # --- inspect what the pipeline produced -------------------------------
        world = await memory.list_memory_units(bank_id, fact_type="world", request_context=request_context)
        experience = await memory.list_memory_units(bank_id, fact_type="experience", request_context=request_context)
        observations = await memory.list_memory_units(bank_id, fact_type="observation", request_context=request_context)
        tree = (await client.get(f"{base}/knowledge-base/tree")).json()
        people = next(n for n in tree["roots"] if n["name"] == "people")
        pages = people.get("children", [])

        print("\n--- MEMORIES (world) ---")
        for m in world["items"]:
            print("  ", m["text"])
        print("--- MEMORIES (experience) ---")
        for m in experience["items"]:
            print("  ", m["text"])
        print("--- OBSERVATIONS ---")
        for m in observations["items"]:
            print("  ", m["text"])
        print("--- PAGES under 'people' ---")
        page_bodies = {}
        for p in pages:
            doc = (await client.get(f"{base}/knowledge-base/pages/{p['id']}")).json()
            page_bodies[p["name"]] = doc.get("body") or ""
            print(f"   [{p['name']}] managed={p.get('managed')}\n      {page_bodies[p['name']][:160]}")

        # --- assertions -------------------------------------------------------
        source_count = world["total"] + experience["total"]
        assert source_count == 1, f"expected 1 memory, got {source_count}"
        assert observations["total"] == 1, f"expected 1 observation, got {observations['total']}"
        assert len(pages) == 2, f"expected 2 pages (marco, anna), got {len(pages)}: {[p['name'] for p in pages]}"

        joined_names = " ".join(p["name"].lower() for p in pages)
        assert "marco" in joined_names, f"no Marco page: {[p['name'] for p in pages]}"
        assert "anna" in joined_names, f"no Anna page: {[p['name'] for p in pages]}"
        for name, body in page_bodies.items():
            assert body.strip(), f"page '{name}' has empty content"
            assert "Generating content" not in body, f"page '{name}' content never synthesized: {body[:80]!r}"

    await memory.delete_bank(bank_id, request_context=request_context)
