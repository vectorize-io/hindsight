"""
Upgrade and backwards compatibility tests.

These tests verify that:
1. Data stored in older versions is accessible after upgrade
2. Database migrations run correctly
3. API behavior remains compatible
"""

import logging

import httpx
import pytest

from .version_runner import VersionRunner

logger = logging.getLogger(__name__)

# Version upgrade paths to test
# Format: (old_version, new_version)
UPGRADE_PATHS = [
    ("v0.3.0", "HEAD"),
]


class TestUpgrade:
    """Tests for version upgrades."""

    @pytest.mark.parametrize("old_version,new_version", UPGRADE_PATHS)
    def test_upgrade_preserves_memories(self, db_url, llm_config, unique_bank_id, old_version, new_version):
        """
        Verify memories stored in old version are accessible after upgrade.

        Workflow:
        1. Start old version
        2. Store memories via retain
        3. Verify recall works on old version
        4. Stop old version
        5. Start new version (same database - migrations run)
        6. Verify recall returns same data
        7. Verify reflect works
        """
        bank_id = unique_bank_id

        # Test data to store
        test_memories = [
            {"content": "Alice is a software engineer at TechCorp.", "context": "team introduction"},
            {"content": "Bob manages the infrastructure team and loves Kubernetes.", "context": "team introduction"},
            {"content": "The project deadline is next Friday.", "context": "project planning"},
        ]

        # Phase 1: Store data with old version
        logger.info(f"=== Phase 1: Setting up data with {old_version} ===")

        with VersionRunner(
            old_version,
            db_url,
            port=8891,
            llm_provider=llm_config["provider"],
            llm_api_key=llm_config["api_key"],
            llm_model=llm_config["model"],
        ) as old:
            server = old.start()
            client = httpx.Client(base_url=server.url, timeout=60)

            # Store memories
            resp = client.post(
                f"/v1/default/banks/{bank_id}/memories",
                json={"items": test_memories},
            )
            assert resp.status_code == 200, f"Failed to store memories: {resp.text}"
            result = resp.json()
            assert result["success"] is True
            assert result["items_count"] == len(test_memories)

            # Verify recall works on old version
            resp = client.post(
                f"/v1/default/banks/{bank_id}/memories/recall",
                json={"query": "Who works at TechCorp?"},
            )
            assert resp.status_code == 200, f"Recall failed on old version: {resp.text}"
            old_results = resp.json()["results"]
            assert len(old_results) > 0, "No results from recall on old version"

            # Get stats for comparison
            resp = client.get(f"/v1/default/banks/{bank_id}/stats")
            assert resp.status_code == 200
            old_stats = resp.json()
            logger.info(f"Old version stats: {old_stats}")

            client.close()

        # Phase 2: Verify data with new version
        logger.info(f"=== Phase 2: Verifying data with {new_version} ===")

        with VersionRunner(
            new_version,
            db_url,
            port=8892,
            llm_provider=llm_config["provider"],
            llm_api_key=llm_config["api_key"],
            llm_model=llm_config["model"],
        ) as new:
            server = new.start()
            client = httpx.Client(base_url=server.url, timeout=60)

            # Verify recall returns data
            resp = client.post(
                f"/v1/default/banks/{bank_id}/memories/recall",
                json={"query": "Who works at TechCorp?"},
            )
            assert resp.status_code == 200, f"Recall failed on new version: {resp.text}"
            new_results = resp.json()["results"]
            assert len(new_results) > 0, f"No results from recall after upgrade. Bank: {bank_id}"

            # Verify Alice is found
            found_alice = any("Alice" in r.get("text", "") for r in new_results)
            assert found_alice, f"Alice not found in results after upgrade: {new_results}"

            # Verify reflect works
            resp = client.post(
                f"/v1/default/banks/{bank_id}/reflect",
                json={"query": "Tell me about the team members"},
            )
            assert resp.status_code == 200, f"Reflect failed after upgrade: {resp.text}"
            reflect_result = resp.json()
            assert len(reflect_result.get("text", "")) > 0, "Empty reflect response after upgrade"

            # Verify stats are preserved
            resp = client.get(f"/v1/default/banks/{bank_id}/stats")
            assert resp.status_code == 200
            new_stats = resp.json()
            logger.info(f"New version stats: {new_stats}")

            # Stats should be similar (might have small differences due to re-indexing)
            assert new_stats["total_nodes"] >= old_stats["total_nodes"], (
                f"Lost nodes after upgrade: {old_stats['total_nodes']} -> {new_stats['total_nodes']}"
            )

            # Cleanup - delete test bank
            resp = client.delete(f"/v1/default/banks/{bank_id}")
            assert resp.status_code == 200

            client.close()

    @pytest.mark.parametrize("old_version,new_version", UPGRADE_PATHS)
    def test_upgrade_preserves_documents(self, db_url, llm_config, unique_bank_id, old_version, new_version):
        """
        Verify documents stored in old version are accessible after upgrade.
        """
        bank_id = unique_bank_id
        doc_id = "test-document-001"

        # Phase 1: Store document with old version
        logger.info(f"=== Phase 1: Storing document with {old_version} ===")

        with VersionRunner(
            old_version,
            db_url,
            port=8893,
            llm_provider=llm_config["provider"],
            llm_api_key=llm_config["api_key"],
            llm_model=llm_config["model"],
        ) as old:
            server = old.start()
            client = httpx.Client(base_url=server.url, timeout=60)

            # Store memory with document
            resp = client.post(
                f"/v1/default/banks/{bank_id}/memories",
                json={
                    "items": [
                        {
                            "content": "The quarterly report shows 25% revenue growth.",
                            "context": "Q1 financial review",
                            "document_id": doc_id,
                        }
                    ]
                },
            )
            assert resp.status_code == 200, f"Failed to store document: {resp.text}"

            # Verify document exists
            resp = client.get(f"/v1/default/banks/{bank_id}/documents")
            assert resp.status_code == 200
            docs = resp.json()["items"]
            doc_ids = [d["id"] for d in docs]
            assert doc_id in doc_ids, f"Document not found in old version: {doc_ids}"

            client.close()

        # Phase 2: Verify document with new version
        logger.info(f"=== Phase 2: Verifying document with {new_version} ===")

        with VersionRunner(
            new_version,
            db_url,
            port=8894,
            llm_provider=llm_config["provider"],
            llm_api_key=llm_config["api_key"],
            llm_model=llm_config["model"],
        ) as new:
            server = new.start()
            client = httpx.Client(base_url=server.url, timeout=60)

            # Verify document still exists
            resp = client.get(f"/v1/default/banks/{bank_id}/documents")
            assert resp.status_code == 200
            docs = resp.json()["items"]
            doc_ids = [d["id"] for d in docs]
            assert doc_id in doc_ids, f"Document not found after upgrade: {doc_ids}"

            # Verify document details
            resp = client.get(f"/v1/default/banks/{bank_id}/documents/{doc_id}")
            assert resp.status_code == 200
            doc_info = resp.json()
            assert doc_info["id"] == doc_id
            assert doc_info["memory_unit_count"] > 0

            # Cleanup
            resp = client.delete(f"/v1/default/banks/{bank_id}")
            assert resp.status_code == 200

            client.close()

    @pytest.mark.parametrize("old_version,new_version", UPGRADE_PATHS)
    def test_upgrade_preserves_bank_profile(self, db_url, llm_config, unique_bank_id, old_version, new_version):
        """
        Verify bank profile (disposition) is preserved after upgrade.
        """
        bank_id = unique_bank_id

        # Phase 1: Create bank with custom disposition
        logger.info(f"=== Phase 1: Creating bank profile with {old_version} ===")

        with VersionRunner(
            old_version,
            db_url,
            port=8895,
            llm_provider=llm_config["provider"],
            llm_api_key=llm_config["api_key"],
            llm_model=llm_config["model"],
        ) as old:
            server = old.start()
            client = httpx.Client(base_url=server.url, timeout=60)

            # Create bank by storing a memory
            resp = client.post(
                f"/v1/default/banks/{bank_id}/memories",
                json={"items": [{"content": "Test memory", "context": "test"}]},
            )
            assert resp.status_code == 200

            # Set custom disposition
            resp = client.put(
                f"/v1/default/banks/{bank_id}/profile",
                json={
                    "disposition": {
                        "skepticism": 4,
                        "literalism": 2,
                        "empathy": 5,
                    }
                },
            )
            assert resp.status_code == 200

            # Verify profile
            resp = client.get(f"/v1/default/banks/{bank_id}/profile")
            assert resp.status_code == 200
            old_profile = resp.json()
            assert old_profile["disposition"]["skepticism"] == 4
            assert old_profile["disposition"]["literalism"] == 2
            assert old_profile["disposition"]["empathy"] == 5

            client.close()

        # Phase 2: Verify profile with new version
        logger.info(f"=== Phase 2: Verifying profile with {new_version} ===")

        with VersionRunner(
            new_version,
            db_url,
            port=8896,
            llm_provider=llm_config["provider"],
            llm_api_key=llm_config["api_key"],
            llm_model=llm_config["model"],
        ) as new:
            server = new.start()
            client = httpx.Client(base_url=server.url, timeout=60)

            # Verify profile is preserved
            resp = client.get(f"/v1/default/banks/{bank_id}/profile")
            assert resp.status_code == 200
            new_profile = resp.json()
            assert new_profile["disposition"]["skepticism"] == 4, "Skepticism not preserved"
            assert new_profile["disposition"]["literalism"] == 2, "Literalism not preserved"
            assert new_profile["disposition"]["empathy"] == 5, "Empathy not preserved"

            # Cleanup
            resp = client.delete(f"/v1/default/banks/{bank_id}")
            assert resp.status_code == 200

            client.close()
