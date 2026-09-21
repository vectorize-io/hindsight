#!/usr/bin/env python3
"""
Memory Banks API examples for Hindsight.
Run: python examples/api/memory-banks.py
"""
import os
import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-bank]
client.create_bank(bank_id="my-bank")
# [/docs:create-bank]


# [docs:bank-with-disposition]
client.create_bank(bank_id="architect-bank")
client.update_bank_config(
    "architect-bank",
    reflect_mission="You're a senior software architect - keep track of system designs, "
            "technology decisions, and architectural patterns. Prefer simplicity over cutting-edge.",
    disposition_skepticism=4,   # Questions new technologies
    disposition_literalism=4,   # Focuses on concrete specs
    disposition_empathy=2,      # Prioritizes technical facts
)
# [/docs:bank-with-disposition]


# [docs:bank-background]
client.create_bank(bank_id="my-bank")
client.update_bank_config(
    "my-bank",
    reflect_mission="I am a research assistant specializing in machine learning.",
)
# [/docs:bank-background]


# [docs:bank-mission]
client.create_bank(bank_id="my-bank")
client.update_bank_config(
    "my-bank",
    reflect_mission="You're a senior software architect - keep track of system designs, "
            "technology decisions, and architectural patterns.",
)
# [/docs:bank-mission]


# [docs:bank-support-agent]
client.create_bank(bank_id="support-bank")
client.update_bank_config(
    "support-bank",
    observations_mission="I am a customer support agent. Track customer preferences, "
            "recurring issues, and resolution history to provide consistent, personalized support.",
)
# [/docs:bank-support-agent]


# [docs:update-bank-config]
client.update_bank_config(
    "my-bank",
    retain_mission="Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges.",
    retain_extraction_mode="verbose",
    observations_mission="Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events.",
    disposition_skepticism=4,
    disposition_literalism=4,
    disposition_empathy=2,
)
# [/docs:update-bank-config]


# [docs:get-bank-config]
# Returns resolved config (server defaults merged with bank overrides) and the raw overrides
data = client.get_bank_config("my-bank")
# data["config"]     — full resolved configuration
# data["overrides"]  — only fields overridden at the bank level
# [/docs:get-bank-config]


# [docs:reset-bank-config]
# Remove all bank-level overrides, reverting to server defaults
client.reset_bank_config("my-bank")
# [/docs:reset-bank-config]


# =============================================================================
# Prompt preview and bank transfer (async: the low-level APIs are async-only)
# =============================================================================
import asyncio

TRANSFER_BANKS = ["transfer-py", "transfer-py-copy", "transfer-py-other", "transfer-py-clone"]


async def wait_for(client, bank_id: str, operation_id: str) -> None:
    for _ in range(120):
        status = await client.operations.get_operation_status(bank_id, operation_id)
        if status.status == "completed":
            return
        if status.status in ("failed", "cancelled"):
            raise RuntimeError(f"operation {operation_id} {status.status}: {status.error_message}")
        await asyncio.sleep(1)
    raise TimeoutError(f"operation {operation_id} did not finish")


async def transfer_examples() -> None:
    # A fresh client: the sync calls above ran on another event loop.
    client = Hindsight(base_url=HINDSIGHT_URL)
    for bank_id in TRANSFER_BANKS:
        requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/{bank_id}")
    # Chunks mode stores text verbatim, so no LLM call is needed.
    await client.acreate_bank(bank_id="transfer-py")
    await client.aupdate_bank_config("transfer-py", retain_extraction_mode="chunks")
    await client.aretain_batch(
        "transfer-py",
        [
            {"content": "Alice leads the payments team.", "document_id": "doc-1"},
            {"content": "Bob moved to the Berlin office.", "document_id": "doc-2"},
        ],
    )
    await client.acreate_bank(bank_id="transfer-py-other")

    # [docs:prompts-preview]
    from hindsight_client_api.models import PromptPreviewRequest

    preview = await client.banks.preview_prompt("my-bank", PromptPreviewRequest(operation="retain"))
    for message in preview.messages:
        print(message.role, len(message.blocks))
    # [/docs:prompts-preview]
    assert preview.messages

    # [docs:transfer-export]
    # Whole bank, memories + config, no history.
    # Submits the export, polls the operation, downloads the ZIP.
    archive = await client.aexport_bank("transfer-py")

    # Just the memories
    memories_only = await client.aexport_bank("transfer-py", include_bank_config=False)

    # Specific documents (a document subset carries no bank-level sections).
    # The low-level call only submits; poll the returned operation yourself.
    submission = await client.bank_transfer.export_bank_transfer(
        "transfer-py", document_id=["doc-1", "doc-2"], include_bank_config=False
    )
    # [/docs:transfer-export]
    assert archive[:2] == b"PK" and memories_only[:2] == b"PK"
    await wait_for(client, "transfer-py", submission.operation_id)

    # [docs:transfer-import]
    # Restore a bank under a new id
    operation_id = await client.aimport_bank("transfer-py", archive, target_bank_id="transfer-py-copy")
    # The restore is recorded against the bank in the URL — poll it there
    status = await client.operations.get_operation_status("transfer-py", operation_id)

    # Merge an archive's documents into an existing bank
    submission = await client.bank_transfer.import_bank_transfer(
        "transfer-py-other",
        ("transfer-py.zip", archive),
        mode="merge",
        document_conflict="replace",
    )
    # [/docs:transfer-import]
    await wait_for(client, "transfer-py", operation_id)
    await wait_for(client, "transfer-py-other", submission.operation_id)
    copied = await client.documents.list_documents("transfer-py-copy")
    assert copied.total == 2, copied

    # [docs:clone-bank]
    operation_id = await client.aclone_bank("transfer-py", "transfer-py-clone")
    # The operation is recorded against the source bank
    status = await client.operations.get_operation_status("transfer-py", operation_id)
    # [/docs:clone-bank]
    await wait_for(client, "transfer-py", operation_id)
    cloned = await client.documents.list_documents("transfer-py-clone")
    assert cloned.total == 2, cloned

    # [docs:document-export]
    # Submits the export (whole bank; pass document_ids=[...] to scope it),
    # polls the operation until completed, downloads the archive.
    archive = await client.aexport_documents("transfer-py")
    with open("transfer-py-documents.zip", "wb") as f:
        f.write(archive)
    # [/docs:document-export]
    assert archive[:2] == b"PK"

    # [docs:document-import]
    with open("transfer-py-documents.zip", "rb") as f:
        submission = await client.document_transfer.import_documents(
            "transfer-py-other", ("transfer-py-documents.zip", f.read()), on_conflict="replace"
        )

    status = await client.operations.get_operation_status("transfer-py-other", submission.operation_id)
    # status.result_metadata -> {"documents_imported": 3, "facts_imported": 42, "observations_imported": 5, ...}
    # [/docs:document-import]
    await wait_for(client, "transfer-py-other", submission.operation_id)
    os.remove("transfer-py-documents.zip")

    for bank_id in TRANSFER_BANKS:
        requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/{bank_id}")
    await client.aclose()


asyncio.run(transfer_examples())


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/architect-bank")

print("memory-banks.py: All examples passed")
