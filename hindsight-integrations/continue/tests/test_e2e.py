"""Gated end-to-end test against a real Hindsight server.

Excluded from the deterministic PR-CI bucket (``-m 'not requires_real_llm'``).
Run on its own with a live Hindsight instance::

    HINDSIGHT_API_URL=http://localhost:8888 \
    HINDSIGHT_CONTINUE_BANK_ID=continue-e2e \
        uv run pytest tests -v -m requires_real_llm

It drives the adapter through Continue's exact HTTP contract — no mocks — and
confirms a retained memory comes back as a Continue context item.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from hindsight_client import Hindsight

from hindsight_continue import build_context_items, configure

requires_real_llm = pytest.mark.requires_real_llm


@requires_real_llm
def test_retain_then_recall_through_adapter():
    api_url = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")
    api_key = os.environ.get("HINDSIGHT_API_KEY")
    bank_id = f"continue-e2e-{uuid.uuid4().hex[:8]}"

    client_kwargs = {"base_url": api_url}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = Hindsight(**client_kwargs)
    client.create_bank(bank_id=bank_id)
    try:
        client.retain(bank_id=bank_id, content="The deploy command for this project is `make ship`.")
        # Hindsight processes retained content asynchronously; give it a moment.
        time.sleep(5)

        configure(hindsight_api_url=api_url, api_key=api_key, bank_id=bank_id)
        items = build_context_items(
            {"query": "how do I deploy?", "fullInput": "how do I deploy?", "options": {}},
        )

        assert items, "expected at least one context item from a real recall"
        assert "make ship" in items[0].content.lower()
    finally:
        try:
            client.delete_bank(bank_id=bank_id)
        except Exception:
            pass
        client.close()
