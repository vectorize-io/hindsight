"""Gated end-to-end: real retain -> recall through the wrapper helpers.

Excluded from PR CI (``-m 'not requires_real_llm'``). Run against a live
Hindsight server::

    HINDSIGHT_API_URL=http://localhost:8888 uv run pytest tests -v -m requires_real_llm
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_aider.config import AiderConfig
from hindsight_aider.runner import do_recall, do_retain

requires_real_llm = pytest.mark.requires_real_llm


@requires_real_llm
def test_retain_then_recall_roundtrip(tmp_path):
    api_url = os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888")
    api_token = os.environ.get("HINDSIGHT_API_TOKEN")
    bank = f"aider-e2e-{uuid.uuid4().hex[:8]}"

    kwargs = {"base_url": api_url}
    if api_token:
        kwargs["api_key"] = api_token
    client = Hindsight(**kwargs)
    client.create_bank(bank_id=bank)
    cfg = AiderConfig(
        hindsight_api_url=api_url,
        hindsight_api_token=api_token,
        bank_id=bank,
        memory_filename=str(tmp_path / "mem.md"),
    )
    try:
        # retain a session transcript, then recall it into the memory file
        do_retain(client, cfg, bank, "Session: the deploy command for this project is `make ship`.")
        time.sleep(5)  # async extraction
        memory_path = Path(cfg.memory_filename)
        ok = False
        for _ in range(8):
            if do_recall(client, cfg, bank, "how do I deploy?", memory_path):
                ok = True
                break
            time.sleep(3)
        assert ok, "recall wrote no memory file"
        assert "make ship" in memory_path.read_text().lower()
    finally:
        try:
            client.delete_bank(bank_id=bank)
        except Exception:
            pass
        client.close()
