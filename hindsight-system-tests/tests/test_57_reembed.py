"""Switching embedding models: memories survive, servers on the old model step aside.

An operator moves a schema to a new embedding model with `hindsight-admin reembed` while a
server on the old model keeps running, then brings up a server on the new one and lifts the
guard with `--finish`. The seams this story crosses are the ones no single test sees:

- memories retained by the old server are found by the new one, through vectors the admin
  command rebuilt from their stored text;
- the old server, after the swap, answers 503 instead of writing vectors of the wrong model or
  failing with a 500, and cannot restart on the old model;
- the new server retains and recalls normally, before and after `--finish`.

The schema is unique per run: the command changes its vector width for good.
"""

from __future__ import annotations

import uuid
from contextlib import suppress

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests import run_hindsight_admin, start_hindsight_server, wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

NEW_DIMENSIONS = "256"


async def test_reembed_moves_a_schema_to_a_new_model_while_servers_run(stub_server, llm, tmp_path):
    schema = f"systest_reembed_{uuid.uuid4().hex[:8]}"
    old_env = {"HINDSIGHT_API_DATABASE_SCHEMA": schema}
    new_env = {
        **old_env,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": "stub-embeddings-v2",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS": NEW_DIMENSIONS,
    }
    bank = "reembed-story"
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays the cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    old = start_hindsight_server(stub_url=stub_server.url, log_path=tmp_path / "old.log", extra_env=old_env)
    old_client = Hindsight(base_url=old.url)
    new = None
    try:
        await old_client.aretain(bank_id=bank, content="Alice lives in Berlin.")
        await wait_until_settled(old_client, bank)

        output = run_hindsight_admin(
            ["reembed", "--schema", schema],
            stub_url=stub_server.url,
            log_path=tmp_path / "reembed.log",
            extra_env=new_env,
        )
        assert f"{schema}: swapped" in output

        # The old server is still up, but its vectors no longer fit: it sheds instead of failing.
        with pytest.raises(ApiException) as refused:
            await old_client.aretain(bank_id=bank, content="Alice plays the cello.")
        assert refused.value.status == 503
        with pytest.raises(ApiException) as unanswered:
            await old_client.arecall(bank_id=bank, query="Where does Alice live?")
        assert unanswered.value.status == 503

        # And it cannot come back on the old model.
        with pytest.raises(RuntimeError, match="exited during startup"):
            start_hindsight_server(stub_url=stub_server.url, log_path=tmp_path / "old-again.log", extra_env=old_env)

        new = start_hindsight_server(stub_url=stub_server.url, log_path=tmp_path / "new.log", extra_env=new_env)
        new_client = Hindsight(base_url=new.url)
        try:
            # The memory the old server wrote is found through its rebuilt vector.
            recalled = await new_client.arecall(bank_id=bank, query="Where does Alice live?", types=["world"])
            assert [r.text for r in recalled.results] == ["Alice lives in Berlin | Involving: Alice"]
            assert recalled.results[0].scores.semantic > 0

            # The new server writes vectors while the guard is still up.
            await new_client.aretain(bank_id=bank, content="Alice plays the cello.")
            await wait_until_settled(new_client, bank)

            old.stop()
            finished = run_hindsight_admin(
                ["reembed", "--schema", schema, "--finish"],
                stub_url=stub_server.url,
                log_path=tmp_path / "finish.log",
                extra_env=new_env,
            )
            assert f"{schema}: finished" in finished

            recalled = await new_client.arecall(bank_id=bank, query="What does Alice play?", types=["world"])
            assert [r.text for r in recalled.results] == [
                "Alice plays the cello | Involving: Alice",
                "Alice lives in Berlin | Involving: Alice",
            ]
            assert all(r.scores.semantic > 0 for r in recalled.results)
        finally:
            with suppress(Exception):
                await new_client.banks.delete_bank(bank)
            await new_client.aclose()
    finally:
        await old_client.aclose()
        old.stop()
        if new is not None:
            new.stop()
