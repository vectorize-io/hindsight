"""Retain identity isolation — a queued retain must ship the identity and item
config that were authoritative at ENQUEUE time, not whatever the provider holds
when the writer thread eventually drains the queue
(NousResearch/hermes-agent#64499).

``_make_turn_retain_job`` snapshots turns/metadata/lineage/bank_id at enqueue
time, but the writer job still calls ``_build_retain_kwargs`` at run time, which
re-reads mutable per-session attributes (``_retain_tags``,
``_observation_scopes``, ``_retain_source``). A session switch in between stamps
a queued OLD-session retain with the NEW session's item config."""

import threading


def test_queued_retain_keeps_enqueue_time_item_config(provider):
    instance, fake = provider(
        {"retain_tags": ["old-tag"], "retain_source": "old-source", "observation_scopes": "per_tag"}
    )

    # Park the writer exactly between dequeue and job execution so the main
    # thread can mutate provider state while the job is still pending.
    gate = threading.Event()
    released = threading.Event()

    def _gate():
        gate.set()
        released.wait(timeout=5.0)

    instance._retain_queue.put(_gate)
    instance.sync_turn("old-user", "old-assistant")
    gate.wait(timeout=5.0)

    # Mutate the item-config attributes the writer must NOT observe — what an
    # on_session_switch() landing between enqueue and drain does.
    instance._retain_tags = ["new-tag"]
    instance._observation_scopes = "all_combinations"
    instance._retain_source = "new-source"

    released.set()
    instance._retain_queue.join()

    assert len(fake.retains) == 1
    item = fake.retains[0]["items"][0]
    assert item["tags"] == ["old-tag", "session:session-1"]
    assert item["observation_scopes"] == "per_tag"
    assert item["metadata"]["source"] == "old-source"
    instance.shutdown()
