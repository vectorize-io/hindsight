"""Wait for the LLM trace recorder's background writes, so tests read settled rows."""

from hindsight_api import MemoryEngine


async def drain_traces(mem: MemoryEngine) -> None:
    """Wait for the bank's background operations and their trace writes to land.

    record_llm_call schedules each INSERT as a detached asyncio task tracked by
    the recorder's ``PendingWrites``, so an HTTP call can return before its trace
    rows exist. A test that reads the trace list twice (or compares two reads)
    must drain first, or a late row lands between the reads. Loop a few times
    because consolidation's attach_memory_ids can spawn a follow-up write after
    the first drain.
    """
    await mem.wait_for_background_tasks()
    await mem._llm_recorder.flush()
    writes = mem._llm_recorder._writes
    for _ in range(10):
        if not writes.in_flight():
            break
        await writes.drain_all(timeout=30)
