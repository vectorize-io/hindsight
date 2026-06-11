"""The Hindsight Zed daemon.

Zed exposes no AI-conversation hook, so a small background process supplies the
automation:

  - **Auto-recall (passive injection):** when a thread is updated, recall memory
    against its latest user message and rewrite the fenced ``<!-- HINDSIGHT -->``
    block in that project's instruction file. Zed always includes that file in
    the agent's context, so memory "just shows up" on the next turn.
  - **Auto-retain (passive capture):** when a thread advances, store its
    transcript into the project's Hindsight bank.

Both sides poll the same ``threads.db`` so a single pass over new/changed
threads drives recall and retain together.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .bank import bank_id_for_thread_paths
from .client import HindsightClient
from .config import ZedConfig, load_config
from .content import compose_recall_query, format_memory_block, format_transcript
from .rules_file import write_memory_block
from .state import DaemonState
from .threads_db import ZedThread, default_threads_db_path, read_threads

logger = logging.getLogger("hindsight_zed.daemon")


def _project_dir(thread: ZedThread) -> Optional[Path]:
    """Return the first existing project folder for a thread, if any."""
    for path in thread.folder_paths:
        p = Path(path)
        if p.is_dir():
            return p
    return None


def process_thread(
    thread: ZedThread,
    client: HindsightClient,
    config: ZedConfig,
    state: DaemonState,
) -> None:
    """Run recall (inject) and retain (capture) for a single updated thread."""
    project = _project_dir(thread)
    bank_id = bank_id_for_thread_paths(thread.folder_paths, config)
    if not bank_id or project is None:
        # No on-disk project to scope a bank or write a rules file to — skip.
        return

    # ── Auto-recall → rewrite the project's memory block ──────────────────────
    if config.auto_recall and thread.messages:
        query = compose_recall_query(thread.messages, config.recall_max_query_chars)
        if query:
            try:
                resp = client.recall(
                    bank_id,
                    query,
                    max_tokens=config.recall_max_tokens,
                    budget=config.recall_budget,
                    types=config.recall_types,
                )
                block = format_memory_block(resp.get("results", []))
                write_memory_block(project, block, preamble=config.recall_preamble)
            except Exception as e:
                logger.debug("recall failed for bank %s: %s", bank_id, e)

    # ── Auto-retain → store the transcript ────────────────────────────────────
    if config.auto_retain and state.needs_retain(thread.id, thread.updated_at):
        transcript = format_transcript(thread)
        if transcript.strip():
            try:
                client.retain(
                    bank_id,
                    transcript,
                    document_id=f"zed-thread-{thread.id}",
                    context=config.retain_context,
                    tags=config.retain_tags,
                    metadata={"source": "zed", "thread_id": thread.id},
                )
                state.mark_retained(thread.id, thread.updated_at)
            except Exception as e:
                logger.debug("retain failed for bank %s: %s", bank_id, e)


def poll_once(
    db_path: Path,
    client: HindsightClient,
    config: ZedConfig,
    state: DaemonState,
    since: Optional[str],
) -> Optional[str]:
    """Process all threads updated since ``since``. Returns the new high-water mark."""
    threads = read_threads(db_path, since=since)
    if not threads:
        return since
    threads.sort(key=lambda t: t.updated_at)
    high = since
    for thread in threads:
        process_thread(thread, client, config, state)
        if high is None or thread.updated_at > high:
            high = thread.updated_at
    state.save()
    return high


def run(db_path: Optional[Path] = None, config: Optional[ZedConfig] = None) -> None:
    """Run the daemon poll loop forever."""
    config = config or load_config()
    db_path = db_path or default_threads_db_path()
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)

    client = HindsightClient(config.hindsight_api_url, config.hindsight_api_token)
    state = DaemonState.load()
    # Start from the newest already-retained revision so we don't reprocess the
    # entire backlog on first run (recall would still refresh on the next turn).
    since: Optional[str] = max(state.retained.values(), default=None)

    logger.info("hindsight-zed daemon started (db=%s, api=%s)", db_path, config.hindsight_api_url)
    while True:
        try:
            since = poll_once(db_path, client, config, state, since)
        except Exception as e:  # never let one bad poll kill the daemon
            logger.debug("poll error: %s", e)
        time.sleep(config.poll_interval)


if __name__ == "__main__":
    run()
