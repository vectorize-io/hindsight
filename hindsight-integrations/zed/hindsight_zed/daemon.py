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
import urllib.error
from pathlib import Path
from typing import Optional

from .bank import bank_id_for_thread_paths
from .client import HindsightClient, HindsightHTTPError
from .config import ZedConfig, load_config
from .content import compose_recall_query, format_memory_block, format_transcript
from .rules_file import write_memory_block
from .state import DaemonState
from .threads_db import ZedThread, default_threads_db_path, read_threads

logger = logging.getLogger("hindsight_zed.daemon")

# Connection-level failures that mean "can't reach the server".
_CONNECTION_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, OSError)


def _log_api_error(op: str, bank_id: str, exc: Exception, state: DaemonState) -> None:
    """Log an API failure, escalating auth/connection errors to WARNING.

    A wrong token or unreachable server otherwise fails silently — the user just
    sees no memory in Zed. Those two cases are surfaced at WARNING (once per
    bank, until a later success clears it, so the poll loop doesn't spam the
    log); everything else stays at DEBUG to avoid noise from transient blips.
    """
    if isinstance(exc, HindsightHTTPError) and exc.status_code in (401, 403):
        key = ("auth", bank_id)
        if key not in state.warned:
            logger.warning(
                "Hindsight rejected %s for bank %s (HTTP %s) — check your "
                "HINDSIGHT_API_TOKEN (~/.hindsight/zed.json). Memory will not "
                "work until this is fixed.",
                op,
                bank_id,
                exc.status_code,
            )
            state.warned.add(key)
        return
    if isinstance(exc, _CONNECTION_ERRORS) or (isinstance(exc, HindsightHTTPError) and exc.status_code >= 500):
        key = ("conn", bank_id)
        if key not in state.warned:
            logger.warning(
                "Hindsight %s could not reach the server for bank %s: %s — check the API URL and your network.",
                op,
                bank_id,
                exc,
            )
            state.warned.add(key)
        return
    logger.debug("%s failed for bank %s: %s", op, bank_id, exc)


def _clear_warnings(bank_id: str, state: DaemonState) -> None:
    """A successful call clears prior warnings so a later failure re-surfaces."""
    state.warned.discard(("auth", bank_id))
    state.warned.discard(("conn", bank_id))


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
                _clear_warnings(bank_id, state)
            except Exception as e:
                _log_api_error("recall", bank_id, e, state)

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
                _clear_warnings(bank_id, state)
            except Exception as e:
                _log_api_error("retain", bank_id, e, state)


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
