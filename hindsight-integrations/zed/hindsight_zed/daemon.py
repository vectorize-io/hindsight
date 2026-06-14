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


def do_recall(thread: ZedThread, client: HindsightClient, config: ZedConfig, state: DaemonState) -> None:
    """Recall memory for a thread and rewrite its project's instruction block.

    Runs eagerly on every change (we want injected memory as fresh as Zed lets
    us), unlike retain which is debounced until the conversation goes idle.
    """
    if not (config.auto_recall and thread.messages):
        return
    project = _project_dir(thread)
    bank_id = bank_id_for_thread_paths(thread.folder_paths, config)
    if not bank_id or project is None:
        return
    query = compose_recall_query(thread.messages, config.recall_max_query_chars)
    if not query:
        return
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


def do_retain(thread: ZedThread, client: HindsightClient, config: ZedConfig, state: DaemonState) -> None:
    """Retain a thread's transcript (idempotent — skips if already at this revision)."""
    if not (config.auto_retain and state.needs_retain(thread.id, thread.updated_at)):
        return
    bank_id = bank_id_for_thread_paths(thread.folder_paths, config)
    if not bank_id:
        return
    transcript = format_transcript(thread)
    if not transcript.strip():
        return
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


def process_thread(thread: ZedThread, client: HindsightClient, config: ZedConfig, state: DaemonState) -> None:
    """Recall then retain a thread immediately (no debounce).

    Convenience for one-shot use; the live daemon debounces retain via
    :class:`RetainDebouncer` so it captures a settled conversation once.
    """
    do_recall(thread, client, config, state)
    do_retain(thread, client, config, state)


class RetainDebouncer:
    """Defers a thread's retain until its ``updated_at`` has been idle.

    Zed has no "conversation finished" event, so we approximate it: each time a
    thread changes we (re)start its timer; once it's been quiet for
    ``idle_seconds`` it becomes due for retain. This collapses a multi-turn
    conversation into one retain and avoids capturing mid-stream snapshots.
    """

    def __init__(self, idle_seconds: float, clock=time.monotonic):
        self.idle_seconds = idle_seconds
        self._clock = clock
        # thread_id -> (latest thread object, updated_at, last_change_time)
        self._pending: dict[str, tuple[ZedThread, str, float]] = {}

    def note(self, thread: ZedThread) -> None:
        """Record a thread sighting, resetting its idle timer if it advanced."""
        prev = self._pending.get(thread.id)
        if prev is None or prev[1] != thread.updated_at:
            self._pending[thread.id] = (thread, thread.updated_at, self._clock())

    def due(self) -> list[ZedThread]:
        """Return (and drop) threads that have been idle ≥ ``idle_seconds``."""
        now = self._clock()
        ready = [(tid, th) for tid, (th, _, t) in self._pending.items() if now - t >= self.idle_seconds]
        for tid, _ in ready:
            del self._pending[tid]
        return [th for _, th in ready]


def poll_once(
    db_path: Path,
    client: HindsightClient,
    config: ZedConfig,
    state: DaemonState,
    since: Optional[str],
    debouncer: RetainDebouncer,
) -> Optional[str]:
    """One poll: recall changed threads eagerly, retain ones that have gone idle.

    Returns the new high-water mark.
    """
    threads = read_threads(db_path, since=since)
    high = since
    for thread in sorted(threads, key=lambda t: t.updated_at):
        do_recall(thread, client, config, state)
        debouncer.note(thread)
        if high is None or thread.updated_at > high:
            high = thread.updated_at
    # Retain any conversation that has settled (idle past the window).
    for thread in debouncer.due():
        do_retain(thread, client, config, state)
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
    debouncer = RetainDebouncer(config.retain_idle_seconds)
    # Start from the newest already-retained revision so we don't reprocess the
    # entire backlog on first run (recall would still refresh on the next turn).
    since: Optional[str] = max(state.retained.values(), default=None)

    logger.info("hindsight-zed daemon started (db=%s, api=%s)", db_path, config.hindsight_api_url)
    while True:
        try:
            since = poll_once(db_path, client, config, state, since, debouncer)
        except Exception as e:  # never let one bad poll kill the daemon
            logger.debug("poll error: %s", e)
        time.sleep(config.poll_interval)


if __name__ == "__main__":
    run()
