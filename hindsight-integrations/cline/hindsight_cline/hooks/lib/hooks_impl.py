"""Hook logic for the four Cline lifecycle events.

Each `main_*` is the entrypoint the matching hook script calls. They never
raise — any failure degrades to a no-op so a memory hiccup can never block
Cline. The handlers are pure-ish (take HookInput + config) and are what the
tests exercise.
"""

import os
from collections.abc import Callable

from . import bank, cline_io, content
from .client import HindsightClient
from .cline_io import RECALL_MIN_CHARS, HookInput
from .config import HindsightClineConfig, debug_log, load_config


def _project_name(hook: HookInput) -> str:
    workspace = hook.workspace_roots[0] if hook.workspace_roots else ""
    return os.path.basename(workspace.rstrip("/")) if workspace else "unknown"


def _render(template: str, hook: HookInput, status: str) -> str:
    """Expand the supported template vars in a tag/metadata string."""
    return (
        str(template)
        .replace("{task_id}", hook.task_id or "")
        .replace("{project}", _project_name(hook))
        .replace("{status}", status)
        .replace("{timestamp}", content.format_current_time())
    )


def _recall_context(hook: HookInput, config: HindsightClineConfig, query: str) -> str:
    """Recall memories for `query` and render the injectable context block."""
    api_url = cline_io.resolve_api_url(config)
    if not api_url:
        return ""
    client = HindsightClient(api_url, config.hindsight_api_token)
    bank_id = bank.derive_bank_id(hook, config)
    bank.ensure_bank_mission(client, bank_id, config, debug_fn=lambda *a: debug_log(config, *a))
    resp = client.recall(
        bank_id,
        query,
        max_tokens=config.recall_max_tokens,
        budget=config.recall_budget,
        types=config.recall_types,
        timeout=config.recall_timeout,
    )
    results = resp.get("results", []) if isinstance(resp, dict) else []
    memories = content.format_memories(results)
    if not memories:
        return ""
    preamble = config.recall_prompt_preamble
    return (
        "<hindsight_memories>\n"
        f"{preamble}\n"
        f"Current time - {content.format_current_time()}\n\n"
        f"{memories}\n"
        "</hindsight_memories>"
    )


def handle_user_prompt_submit(hook: HookInput, config: HindsightClineConfig) -> str:
    """Accumulate the prompt (for retain) and recall relevant memories."""
    if config.auto_retain:
        content.append_turn(hook.task_id, "user", hook.prompt)
    if not config.auto_recall or len(hook.prompt.strip()) < RECALL_MIN_CHARS:
        return ""
    messages = content.read_transcript(hook.task_id)
    query = content.compose_recall_query(hook.prompt, messages, config.recall_context_turns)
    query = content.truncate_recall_query(query, hook.prompt, config.recall_max_query_chars)
    return _recall_context(hook, config, query)


def handle_task_start(hook: HookInput, config: HindsightClineConfig) -> str:
    """Seed the transcript with the task description and recall kickoff context."""
    if config.auto_retain and hook.task:
        content.append_turn(hook.task_id, "user", hook.task)
    if not config.auto_recall:
        return ""
    query = (hook.task or "").strip()
    if len(query) < RECALL_MIN_CHARS:
        return ""
    return _recall_context(hook, config, query)


def handle_retain(hook: HookInput, config: HindsightClineConfig, status: str) -> None:
    """Retain the accumulated task transcript, then clear it."""
    if not config.auto_retain:
        return
    # The completion hook's `task` field carries the final summary — record it.
    if hook.task:
        content.append_turn(hook.task_id, "assistant", hook.task)
    messages = content.read_transcript(hook.task_id)
    transcript = content.format_retention(messages)
    if not transcript.strip():
        return
    api_url = cline_io.resolve_api_url(config)
    if not api_url:
        return
    client = HindsightClient(api_url, config.hindsight_api_token)
    bank_id = bank.derive_bank_id(hook, config)
    bank.ensure_bank_mission(client, bank_id, config, debug_fn=lambda *a: debug_log(config, *a))

    metadata = {"task_id": hook.task_id, "project": _project_name(hook), "status": status}
    for k, v in config.retain_metadata.items():
        metadata[k] = _render(v, hook, status) if isinstance(v, str) else v
    tags = [_render(t, hook, status) for t in config.retain_tags]

    client.retain(
        bank_id,
        transcript,
        document_id=hook.task_id or "cline-task",
        context=config.retain_context,
        metadata=metadata,
        tags=tags,
        timeout=config.retain_timeout,
    )
    content.clear_transcript(hook.task_id)


# ── Entrypoints (called by the hook scripts) ─────────────────────────────────


def _run_recall(handler: Callable[[HookInput, HindsightClineConfig], str]) -> None:
    config = load_config()
    hook = cline_io.read_hook_input()
    try:
        cline_io.emit(context_modification=handler(hook, config))
    except Exception as e:  # never block Cline on a memory error
        debug_log(config, f"hook error: {e}")
        cline_io.emit()


def _run_retain(status: str) -> None:
    config = load_config()
    hook = cline_io.read_hook_input()
    try:
        handle_retain(hook, config, status)
    except Exception as e:
        debug_log(config, f"retain hook error: {e}")
    cline_io.emit()


def main_user_prompt_submit() -> None:
    _run_recall(handle_user_prompt_submit)


def main_task_start() -> None:
    _run_recall(handle_task_start)


def main_task_complete() -> None:
    _run_retain("completed")


def main_task_cancel() -> None:
    _run_retain("cancelled")
