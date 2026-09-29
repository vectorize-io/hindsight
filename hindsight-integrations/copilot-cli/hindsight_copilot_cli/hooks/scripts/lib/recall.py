"""Shared recall-context building logic for `session_start.py` and
`subagent_start.py`.

Both hooks call Hindsight's recall API and format the results into an
`<hindsight_memories>` block. They differ only in what query they recall
with:

  - `session_start.py` uses the `initialPrompt` from the payload when
    present (falls back to the generic project query otherwise).
  - `subagent_start.py` never receives per-invocation task text (the
    `subagentStart` payload has no such field — only the static agent
    name/description), so it always uses the generic fallback query.
"""

import os

from .content import format_current_time, format_memories


def fallback_query(cwd, config):
    """Build a generic project-context recall query from a hook's `cwd`.

    Used whenever a hook has no specific prompt/task text to recall
    against (interactive `sessionStart` with no queued prompt, or any
    `subagentStart`).
    """
    project = os.path.basename(cwd.rstrip("/\\")) if cwd else "unknown"
    template = config.get(
        "recallFallbackQueryTemplate",
        "Project: {project}. Recall recent decisions, conventions, and context relevant to continuing work here.",
    )
    try:
        return template.format(project=project or "unknown")
    except (KeyError, IndexError):
        # Malformed user-supplied template — degrade to the raw template text
        # rather than crashing a hook that must never block the agent.
        return template


def build_recall_context(client, bank_id, query, config, debug_fn=None):
    """Call Hindsight's recall API and format results as a context block.

    Returns the formatted `<hindsight_memories>...</hindsight_memories>`
    string, or None if there are no results (or the call fails — the
    caller should treat that as "nothing to inject", not an error to
    surface to the user).
    """
    query = (query or "").strip()
    if not query:
        return None

    recall_timeout = config.get("recallTimeout", 10)
    if debug_fn:
        debug_fn(f"Recalling from bank '{bank_id}', query length: {len(query)}, timeout: {recall_timeout}")

    try:
        response = client.recall(
            bank_id=bank_id,
            query=query,
            max_tokens=config.get("recallMaxTokens", 1024),
            budget=config.get("recallBudget", "mid"),
            types=config.get("recallTypes"),
            timeout=recall_timeout,
        )
    except Exception as e:
        if debug_fn:
            debug_fn(f"Recall failed: {e}")
        return None

    results = response.get("results", [])
    if not results:
        if debug_fn:
            debug_fn("No memories found")
        return None

    if debug_fn:
        debug_fn(f"Injecting {len(results)} memories")

    memories_formatted = format_memories(results)
    preamble = config.get("recallPromptPreamble", "")
    current_time = format_current_time()

    return (
        f"<hindsight_memories>\n"
        f"{preamble}\n"
        f"Current time - {current_time}\n\n"
        f"{memories_formatted}\n"
        f"</hindsight_memories>"
    )
