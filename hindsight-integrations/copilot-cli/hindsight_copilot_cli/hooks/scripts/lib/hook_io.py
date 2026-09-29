"""Hook payload field access helpers for GitHub Copilot CLI hooks.

Copilot CLI supports two payload formats per hook, selected by how the event
is named in ``hooks.json``:

  - camelCase (event configured as e.g. ``sessionStart``) — fields like
    ``sessionId``, ``transcriptPath``, ``initialPrompt``.
  - "VS Code compatible" / Claude-style (event configured as e.g.
    ``SessionStart``) — snake_case fields like ``session_id``,
    ``transcript_path``, ``initial_prompt``, plus a ``hook_event_name`` key.

This integration's ``hooks.json`` registers the camelCase event names, so the
native payload is camelCase. ``field()`` defensively also accepts the
snake_case equivalent so hand-edited or VS Code-compatible hook
configurations still work.
"""


def field(hook_input, camel_name, snake_name=None, default=None):
    """Read a hook payload field, accepting camelCase or snake_case keys."""
    if not isinstance(hook_input, dict):
        return default
    if camel_name in hook_input:
        return hook_input[camel_name]
    snake_name = snake_name or _to_snake(camel_name)
    if snake_name in hook_input:
        return hook_input[snake_name]
    return default


def _to_snake(name):
    out = []
    for ch in name:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)
