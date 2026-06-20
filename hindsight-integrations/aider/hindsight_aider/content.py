"""Compose the recall query, the injected memory file, and the retain transcript."""

from __future__ import annotations

from typing import Any, Optional


def compose_recall_query(aider_args: list[str], default_query: str, max_chars: int = 800) -> str:
    """Pick the recall query: Aider's ``-m``/``--message`` if given, else the default.

    When the user launches a one-shot task (``aider -m "fix the auth bug"``) we
    recall against that; for an interactive session we recall a general
    project-context query.
    """
    for i, arg in enumerate(aider_args):
        if arg in ("-m", "--message") and i + 1 < len(aider_args):
            return aider_args[i + 1][:max_chars]
        if arg.startswith("--message="):
            return arg[len("--message=") :][:max_chars]
    return default_query


def format_memory(results: list[Any], preamble: str) -> str:
    """Render recalled memories as a Markdown file for Aider to ``--read``.

    Returns an empty string when there are no results so the caller can skip
    writing/injecting an empty memory file.
    """
    lines = [getattr(r, "text", "") for r in results]
    lines = [text for text in lines if text]
    if not lines:
        return ""
    body = "\n".join(f"- {text}" for text in lines)
    return f"# Project memory (Hindsight)\n\n{preamble.strip()}\n\n{body}\n"


def format_transcript(session_text: str, max_chars: int = 200_000) -> str:
    """Trim a captured Aider chat-history slice into a transcript to retain."""
    text = session_text.strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def find_workdir(aider_args: list[str]) -> Optional[str]:
    """Return an explicit working dir if Aider was pointed at one (``--cwd``)."""
    for i, arg in enumerate(aider_args):
        if arg == "--cwd" and i + 1 < len(aider_args):
            return aider_args[i + 1]
        if arg.startswith("--cwd="):
            return arg[len("--cwd=") :]
    return None
