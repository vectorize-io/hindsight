"""Bracket an Aider session with recall (before) and retain (after).

The flow:

1. Recall relevant project memory and write it to a Markdown file.
2. Launch ``aider --read <memory-file> <user args>`` so the memory is in context.
3. After Aider exits, read the slice of the chat-history file written during the
   session and retain it to the project's Hindsight bank.

The orchestration takes an injectable ``run_aider`` callable so it is fully
testable without a real ``aider`` binary or a live Hindsight server.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from ._client import resolve_client
from .bank import resolve_bank_id
from .config import AiderConfig, load_config
from .content import compose_recall_query, format_memory, format_transcript

logger = logging.getLogger("hindsight_aider")

RunAider = Callable[[list[str]], int]


def build_aider_command(config: AiderConfig, aider_args: list[str], memory_path: Optional[Path]) -> list[str]:
    """The aider argv, with ``--read <memory-file>`` prepended when memory exists."""
    cmd = [config.aider_command]
    if memory_path is not None:
        cmd += ["--read", str(memory_path)]
    cmd += list(aider_args)
    return cmd


def do_recall(client: Any, config: AiderConfig, bank_id: str, query: str, memory_path: Path) -> bool:
    """Recall memory and write it to ``memory_path``. Returns whether a file was written.

    Recall failures never block the Aider session — they're logged and skipped.
    """
    try:
        resp = client.recall(
            bank_id=bank_id,
            query=query,
            budget=config.recall_budget,
            max_tokens=config.recall_max_tokens,
            types=config.recall_types,
        )
        results = getattr(resp, "results", None) or []
        text = format_memory(results, config.recall_preamble)
    except Exception as e:
        logger.warning("Hindsight recall failed (continuing without memory): %s", e)
        return False
    if not text:
        return False
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(text, encoding="utf-8")
    return True


def history_size(path: Path) -> int:
    """Byte length of the chat-history file (0 if it doesn't exist yet)."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_history_delta(path: Path, prev_size: int) -> str:
    """Return the chat-history text written after ``prev_size`` bytes.

    If the file shrank (rotated/cleared) since the snapshot, return the whole file.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    chunk = data[prev_size:] if len(data) >= prev_size else data
    return chunk.decode("utf-8", "replace")


def do_retain(client: Any, config: AiderConfig, bank_id: str, transcript: str) -> None:
    """Retain the session transcript. Failures are logged, not raised."""
    try:
        client.retain(
            bank_id=bank_id,
            content=transcript,
            context="aider",
            metadata={"source": "aider"},
        )
    except Exception as e:
        logger.warning("Hindsight retain failed: %s", e)


def _default_run_aider(cmd: list[str]) -> int:
    """Run aider, inheriting stdio so the session is interactive."""
    try:
        return subprocess.run(cmd).returncode
    except FileNotFoundError:
        print(
            f"hindsight-aider: '{cmd[0]}' not found on PATH. Install Aider "
            "(https://aider.chat) or set HINDSIGHT_AIDER_COMMAND.",
            file=sys.stderr,
        )
        return 127


def run(
    aider_args: list[str],
    *,
    config: Optional[AiderConfig] = None,
    client: Optional[Any] = None,
    run_aider: Optional[RunAider] = None,
) -> int:
    """Recall -> run aider -> retain. Returns Aider's exit code."""
    config = config or load_config()
    client = client or resolve_client(config)
    bank_id = resolve_bank_id(config)

    memory_path = Path(config.memory_filename)
    history_path = Path(config.chat_history_file)

    injected = False
    if config.auto_recall:
        query = compose_recall_query(aider_args, config.recall_default_query)
        injected = do_recall(client, config, bank_id, query, memory_path)

    prev_size = history_size(history_path)

    cmd = build_aider_command(config, aider_args, memory_path if injected else None)
    code = (run_aider or _default_run_aider)(cmd)

    if config.auto_retain:
        transcript = format_transcript(read_history_delta(history_path, prev_size))
        if transcript:
            do_retain(client, config, bank_id, transcript)

    return code
