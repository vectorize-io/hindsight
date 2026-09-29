"""Resolve the Hindsight bank for an Aider session (one bank per git repo)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from .config import AiderConfig

_FALLBACK_BANK = "aider"


def _sanitize(name: str) -> str:
    """Make a filesystem/URL-safe bank id from a repo name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()
    return slug or _FALLBACK_BANK


def _git_repo_name(cwd: Optional[Path] = None) -> Optional[str]:
    """The basename of the git working-tree root, or ``None`` if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    top = out.stdout.strip()
    return Path(top).name if top else None


def resolve_bank_id(config: AiderConfig, cwd: Optional[Path] = None) -> str:
    """Resolve the bank: explicit ``config.bank_id``, else the git repo name.

    Using the repo name (not a tool-specific prefix) lets Aider share a project's
    memory with the other Hindsight editor integrations on the same repo.
    """
    if config.bank_id:
        return _sanitize(config.bank_id)
    name = _git_repo_name(cwd)
    if name:
        return _sanitize(name)
    # Not a git repo: fall back to the directory name, then a constant.
    here = (cwd or Path.cwd()).name
    return _sanitize(here) if here else _FALLBACK_BANK
