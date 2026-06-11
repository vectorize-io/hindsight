"""Per-project memory-bank derivation.

Each project gets its own Hindsight bank, so memory from one codebase doesn't
bleed into another. The bank id is derived from the project's git repository
root (so all linked worktrees of a repo share one bank), falling back to the
directory basename when git is unavailable.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

from .config import ZedConfig


def _git_repo_root(directory: str) -> Optional[str]:
    """Return the main worktree root for *directory*, or None if not a repo.

    ``git rev-parse --git-common-dir`` resolves to the *main* worktree's ``.git``
    even from a linked worktree, so all worktrees of a repo map to one bank.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    common = out.stdout.strip()
    if not common:
        return None
    common_path = (Path(directory) / common).resolve() if not Path(common).is_absolute() else Path(common)
    # ``<root>/.git`` → parent is the worktree root; a bare ``repo.git`` → itself.
    if common_path.name == ".git":
        return str(common_path.parent)
    return str(common_path)


def _slugify(name: str) -> str:
    """Reduce a name to a stable, bank-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()
    return slug or "project"


def project_name(directory: str) -> str:
    """Derive a stable project name from a directory (git root basename)."""
    root = _git_repo_root(directory) or directory
    return _slugify(Path(root).name)


def bank_id_for_project(directory: str, config: ZedConfig) -> str:
    """Resolve the bank id for a project directory.

    Honors a configured ``fixed_bank_id`` (single shared bank); otherwise
    ``<prefix>-<project>`` (per-project).
    """
    if config.fixed_bank_id:
        return config.fixed_bank_id
    name = project_name(directory)
    prefix = config.bank_prefix.strip("-")
    return f"{prefix}-{name}" if prefix else name


def bank_id_for_thread_paths(folder_paths: list, config: ZedConfig) -> Optional[str]:
    """Resolve the bank id for a thread, from its ``folder_paths``.

    Returns None when the thread has no associated project folder (so the caller
    can fall back to a default). Uses the first existing folder path.
    """
    if config.fixed_bank_id:
        return config.fixed_bank_id
    for path in folder_paths or []:
        if path and Path(path).exists():
            return bank_id_for_project(path, config)
    # No usable folder — first path by name, or nothing.
    if folder_paths:
        return bank_id_for_project(folder_paths[0], config)
    return None
