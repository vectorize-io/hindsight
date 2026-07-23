"""Derive a stable per-project bank id for the current repository.

Devin Desktop's MCP config is a single global file — it can't carry a different
bank per workspace, and it passes no reliable project identifier to the MCP
server. So the project bank must be derived **client-side at ``init`` time**
(inside the repo, where git context exists) and baked into that repo's committed
``.devin/rules/hindsight.md``.

Derivation prefers the **git remote** (stable across machines, clones, and
checkout paths, and identical for teammates → shared project memory), then the
git repo folder, then the current folder. The result is ``<global>-<slug>`` so
project banks are grouped under the global bank and never collide with it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def _slugify(text: str) -> str:
    """Lowercase, collapse non-alphanumerics to single dashes, trim dashes."""
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    """Run ``git <args>`` in ``cwd``; return trimmed stdout or ``None``."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def slug_from_remote(url: str) -> Optional[str]:
    """Slug from a git remote URL's ``owner/repo`` (host and ``.git`` dropped).

    Handles ``git@host:owner/repo.git``, ``https://host/owner/repo.git``, and
    ``ssh://git@host/owner/repo``. Dropping the host keeps the slug identical for
    teammates who clone the same repo over different protocols.
    """
    url = url.strip()
    if url.endswith(".git"):
        url = url[:-4]
    scp = re.match(r"^[^/@]+@[^:/]+:(.+)$", url)  # scp-like: git@host:owner/repo
    if scp:
        path = scp.group(1)
    else:
        scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://[^/]+/(.+)$", url)
        path = scheme.group(1) if scheme else url
    return _slugify(path) or None


def derive_project_slug(cwd: Path) -> Tuple[Optional[str], str]:
    """Return ``(slug, source)`` for ``cwd``; ``slug`` is ``None`` if none found.

    ``source`` is a short human-readable description of where the slug came from
    (for surfacing in ``init`` output), even when ``slug`` is ``None``.
    """
    remote = _run_git(["remote", "get-url", "origin"], cwd)
    if remote:
        slug = slug_from_remote(remote)
        if slug:
            return slug, f"git remote origin ({remote})"

    toplevel = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if toplevel:
        slug = _slugify(Path(toplevel).name)
        if slug:
            return slug, f"git repo folder ({Path(toplevel).name})"

    slug = _slugify(Path(cwd).name)
    if slug:
        return slug, f"current folder ({Path(cwd).name})"

    return None, "no git remote, git repo, or usable folder name"


def project_bank_id(global_bank: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Return ``(<global>-<slug>, source)``; bank is ``None`` if underivable."""
    slug, source = derive_project_slug(cwd)
    if slug is None:
        return None, source
    return f"{global_bank}-{slug}", source
