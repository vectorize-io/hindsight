"""Opt-in project identity, captured before background memory work starts.

Adapted from coding-agents/src/core/git-layout.ts: inspect Git's filesystem layout,
not a subprocess with a timeout that can silently fragment worktree memory banks.
Unlike that probe, permission errors and malformed/dangling pointers fail closed.
Inherited GIT_DIR / GIT_COMMON_DIR never select the repository for this path.
"""

from __future__ import annotations

import errno
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal


class GitProjectResolutionError(RuntimeError):
    """Cannot determine a safe project bank; do not substitute another identity."""


@dataclass(frozen=True)
class ProjectContext:
    cwd: str
    backend: str


def capture_project_context(cwd: str = "") -> ProjectContext:
    """Read stock Hermes' owning scope without constructing a terminal backend.

    These are internal Hermes APIs, deliberately imported only for {gitProject}.
    Missing APIs and refusal scopes propagate, rather than using another profile's
    process environment. Read backend even with an explicit cwd/override so a
    TerminalPolicyUnavailable refusal cannot be bypassed.
    """
    from agent.runtime_cwd import scoped_session_cwd
    from tools.terminal_scope import terminal_env

    backend = terminal_env("TERMINAL_ENV", "local").strip() or "local"
    logical_cwd = str(cwd or "").strip() or scoped_session_cwd() or terminal_env("TERMINAL_CWD", "").strip()
    return ProjectContext(logical_cwd, backend)


@dataclass(frozen=True)
class GitLayout:
    status: Literal["resolved", "absent", "failed"]
    common_dir: Path | None = None
    bare: bool = False
    reason: str = ""


_TRANSIENT = {errno.EAGAIN, errno.EMFILE, errno.ENFILE, errno.EBUSY, errno.EINTR, errno.EIO, errno.ETIMEDOUT}


def _entry_kind(path: Path) -> Literal["dir", "file", "absent"]:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "absent"
    # A dangling symlink is a broken layout, not an absent marker. Other errors
    # (including EACCES and ENOTDIR) must not let the walk choose a parent repo.
    if stat.S_ISLNK(mode):
        mode = path.stat().st_mode
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    raise ValueError(f"unsupported Git layout entry: {path}")


def _pointer(text: str, holder: Path) -> Path:
    text = text.strip()
    if not text or "\n" in text or "\r" in text or "\0" in text:
        raise ValueError("empty or malformed Git pointer")
    return (holder / text).resolve(strict=True)


def _resolved(git_dir: Path) -> GitLayout:
    git_dir = git_dir.resolve(strict=True)
    if _entry_kind(git_dir) != "dir":
        raise ValueError("Git directory is not a directory")
    common = git_dir
    common_pointer = git_dir / "commondir"
    kind = _entry_kind(common_pointer)
    if kind != "absent":
        if kind != "file":
            raise ValueError("commondir is not a file")
        common = _pointer(common_pointer.read_text(encoding="utf-8"), git_dir)
    if (
        _entry_kind(git_dir / "HEAD") != "file"
        or _entry_kind(common / "objects") != "dir"
        or _entry_kind(common / "refs") != "dir"
    ):
        raise ValueError("incomplete Git directory layout")
    # Read only [core]'s bare flag; a similarly named setting in another section
    # must not turn a hidden separate git-dir into a bare-hub identity.
    config_path = common / "config"
    if _entry_kind(config_path) != "file":
        raise ValueError("missing or unsupported Git config")
    config = config_path.read_text(encoding="utf-8")
    core = False
    bare = False
    for line in config.splitlines():
        if line.lstrip().startswith("["):
            core = bool(re.fullmatch(r"\s*\[core\]\s*(?:[#;].*)?", line, re.IGNORECASE))
        elif core and (match := re.fullmatch(r"\s*bare(?:\s*=\s*(.*?))?\s*(?:[#;].*)?", line, re.IGNORECASE)):
            value = match[1]
            # Git booleans are case-insensitive; an implicit key means true,
            # an empty value means false, and the last value wins.
            value = "true" if value is None else value.strip().strip('"').lower()
            if value not in {"true", "yes", "on", "1", "false", "no", "off", "0", ""}:
                raise ValueError("invalid core.bare value")
            bare = value in {"true", "yes", "on", "1"}
    return GitLayout("resolved", common, bare)


def _probe_once(directory: str) -> GitLayout:
    current = Path(directory).expanduser().resolve(strict=True)
    if _entry_kind(current) != "dir":
        raise ValueError("project cwd is not a directory")
    while True:
        marker = current / ".git"
        kind = _entry_kind(marker)
        if kind == "dir":
            return _resolved(marker)
        if kind == "file":
            text = marker.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                raise ValueError("malformed .git pointer")
            return _resolved(_pointer(text[len("gitdir:") :], current))
        if (
            _entry_kind(current / "HEAD") == "file"
            and _entry_kind(current / "objects") == "dir"
            and _entry_kind(current / "refs") == "dir"
        ):
            return _resolved(current)
        if current.parent == current:
            return GitLayout("absent")
        current = current.parent


def probe_git_layout(directory: str) -> GitLayout:
    """Return resolved/absent/failed; only a completed ancestor walk proves absence."""
    for attempt in range(3):
        try:
            if not directory:
                raise ValueError("missing local project cwd")
            return _probe_once(directory)
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, OSError) and exc.errno in _TRANSIENT and attempt < 2:
                time.sleep(0.02 * (attempt + 1))
                continue
            return GitLayout("failed", reason=str(exc))
    raise AssertionError("unreachable")


def resolve_git_project(cwd: str, override: str = "", *, backend: str = "local") -> str:
    """Explicit name, local common Git identity, or genuine nonrepo basename.

    Remote backends use only the logical cwd basename, NOT remote Git discovery.
    No host filesystem reads, subprocesses or terminal launches occur on that path.
    """
    if override and str(override).strip():
        return str(override).strip()
    if backend != "local":
        path_type = PureWindowsPath if "\\" in cwd or PureWindowsPath(cwd).drive else PurePosixPath
        return path_type(cwd).name if cwd else ""
    # Delay even getcwd until after the explicit override: it can fail when a
    # launch directory has been removed, and an override needs no host path.
    cwd = cwd or os.getcwd()
    layout = probe_git_layout(cwd)
    if layout.status == "failed":
        raise GitProjectResolutionError(f"Cannot resolve {{gitProject}} for {cwd!r}: {layout.reason}")
    if layout.status == "absent":
        return Path(os.path.abspath(Path(cwd).expanduser())).name
    common = layout.common_dir
    assert common is not None
    # Match coding-agents: ordinary .git and hidden bare hubs name the parent;
    # standalone bare repositories retain their directory name (including .git).
    return common.parent.name if common.name == ".git" or (layout.bare and common.name.startswith(".")) else common.name
