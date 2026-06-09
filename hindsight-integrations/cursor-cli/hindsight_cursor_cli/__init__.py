"""Hindsight long-term memory integration for Cursor CLI."""

from .install import (
    get_install_dir,
    merge_hooks,
    render_hooks_block,
    run_install,
    run_uninstall,
    seed_user_config,
    write_settings,
)

__all__ = [
    "get_install_dir",
    "merge_hooks",
    "render_hooks_block",
    "run_install",
    "run_uninstall",
    "seed_user_config",
    "write_settings",
]
