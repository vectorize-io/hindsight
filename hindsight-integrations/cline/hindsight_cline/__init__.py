"""Hindsight long-term memory integration for Cline (lifecycle hooks)."""

from .install import get_hooks_dir, install_hooks, run_install, run_uninstall, write_user_config

__all__ = [
    "get_hooks_dir",
    "install_hooks",
    "run_install",
    "run_uninstall",
    "write_user_config",
]
