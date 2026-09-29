"""Hindsight long-term memory integration for ZCode."""

from .install import (
    get_config_path,
    get_install_dir,
    merge_hooks,
    render_hooks_events,
    run_install,
    run_uninstall,
    seed_user_config,
    write_settings,
)

__all__ = [
    "get_config_path",
    "get_install_dir",
    "merge_hooks",
    "render_hooks_events",
    "run_install",
    "run_uninstall",
    "seed_user_config",
    "write_settings",
]
