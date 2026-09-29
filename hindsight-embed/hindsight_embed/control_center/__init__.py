"""Hindsight Embed control center.

A persistent, localhost-only web app bundled in hindsight-embed that writes
profile configuration (the LLM setup wizard) and supervises the daemon
(start/stop/restart/status). It is a thin web front-end over ``ProfileManager``
and ``daemon_client`` — it owns no lifecycle logic of its own, so it can be
restarted without affecting a running daemon.

The desktop app (and headless users) simply open the URL printed by
``hindsight-embed control start``; no logic is reimplemented downstream.
"""

from .lifecycle import (
    CONTROL_PORT_DEFAULT,
    control_status,
    resolve_control_port,
    start_control_center,
    stop_control_center,
)

__all__ = [
    "CONTROL_PORT_DEFAULT",
    "control_status",
    "resolve_control_port",
    "start_control_center",
    "stop_control_center",
]
