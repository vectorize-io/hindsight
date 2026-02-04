"""Hindsight embedded CLI - local memory operations without a server."""

from .daemon_embed_manager import DaemonEmbedManager
from .embed_manager import EmbedManager

__version__ = "0.4.9"

__all__ = [
    "EmbedManager",
    "DaemonEmbedManager",
]


def get_embed_manager() -> EmbedManager:
    """Get the default embed manager instance."""
    return DaemonEmbedManager()
