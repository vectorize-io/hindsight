"""Hindsight memory wrapper for the Aider pair-programming CLI.

``hindsight-aider`` wraps ``aider``: it recalls relevant project memory before a
session (injected via a ``--read`` file) and retains the session transcript
afterwards, so memory persists across Aider sessions. Bank is per git repo.
"""

from .config import AiderConfig, load_config
from .runner import run

__version__ = "0.1.0"

__all__ = ["AiderConfig", "load_config", "run"]
