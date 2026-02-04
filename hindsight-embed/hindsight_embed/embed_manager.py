"""
Abstract interface for managing Hindsight embedded servers and profiles.

This module provides a clean interface for daemon lifecycle and profile management,
abstracting away the implementation details.
"""

from abc import ABC, abstractmethod
from typing import Optional


class EmbedManager(ABC):
    """Abstract interface for managing Hindsight embedded servers and profiles."""

    @abstractmethod
    def ensure_running(self, config: dict, profile: str) -> bool:
        """
        Ensure daemon is running for the given profile with config.

        Args:
            config: Environment configuration dict (HINDSIGHT_API_* vars)
            profile: Profile name for isolation

        Returns:
            True if daemon is running (started or already running), False on failure
        """
        pass

    @abstractmethod
    def get_url(self, profile: str) -> str:
        """
        Get the URL for the daemon serving this profile.

        Args:
            profile: Profile name

        Returns:
            URL string (e.g., "http://127.0.0.1:54321")

        Raises:
            RuntimeError: If daemon is not running
        """
        pass

    @abstractmethod
    def stop(self, profile: str) -> bool:
        """
        Stop the daemon for this profile.

        Args:
            profile: Profile name

        Returns:
            True if stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_running(self, profile: str) -> bool:
        """
        Check if daemon is running for this profile.

        Args:
            profile: Profile name

        Returns:
            True if daemon is running and responsive
        """
        pass

    @abstractmethod
    def get_database_url(self, profile: str, db_url: Optional[str] = None) -> str:
        """
        Get the database URL for this profile.

        Args:
            profile: Profile name
            db_url: Optional override database URL

        Returns:
            Database connection string
        """
        pass
