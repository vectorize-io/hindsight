"""
API namespace classes for organizing client methods.

These classes provide organized access to different parts of the Hindsight API
while ensuring the daemon is running before each call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .embedded import HindsightEmbedded


class BanksAPI:
    """Namespace for bank-related operations."""

    def __init__(self, embedded: "HindsightEmbedded"):
        self._embedded = embedded

    def create(
        self,
        bank_id: str,
        name: str | None = None,
        mission: str | None = None,
        disposition: dict[str, Any] | None = None,
    ):
        """Create a new bank."""
        self._embedded._ensure_started()
        return self._embedded._client.create_bank(
            bank_id=bank_id,
            name=name,
            mission=mission,
            disposition=disposition,
        )

    def delete(self, bank_id: str):
        """Delete a bank."""
        self._embedded._ensure_started()
        return self._embedded._client.delete_bank(bank_id=bank_id)

    def set_mission(self, bank_id: str, mission: str):
        """Set or update the mission for a bank."""
        self._embedded._ensure_started()
        return self._embedded._client.set_mission(bank_id=bank_id, mission=mission)

    def set_disposition(self, bank_id: str, disposition: dict[str, Any]):
        """Set or update the disposition for a bank."""
        self._embedded._ensure_started()
        return self._embedded._client.set_disposition(bank_id=bank_id, disposition=disposition)


class MentalModelsAPI:
    """Namespace for mental model operations."""

    def __init__(self, embedded: "HindsightEmbedded"):
        self._embedded = embedded

    def create(
        self,
        bank_id: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
    ):
        """Create a new mental model."""
        self._embedded._ensure_started()
        return self._embedded._client.create_mental_model(
            bank_id=bank_id,
            name=name,
            content=content,
            tags=tags,
        )

    def list(self, bank_id: str, tags: list[str] | None = None):
        """List all mental models for a bank."""
        self._embedded._ensure_started()
        return self._embedded._client.list_mental_models(bank_id=bank_id, tags=tags)

    def get(self, bank_id: str, mental_model_id: str):
        """Get a specific mental model."""
        self._embedded._ensure_started()
        return self._embedded._client.get_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)

    def refresh(self, bank_id: str, mental_model_id: str):
        """Refresh a mental model."""
        self._embedded._ensure_started()
        return self._embedded._client.refresh_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)

    def update(
        self,
        bank_id: str,
        mental_model_id: str,
        name: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ):
        """Update a mental model."""
        self._embedded._ensure_started()
        return self._embedded._client.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model_id,
            name=name,
            content=content,
            tags=tags,
        )

    def delete(self, bank_id: str, mental_model_id: str):
        """Delete a mental model."""
        self._embedded._ensure_started()
        return self._embedded._client.delete_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)


class DirectivesAPI:
    """Namespace for directive operations."""

    def __init__(self, embedded: "HindsightEmbedded"):
        self._embedded = embedded

    def create(
        self,
        bank_id: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
    ):
        """Create a new directive."""
        self._embedded._ensure_started()
        return self._embedded._client.create_directive(
            bank_id=bank_id,
            name=name,
            content=content,
            tags=tags,
        )

    def list(self, bank_id: str, tags: list[str] | None = None):
        """List all directives for a bank."""
        self._embedded._ensure_started()
        return self._embedded._client.list_directives(bank_id=bank_id, tags=tags)

    def get(self, bank_id: str, directive_id: str):
        """Get a specific directive."""
        self._embedded._ensure_started()
        return self._embedded._client.get_directive(bank_id=bank_id, directive_id=directive_id)

    def update(
        self,
        bank_id: str,
        directive_id: str,
        name: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ):
        """Update a directive."""
        self._embedded._ensure_started()
        return self._embedded._client.update_directive(
            bank_id=bank_id,
            directive_id=directive_id,
            name=name,
            content=content,
            tags=tags,
        )

    def delete(self, bank_id: str, directive_id: str):
        """Delete a directive."""
        self._embedded._ensure_started()
        return self._embedded._client.delete_directive(bank_id=bank_id, directive_id=directive_id)


class MemoriesAPI:
    """Namespace for memory operations."""

    def __init__(self, embedded: "HindsightEmbedded"):
        self._embedded = embedded

    def list(
        self,
        bank_id: str,
        type: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List memories in a bank."""
        self._embedded._ensure_started()
        return self._embedded._client.list_memories(
            bank_id=bank_id,
            type=type,
            search_query=search_query,
            limit=limit,
            offset=offset,
        )
