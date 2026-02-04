"""
Wrapper for Hindsight client that adds API namespaces.

Provides organized access to different parts of the Hindsight API through
namespaces like .banks, .mental_models, etc.
"""

from __future__ import annotations

from typing import Any

from hindsight_client import Hindsight


class BanksAPI:
    """Namespace for bank-related operations."""

    def __init__(self, client: Hindsight):
        self._client = client

    def create(
        self,
        bank_id: str,
        name: str | None = None,
        mission: str | None = None,
        disposition: dict[str, Any] | None = None,
    ):
        """Create a new bank."""
        return self._client.create_bank(
            bank_id=bank_id,
            name=name,
            mission=mission,
            disposition=disposition,
        )

    def delete(self, bank_id: str):
        """Delete a bank."""
        return self._client.delete_bank(bank_id=bank_id)

    def set_mission(self, bank_id: str, mission: str):
        """Set or update the mission for a bank."""
        return self._client.set_mission(bank_id=bank_id, mission=mission)

    def set_disposition(self, bank_id: str, disposition: dict[str, Any]):
        """Set or update the disposition for a bank."""
        return self._client.set_disposition(bank_id=bank_id, disposition=disposition)

    def list(self):
        """List all banks."""
        from hindsight_client.hindsight_client import _run_async

        return _run_async(self._client._banks_api.list_banks())


class MentalModelsAPI:
    """Namespace for mental model operations."""

    def __init__(self, client: Hindsight):
        self._client = client

    def create(
        self,
        bank_id: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
    ):
        """Create a new mental model."""
        return self._client.create_mental_model(
            bank_id=bank_id,
            name=name,
            content=content,
            tags=tags,
        )

    def list(self, bank_id: str, tags: list[str] | None = None):
        """List all mental models for a bank."""
        return self._client.list_mental_models(bank_id=bank_id, tags=tags)

    def get(self, bank_id: str, mental_model_id: str):
        """Get a specific mental model."""
        return self._client.get_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)

    def refresh(self, bank_id: str, mental_model_id: str):
        """Refresh a mental model."""
        return self._client.refresh_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)

    def update(
        self,
        bank_id: str,
        mental_model_id: str,
        name: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ):
        """Update a mental model."""
        return self._client.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model_id,
            name=name,
            content=content,
            tags=tags,
        )

    def delete(self, bank_id: str, mental_model_id: str):
        """Delete a mental model."""
        return self._client.delete_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)


class DirectivesAPI:
    """Namespace for directive operations."""

    def __init__(self, client: Hindsight):
        self._client = client

    def create(
        self,
        bank_id: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
    ):
        """Create a new directive."""
        return self._client.create_directive(
            bank_id=bank_id,
            name=name,
            content=content,
            tags=tags,
        )

    def list(self, bank_id: str, tags: list[str] | None = None):
        """List all directives for a bank."""
        return self._client.list_directives(bank_id=bank_id, tags=tags)

    def get(self, bank_id: str, directive_id: str):
        """Get a specific directive."""
        return self._client.get_directive(bank_id=bank_id, directive_id=directive_id)

    def update(
        self,
        bank_id: str,
        directive_id: str,
        name: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ):
        """Update a directive."""
        return self._client.update_directive(
            bank_id=bank_id,
            directive_id=directive_id,
            name=name,
            content=content,
            tags=tags,
        )

    def delete(self, bank_id: str, directive_id: str):
        """Delete a directive."""
        return self._client.delete_directive(bank_id=bank_id, directive_id=directive_id)


class MemoriesAPI:
    """Namespace for memory operations."""

    def __init__(self, client: Hindsight):
        self._client = client

    def list(
        self,
        bank_id: str,
        type: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """List memories in a bank."""
        return self._client.list_memories(
            bank_id=bank_id,
            type=type,
            search_query=search_query,
            limit=limit,
            offset=offset,
        )


class HindsightClient(Hindsight):
    """
    Enhanced Hindsight client with organized API namespaces.

    This wrapper extends the auto-generated Hindsight client with organized
    access to different parts of the API through namespaces.

    Example:
        ```python
        from hindsight import HindsightClient

        client = HindsightClient(base_url="http://localhost:8888")

        # Core operations (inherited from Hindsight)
        client.retain(bank_id="test", content="Hello")
        results = client.recall(bank_id="test", query="Hello")

        # Organized API access through namespaces
        client.banks.create(bank_id="test", name="Test Bank")
        models = client.mental_models.list(bank_id="test")
        directives = client.directives.list(bank_id="test")
        memories = client.memories.list(bank_id="test")
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._banks_namespace: BanksAPI | None = None
        self._mental_models_namespace: MentalModelsAPI | None = None
        self._directives_namespace: DirectivesAPI | None = None
        self._memories_namespace: MemoriesAPI | None = None

    @property
    def banks(self) -> BanksAPI:
        """Access bank management operations."""
        if self._banks_namespace is None:
            self._banks_namespace = BanksAPI(self)
        return self._banks_namespace

    @property
    def mental_models(self) -> MentalModelsAPI:
        """Access mental model operations."""
        if self._mental_models_namespace is None:
            self._mental_models_namespace = MentalModelsAPI(self)
        return self._mental_models_namespace

    @property
    def directives(self) -> DirectivesAPI:
        """Access directive operations."""
        if self._directives_namespace is None:
            self._directives_namespace = DirectivesAPI(self)
        return self._directives_namespace

    @property
    def memories(self) -> MemoriesAPI:
        """Access memory listing operations."""
        if self._memories_namespace is None:
            self._memories_namespace = MemoriesAPI(self)
        return self._memories_namespace
