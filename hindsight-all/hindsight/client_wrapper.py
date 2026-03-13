"""
Wrapper for Hindsight client that adds API namespaces.

Provides organized access to different parts of the Hindsight API through
namespaces like .banks, .mental_models, etc.
"""

from __future__ import annotations

from typing import Any

from hindsight_client import Hindsight


class BanksAPI:
    """Namespace for bank-related operations.
    
    Provides methods to create, delete, and manage memory banks.
    """

    def __init__(self, client: Hindsight):
        self._client = client

    def create(
        self,
        bank_id: str,
        name: str | None = None,
        mission: str | None = None,
        disposition: dict[str, Any] | None = None,
    ) -> Any:
        """Create a new bank.
        
        Args:
            bank_id: Unique identifier for the bank.
            name: Optional display name for the bank.
            mission: Optional mission statement for the bank.
            disposition: Optional disposition configuration dict.
            
        Returns:
            Bank creation response from the API.
        """
        return self._client.create_bank(
            bank_id=bank_id,
            name=name,
            mission=mission,
            disposition=disposition,
        )

    def delete(self, bank_id: str) -> Any:
        """Delete a bank.
        
        Args:
            bank_id: The ID of the bank to delete.
            
        Returns:
            Deletion response from the API.
        """
        return self._client.delete_bank(bank_id=bank_id)

    def set_mission(self, bank_id: str, mission: str) -> Any:
        """Set or update the mission for a bank.
        
        Args:
            bank_id: The ID of the bank.
            mission: The mission statement to set.
            
        Returns:
            API response confirming the update.
        """
        return self._client.set_mission(bank_id=bank_id, mission=mission)

    def set_disposition(self, bank_id: str, disposition: dict[str, Any]) -> Any:
        """Set or update the disposition for a bank.
        
        Args:
            bank_id: The ID of the bank.
            disposition: The disposition configuration dict.
            
        Returns:
            API response confirming the update.
        """
        return self._client.set_disposition(bank_id=bank_id, disposition=disposition)

    def list(self) -> Any:
        """List all banks.
        
        Returns:
            List of banks from the API.
        """
        from hindsight_client.hindsight_client import _run_async

        return _run_async(self._client._banks_api.list_banks())


class MentalModelsAPI:
    """Namespace for mental model operations.
    
    Mental models are reusable knowledge structures that guide agent behavior.
    """

    def __init__(self, client: Hindsight):
        self._client = client

    def create(
        self,
        bank_id: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
    ) -> Any:
        """Create a new mental model.
        
        Args:
            bank_id: The ID of the bank to add the model to.
            name: Name for the mental model.
            content: The content/instructions for the mental model.
            tags: Optional list of tags for categorization.
            
        Returns:
            Creation response from the API.
        """
        return self._client.create_mental_model(
            bank_id=bank_id,
            name=name,
            content=content,
            tags=tags,
        )

    def list(self, bank_id: str, tags: list[str] | None = None) -> Any:
        """List all mental models for a bank.
        
        Args:
            bank_id: The ID of the bank.
            tags: Optional filter by tags.
            
        Returns:
            List of mental models.
        """
        return self._client.list_mental_models(bank_id=bank_id, tags=tags)

    def get(self, bank_id: str, mental_model_id: str) -> Any:
        """Get a specific mental model.
        
        Args:
            bank_id: The ID of the bank.
            mental_model_id: The ID of the mental model.
            
        Returns:
            The mental model details.
        """
        return self._client.get_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)

    def refresh(self, bank_id: str, mental_model_id: str) -> Any:
        """Refresh a mental model.
        
        Args:
            bank_id: The ID of the bank.
            mental_model_id: The ID of the mental model to refresh.
            
        Returns:
            Refresh response from the API.
        """
        return self._client.refresh_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)

    def update(
        self,
        bank_id: str,
        mental_model_id: str,
        name: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """Update a mental model.
        
        Args:
            bank_id: The ID of the bank.
            mental_model_id: The ID of the mental model to update.
            name: Optional new name.
            content: Optional new content.
            tags: Optional new tags list.
            
        Returns:
            Update response from the API.
        """
        return self._client.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model_id,
            name=name,
            content=content,
            tags=tags,
        )

    def delete(self, bank_id: str, mental_model_id: str) -> Any:
        """Delete a mental model.
        
        Args:
            bank_id: The ID of the bank.
            mental_model_id: The ID of the mental model to delete.
            
        Returns:
            Deletion response from the API.
        """
        return self._client.delete_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)


class DirectivesAPI:
    """Namespace for directive operations.
    
    Directives are explicit instructions that guide agent behavior.
    """

    def __init__(self, client: Hindsight):
        self._client = client

    def create(
        self,
        bank_id: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
    ) -> Any:
        """Create a new directive.
        
        Args:
            bank_id: The ID of the bank to add the directive to.
            name: Name for the directive.
            content: The directive content/instructions.
            tags: Optional list of tags for categorization.
            
        Returns:
            Creation response from the API.
        """
        return self._client.create_directive(
            bank_id=bank_id,
            name=name,
            content=content,
            tags=tags,
        )

    def list(self, bank_id: str, tags: list[str] | None = None) -> Any:
        """List all directives for a bank.
        
        Args:
            bank_id: The ID of the bank.
            tags: Optional filter by tags.
            
        Returns:
            List of directives.
        """
        return self._client.list_directives(bank_id=bank_id, tags=tags)

    def get(self, bank_id: str, directive_id: str) -> Any:
        """Get a specific directive.
        
        Args:
            bank_id: The ID of the bank.
            directive_id: The ID of the directive.
            
        Returns:
            The directive details.
        """
        return self._client.get_directive(bank_id=bank_id, directive_id=directive_id)

    def update(
        self,
        bank_id: str,
        directive_id: str,
        name: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """Update a directive.
        
        Args:
            bank_id: The ID of the bank.
            directive_id: The ID of the directive to update.
            name: Optional new name.
            content: Optional new content.
            tags: Optional new tags list.
            
        Returns:
            Update response from the API.
        """
        return self._client.update_directive(
            bank_id=bank_id,
            directive_id=directive_id,
            name=name,
            content=content,
            tags=tags,
        )

    def delete(self, bank_id: str, directive_id: str) -> Any:
        """Delete a directive.
        
        Args:
            bank_id: The ID of the bank.
            directive_id: The ID of the directive to delete.
            
        Returns:
            Deletion response from the API.
        """
        return self._client.delete_directive(bank_id=bank_id, directive_id=directive_id)


class MemoriesAPI:
    """Namespace for memory operations.
    
    Provides methods to query and retrieve stored memories.
    """

    def __init__(self, client: Hindsight):
        self._client = client

    def list(
        self,
        bank_id: str,
        type: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        """List memories in a bank.
        
        Args:
            bank_id: The ID of the bank to query.
            type: Optional filter by memory type.
            search_query: Optional search query for filtering.
            limit: Maximum number of results to return (default: 100).
            offset: Number of results to skip for pagination (default: 0).
            
        Returns:
            List of memories matching the criteria.
        """
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
        
    Attributes:
        banks: Namespace for bank management operations.
        mental_models: Namespace for mental model operations.
        directives: Namespace for directive operations.
        memories: Namespace for memory listing operations.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._banks_namespace: BanksAPI | None = None
        self._mental_models_namespace: MentalModelsAPI | None = None
        self._directives_namespace: DirectivesAPI | None = None
        self._memories_namespace: MemoriesAPI | None = None

    @property
    def banks(self) -> BanksAPI:
        """Access bank management operations.
        
        Returns:
            BanksAPI instance for bank operations.
        """
        if self._banks_namespace is None:
            self._banks_namespace = BanksAPI(self)
        return self._banks_namespace

    @property
    def mental_models(self) -> MentalModelsAPI:
        """Access mental model operations.
        
        Returns:
            MentalModelsAPI instance for mental model operations.
        """
        if self._mental_models_namespace is None:
            self._mental_models_namespace = MentalModelsAPI(self)
        return self._mental_models_namespace

    @property
    def directives(self) -> DirectivesAPI:
        """Access directive operations.
        
        Returns:
            DirectivesAPI instance for directive operations.
        """
        if self._directives_namespace is None:
            self._directives_namespace = DirectivesAPI(self)
        return self._directives_namespace

    @property
    def memories(self) -> MemoriesAPI:
        """Access memory listing operations.
        
        Returns:
            MemoriesAPI instance for memory operations.
        """
        if self._memories_namespace is None:
            self._memories_namespace = MemoriesAPI(self)
        return self._memories_namespace
