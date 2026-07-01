"""Execution adapter protocol and registry."""

from abc import ABC, abstractmethod
from typing import Any, Protocol


class BaseAdapter(ABC):
    """Protocol for execution adapters."""

    async def execute(self, execution: dict) -> dict:
        """Execute action.
        
        Args:
            execution: Execution record dict (action_type, target, params, etc)
        
        Returns:
            Result dict with keys: exit_code, output, result
        """
        pass


class AdapterRegistry:
    """Registry for adapter implementations."""

    def __init__(self):
        self._adapters: dict[str, type[BaseAdapter]] = {}

    def register(self, name: str, adapter_class: type[BaseAdapter]) -> None:
        """Register adapter class by action type name."""
        self._adapters[name] = adapter_class

    def get(self, name: str) -> type[BaseAdapter] | None:
        """Get adapter class by action type name."""
        return self._adapters.get(name)

    def list(self) -> dict[str, type[BaseAdapter]]:
        """List all registered adapters."""
        return self._adapters.copy()


# Global registry
_registry = AdapterRegistry()


def register_adapter(name: str, adapter_class: type[BaseAdapter]) -> None:
    """Register adapter globally."""
    _registry.register(name, adapter_class)


def get_adapter(name: str) -> type[BaseAdapter] | None:
    """Get adapter from global registry."""
    return _registry.get(name)


def get_registry() -> AdapterRegistry:
    """Get global registry instance."""
    return _registry
