"""Execution adapters package."""

from app.execution.adapters.base import (
    BaseAdapter,
    AdapterRegistry,
    register_adapter,
    get_adapter,
    get_registry,
)
from app.execution.adapters.docker import DockerAdapter

# Register default adapters
register_adapter("docker_deploy", DockerAdapter)

__all__ = [
    "BaseAdapter",
    "AdapterRegistry",
    "register_adapter",
    "get_adapter",
    "get_registry",
    "DockerAdapter",
]
