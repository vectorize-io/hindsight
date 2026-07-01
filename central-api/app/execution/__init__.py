"""Execution ledger and lineage tracking."""

# Import adapters to register them
from app.execution.adapters import docker, github, ssh  # noqa: F401
