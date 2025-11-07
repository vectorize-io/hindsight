"""
Web interface for memory system.

Provides FastAPI app and visualization interface.
"""
from .server import app, create_app

__all__ = ["app", "create_app"]
