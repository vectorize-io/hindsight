"""
HTTP Extension for adding custom endpoints to the Hindsight API.

This extension allows adding custom HTTP endpoints under the /ext/ path prefix.
The extension provides a FastAPI router that is mounted on the main application.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fastapi import APIRouter

from hindsight_api.extensions.base import Extension

if TYPE_CHECKING:
    from hindsight_api import MemoryEngine


class HttpExtension(Extension, ABC):
    """
    Base class for HTTP extensions that add custom API endpoints.

    HTTP extensions provide a FastAPI router that gets mounted under /ext/.
    The extension has full control over the routes, request/response models, and handlers.

    Example:
        ```python
        from fastapi import APIRouter
        from hindsight_api.extensions import HttpExtension

        class MyHttpExtension(HttpExtension):
            def get_router(self, memory: MemoryEngine) -> APIRouter:
                router = APIRouter()

                @router.get("/hello")
                async def hello():
                    return {"message": "Hello from extension!"}

                @router.post("/custom/{bank_id}/action")
                async def custom_action(bank_id: str):
                    # Access memory engine for database operations
                    pool = await memory._get_pool()
                    # ... custom logic
                    return {"status": "ok"}

                return router
        ```

    The routes will be available at:
        - GET /ext/hello
        - POST /ext/custom/{bank_id}/action

    Configuration via environment variables:
        HINDSIGHT_API_HTTP_EXTENSION=mypackage.ext:MyHttpExtension
        HINDSIGHT_API_HTTP_SOME_CONFIG=value

    The extension receives config: {"some_config": "value"}
    """

    @abstractmethod
    def get_router(self, memory: "MemoryEngine") -> APIRouter:
        """
        Return a FastAPI router with custom endpoints.

        The router will be mounted at /ext/ on the main application.
        All routes defined in the router will be prefixed with /ext/.

        Args:
            memory: The MemoryEngine instance for database access and core operations.
                   Use this to access the connection pool, run queries, or call
                   memory operations like retain, recall, etc.

        Returns:
            A FastAPI APIRouter with the custom endpoints defined.

        Example:
            ```python
            def get_router(self, memory: MemoryEngine) -> APIRouter:
                router = APIRouter(tags=["My Extension"])

                @router.get("/status")
                async def status():
                    health = await memory.health_check()
                    return {"extension": "healthy", "memory": health}

                return router
            ```
        """
        pass
