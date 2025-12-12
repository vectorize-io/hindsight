import asyncio
import logging
from typing import Optional

from pg0 import Pg0

logger = logging.getLogger(__name__)

DEFAULT_PORT = 5555
DEFAULT_USERNAME = "hindsight"
DEFAULT_PASSWORD = "hindsight"
DEFAULT_DATABASE = "hindsight"


class EmbeddedPostgres:
    """Manages an embedded PostgreSQL server instance using pg0-embedded."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        database: str = DEFAULT_DATABASE,
        name: str = "hindsight",
        **kwargs,
    ):
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.name = name
        self._pg0: Optional[Pg0] = None

    def _get_pg0(self) -> Pg0:
        if self._pg0 is None:
            self._pg0 = Pg0(
                name=self.name,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
            )
        return self._pg0

    async def start(self, max_retries: int = 3, retry_delay: float = 2.0) -> str:
        """Start the PostgreSQL server with retry logic."""
        logger.info(f"Starting embedded PostgreSQL (name: {self.name}, port: {self.port})...")

        pg0 = self._get_pg0()
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(None, pg0.start)
                logger.info(f"PostgreSQL started on port {self.port}")
                return info.uri
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    logger.debug(f"pg0 start attempt {attempt}/{max_retries} failed: {last_error}")
                    logger.debug(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.debug(f"pg0 start attempt {attempt}/{max_retries} failed: {last_error}")

        raise RuntimeError(
            f"Failed to start embedded PostgreSQL after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def stop(self) -> None:
        """Stop the PostgreSQL server."""
        pg0 = self._get_pg0()
        logger.info(f"Stopping embedded PostgreSQL (name: {self.name})...")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, pg0.stop)
            logger.info("Embedded PostgreSQL stopped")
        except Exception as e:
            if "not running" in str(e).lower():
                return
            raise RuntimeError(f"Failed to stop PostgreSQL: {e}")

    async def get_uri(self) -> str:
        """Get the connection URI for the PostgreSQL server."""
        pg0 = self._get_pg0()
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, pg0.info)
        if info is None or not info.running:
            raise RuntimeError("PostgreSQL server is not running or URI not available")
        return info.uri

    async def is_running(self) -> bool:
        """Check if the PostgreSQL server is currently running."""
        try:
            pg0 = self._get_pg0()
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, pg0.info)
            return info is not None and info.running
        except Exception:
            return False

    async def ensure_running(self) -> str:
        """Ensure the PostgreSQL server is running, starting it if needed."""
        if await self.is_running():
            return await self.get_uri()
        return await self.start()


_default_instance: Optional[EmbeddedPostgres] = None


def get_embedded_postgres() -> EmbeddedPostgres:
    """Get or create the default EmbeddedPostgres instance."""
    global _default_instance
    if _default_instance is None:
        _default_instance = EmbeddedPostgres()
    return _default_instance


async def start_embedded_postgres() -> str:
    """Quick start function for embedded PostgreSQL."""
    return await get_embedded_postgres().ensure_running()


async def stop_embedded_postgres() -> None:
    """Stop the default embedded PostgreSQL instance."""
    global _default_instance
    if _default_instance:
        await _default_instance.stop()
