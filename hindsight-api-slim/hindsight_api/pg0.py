from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pg0 import Pg0

logger = logging.getLogger(__name__)

DEFAULT_USERNAME = "hindsight"
DEFAULT_PASSWORD = "hindsight"
DEFAULT_DATABASE = "hindsight"


class EmbeddedPostgres:
    """Manages an embedded PostgreSQL server instance using pg0-embedded."""

    def __init__(
        self,
        port: int | None = None,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        database: str = DEFAULT_DATABASE,
        name: str = "hindsight",
        config: dict[str, str] | None = None,
        **kwargs,
    ):
        self.port = port  # None means pg0 will auto-assign
        self.username = username
        self.password = password
        self.database = database
        self.name = name
        # Extra postgresql.conf settings forwarded to Pg0 (e.g. ``max_connections``).
        # Useful when tests spawn many xdist workers that each open a pool against
        # the same pg0 instance — the postgres default of 100 max_connections is
        # easy to exhaust under that fan-out.
        self.config = config
        self._pg0: Pg0 | None = None

    def _get_pg0(self) -> Pg0:
        if self._pg0 is None:
            try:
                from pg0 import Pg0
            except ImportError:
                raise ImportError(
                    "pg0-embedded is required for embedded PostgreSQL. "
                    "Install it with: pip install 'hindsight-api-slim[embedded-db]'"
                )
            kwargs = {
                "name": self.name,
                "username": self.username,
                "password": self.password,
                "database": self.database,
            }
            # Only set port if explicitly specified
            if self.port is not None:
                kwargs["port"] = self.port
            if self.config is not None:
                kwargs["config"] = self.config
            self._pg0 = Pg0(**kwargs)
        return self._pg0

    async def start(self, max_retries: int = 5, retry_delay: float = 4.0) -> str:
        """Start the PostgreSQL server with retry logic."""
        port_info = f"port={self.port}" if self.port else "port=auto"
        logger.info(f"Starting embedded PostgreSQL (name={self.name}, {port_info})...")

        pg0 = self._get_pg0()
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(None, pg0.start)
                # Get URI from pg0 (includes auto-assigned port)
                uri = info.uri
                logger.info(f"PostgreSQL started: {uri}")
                return uri
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
            f"Failed to start embedded PostgreSQL after {max_retries} attempts. Last error: {last_error}"
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


_default_instance: EmbeddedPostgres | None = None


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


@dataclass(frozen=True)
class Pg0Url:
    """Parsed representation of a ``pg0`` embedded-database URL.

    ``username``/``password`` are ``None`` when the URL omits credentials, in
    which case the pg0 defaults (``hindsight``/``hindsight``) apply.
    """

    is_pg0: bool
    instance_name: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None


def parse_pg0_url(db_url: str) -> Pg0Url:
    """
    Parse a database URL and check if it's a pg0:// embedded database URL.

    Supports:
    - "pg0" -> default instance "hindsight"
    - "pg0://instance-name" -> named instance
    - "pg0://instance-name:port" -> named instance with explicit port
    - "pg0://user:pwd@instance-name:port" -> named instance with credentials
      (``user`` or ``user:pwd``; either half may be present)
    - Any other URL (e.g., postgresql://) -> not a pg0 URL

    Args:
        db_url: The database URL to parse

    Returns:
        A :class:`Pg0Url`. When ``is_pg0`` is False the remaining fields are None.
    """
    if db_url == "pg0":
        return Pg0Url(is_pg0=True, instance_name="hindsight")

    if not db_url.startswith("pg0://"):
        return Pg0Url(is_pg0=False)

    url_part = db_url[6:]  # Remove "pg0://"

    # Split optional "user:pwd@" credentials from the "instance:port" host part.
    # rsplit on the last "@" so passwords may contain "@".
    username: str | None = None
    password: str | None = None
    if "@" in url_part:
        creds, url_part = url_part.rsplit("@", 1)
        user_part, sep, pwd_part = creds.partition(":")
        username = user_part or None
        password = pwd_part if sep else None

    if ":" in url_part:
        instance_name, port_str = url_part.rsplit(":", 1)
        port: int | None = int(port_str)
    else:
        instance_name, port = url_part, None

    return Pg0Url(
        is_pg0=True,
        instance_name=instance_name or "hindsight",
        port=port,
        username=username,
        password=password,
    )


async def resolve_database_url(db_url: str) -> str:
    """
    Resolve a database URL, handling pg0:// embedded database URLs.

    If the URL is a pg0:// URL, starts the embedded PostgreSQL and returns
    the actual postgresql:// connection URL. Otherwise, returns the URL unchanged.

    Args:
        db_url: Database URL (pg0://, pg0, or postgresql://)

    Returns:
        The resolved postgresql:// connection URL
    """
    parsed = parse_pg0_url(db_url)
    if parsed.is_pg0:
        kwargs: dict[str, object] = {"name": parsed.instance_name, "port": parsed.port}
        if parsed.username is not None:
            kwargs["username"] = parsed.username
        if parsed.password is not None:
            kwargs["password"] = parsed.password
        pg0 = EmbeddedPostgres(**kwargs)
        return await pg0.ensure_running()
    return db_url
