from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
from pathlib import Path
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
        **kwargs,
    ):
        self.port = port  # None means pg0 will auto-assign
        self.username = username
        self.password = password
        self.database = database
        self.name = name
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

        # On Linux, check for missing shared libraries which produce misleading errors
        # from initdb (e.g. "program postgres is needed by initdb but was not found").
        missing_libs_hint = _check_postgres_missing_libs()
        if missing_libs_hint:
            raise RuntimeError(
                f"Failed to start embedded PostgreSQL after {max_retries} attempts.\n"
                f"{missing_libs_hint}\n"
                f"Last error: {last_error}"
            )

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


def _check_postgres_missing_libs() -> str | None:
    """Check if the embedded PostgreSQL binary is missing shared libraries.

    On some Linux distributions (e.g., Arch Linux), the pre-built PostgreSQL binary
    distributed by pg0-embedded may require libraries not installed by default
    (e.g., libxml2-legacy). When a required library is missing, initdb reports a
    misleading error: "program postgres is needed by initdb but was not found in the
    same directory". This function detects that root cause and returns a human-readable
    hint.

    Returns:
        A hint string describing the missing libraries and how to fix them,
        or None if no missing libraries are detected or the check cannot run.
    """
    if platform.system() != "Linux":
        return None

    pg0_base = Path.home() / ".pg0" / "installation"
    if not pg0_base.exists():
        return None

    postgres_binary: Path | None = None
    for version_dir in pg0_base.iterdir():
        candidate = version_dir / "bin" / "postgres"
        if candidate.exists():
            postgres_binary = candidate
            break

    if postgres_binary is None:
        return None

    try:
        result = subprocess.run(
            ["ldd", str(postgres_binary)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        missing = [line.strip() for line in result.stdout.splitlines() if "not found" in line]
        if not missing:
            return None

        hint = f"Missing shared libraries detected for {postgres_binary}:\n"
        for lib in missing:
            hint += f"  {lib}\n"
        hint += (
            "\nTo fix this on Arch Linux, install the required packages, e.g.:\n"
            "  sudo pacman -S libxml2-legacy\n"
            "On other distributions, install the equivalent packages for the missing libraries."
        )
        return hint
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


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


def parse_pg0_url(db_url: str) -> tuple[bool, str | None, int | None]:
    """
    Parse a database URL and check if it's a pg0:// embedded database URL.

    Supports:
    - "pg0" -> default instance "hindsight"
    - "pg0://instance-name" -> named instance
    - "pg0://instance-name:port" -> named instance with explicit port
    - Any other URL (e.g., postgresql://) -> not a pg0 URL

    Args:
        db_url: The database URL to parse

    Returns:
        Tuple of (is_pg0, instance_name, port)
        - is_pg0: True if this is a pg0 URL
        - instance_name: The instance name (or None if not pg0)
        - port: The explicit port (or None for auto-assign)
    """
    if db_url == "pg0":
        return True, "hindsight", None

    if db_url.startswith("pg0://"):
        url_part = db_url[6:]  # Remove "pg0://"
        if ":" in url_part:
            instance_name, port_str = url_part.rsplit(":", 1)
            return True, instance_name or "hindsight", int(port_str)
        else:
            return True, url_part or "hindsight", None

    return False, None, None


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
    is_pg0, instance_name, port = parse_pg0_url(db_url)
    if is_pg0:
        pg0 = EmbeddedPostgres(name=instance_name, port=port)
        return await pg0.ensure_running()
    return db_url
