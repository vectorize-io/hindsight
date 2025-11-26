import asyncio
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("HINDSIGHT_API_PG0_DATA_DIR", Path.home() / ".hindsight" / "pg_data"))
DEFAULT_INSTALL_DIR = Path.home() / ".hindsight" / "bin"
BINARY_NAME = "pg0"
DEFAULT_PORT = 5555
DEFAULT_USERNAME = "hindsight"
DEFAULT_PASSWORD = "hindsight"
DEFAULT_DATABASE = "hindsight"


def get_platform_binary_name() -> str:
    """Get the appropriate binary name for the current platform.

    Supported platforms:
    - macOS ARM64 (darwin-aarch64)
    - Linux x86_64
    - Windows x86_64
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize architecture names
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        raise RuntimeError(
            f"Embedded PostgreSQL is not supported on architecture: {machine}. "
            f"Supported architectures: x86_64/amd64 (Linux, Windows), aarch64/arm64 (macOS)"
        )

    if system == "darwin" and arch == "aarch64":
        return "pg0-darwin-aarch64"
    elif system == "linux" and arch == "x86_64":
        return "pg0-linux-x86_64"
    elif system == "windows" and arch == "x86_64":
        return "pg0-windows-x86_64.exe"
    else:
        raise RuntimeError(
            f"Embedded PostgreSQL is not supported on {system}-{arch}. "
            f"Supported platforms: darwin-aarch64 (macOS ARM), linux-x86_64, windows-x86_64"
        )


def get_download_url(
    version: str = "latest",
    repo: str = "vectorize-io/pg0",
) -> str:
    """
    """
    # Check for direct URL override
    binary_name = get_platform_binary_name()

    if version == "latest":
        return f"https://github.com/{repo}/releases/latest/download/{binary_name}"
    else:
        return f"https://github.com/{repo}/releases/download/{version}/{binary_name}"


class EmbeddedPostgres:
    """
    Manages an embedded PostgreSQL server instance.

    This class handles:
    - Downloading and installing the embedded-postgres CLI
    - Starting/stopping the PostgreSQL server
    - Getting the connection URI

    Example:
        pg = EmbeddedPostgres(data_dir="~/.myapp/data")
        await pg.ensure_installed()
        await pg.start()
        uri = await pg.get_uri()
        # ... use uri with asyncpg ...
        await pg.stop()
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        install_dir: Optional[Path] = None,
        version: str = "latest",
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        database: str = DEFAULT_DATABASE,
        name: str = "hindsight",
    ):
        """
        Initialize the embedded PostgreSQL manager.

        Args:
            data_dir: Directory to store PostgreSQL data. Defaults to ~/.hindsight/pg_data
            install_dir: Directory to install the CLI binary. Defaults to ~/.hindsight/bin
            version: Version of embedded-postgres to use. Defaults to "latest"
            port: Port to listen on. Defaults to 5555
            username: Username for the database. Defaults to "hindsight"
            password: Password for the database. Defaults to "hindsight"
            database: Database name to create. Defaults to "hindsight"
            name: Instance name for pg0. Defaults to "hindsight"
        """
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR).expanduser()
        self.install_dir = Path(install_dir or DEFAULT_INSTALL_DIR).expanduser()
        self.version = version
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.name = name

        # Binary path
        binary_name = "pg0.exe" if platform.system() == "Windows" else "pg0"
        self.binary_path = self.install_dir / binary_name

        self._process: Optional[subprocess.Popen] = None

    def is_installed(self) -> bool:
        """Check if the embedded-postgres CLI is installed."""
        return self.binary_path.exists() and os.access(self.binary_path, os.X_OK)

    async def ensure_installed(self) -> None:
        """
        Ensure the embedded-postgres CLI is installed.

        Downloads and installs the binary if not already present.
        """
        if self.is_installed():
            logger.debug(f"pg0 already installed at {self.binary_path}")
            return

        logger.info("Installing pg0 CLI...")

        # Create install directory
        self.install_dir.mkdir(parents=True, exist_ok=True)

        # Download the binary
        download_url = get_download_url(self.version)
        logger.info(f"Downloading from {download_url}")

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
                response = await client.get(download_url)
                response.raise_for_status()

                # Write binary to disk
                with open(self.binary_path, "wb") as f:
                    f.write(response.content)

                # Make executable on Unix
                if platform.system() != "Windows":
                    st = os.stat(self.binary_path)
                    os.chmod(self.binary_path, st.st_mode | stat.S_IEXEC)

                logger.info(f"Installed pg0 to {self.binary_path}")

        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to download pg0: {e}") from e

    def _run_command(self, *args: str, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run an embedded-postgres command synchronously."""
        cmd = [str(self.binary_path), *args]

        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
        )

    async def _run_command_async(self, *args: str) -> tuple[int, str, str]:
        """Run an embedded-postgres command asynchronously."""
        cmd = [str(self.binary_path), *args]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    async def start(self) -> str:
        """
        Start the PostgreSQL server.

        Returns:
            The connection URI for the started server.

        Raises:
            RuntimeError: If the server fails to start.
        """
        if not self.is_installed():
            raise RuntimeError("pg0 is not installed. Call ensure_installed() first.")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting embedded PostgreSQL (name: {self.name}, data: {self.data_dir}, install: {self.install_dir}, port: {self.port})...")

        returncode, stdout, stderr = await self._run_command_async(
            "start",
            "--name", self.name,
            "--port", str(self.port),
            "--username", self.username,
            "--password", self.password,
            "--database", self.database,
            "--data-dir", self.data_dir.as_posix()
        )

        if returncode != 0:
            raise RuntimeError(f"Failed to start PostgreSQL: {stderr}")

        logger.info("Embedded PostgreSQL started")

        # Get and return the URI
        return await self.get_uri()

    async def stop(self) -> None:
        """
        Stop the PostgreSQL server.

        Raises:
            RuntimeError: If the server fails to stop.
        """
        if not self.is_installed():
            return

        logger.info(f"Stopping embedded PostgreSQL (name: {self.name})...")

        returncode, stdout, stderr = await self._run_command_async("stop", "--name", self.name)

        if returncode != 0:
            # Don't raise if server wasn't running
            if "not running" in stderr.lower():
                logger.debug("PostgreSQL was not running")
                return
            raise RuntimeError(f"Failed to stop PostgreSQL: {stderr}")

        logger.info("Embedded PostgreSQL stopped")

    async def _get_info(self) -> dict:
        """
        Get info from pg0 using the `info -o json` command.

        Returns:
            Dictionary with 'running' (bool) and 'uri' (str) keys.

        Raises:
            RuntimeError: If unable to get info.
        """
        if not self.is_installed():
            raise RuntimeError("pg0 is not installed.")

        returncode, stdout, stderr = await self._run_command_async(
            "info", "--name", self.name, "-o", "json")

        if returncode != 0:
            raise RuntimeError(f"Failed to get PostgreSQL info: {stderr}")

        try:
            return json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse pg0 info output: {e}")

    async def get_uri(self) -> str:
        """
        Get the connection URI for the PostgreSQL server.

        Returns:
            PostgreSQL connection URI (e.g., postgresql://user:pass@localhost:5432/db)

        Raises:
            RuntimeError: If unable to get the URI or server is not running.
        """
        info = await self._get_info()
        uri = info.get("uri")
        if not uri:
            raise RuntimeError("PostgreSQL server is not running or URI not available")
        return uri

    async def status(self) -> dict:
        """
        Get the status of the PostgreSQL server.

        Returns:
            Dictionary with status information including 'running' boolean and 'uri'.
        """
        if not self.is_installed():
            return {"installed": False, "running": False}

        try:
            info = await self._get_info()
            return {
                "installed": True,
                "running": info.get("running", False),
                "uri": info.get("uri"),
                "data_dir": str(self.data_dir),
                "binary_path": str(self.binary_path),
            }
        except RuntimeError:
            return {
                "installed": True,
                "running": False,
                "data_dir": str(self.data_dir),
                "binary_path": str(self.binary_path),
            }

    async def is_running(self) -> bool:
        """Check if the PostgreSQL server is currently running."""
        if not self.is_installed():
            return False
        try:
            info = await self._get_info()
            return info.get("running", False)
        except RuntimeError:
            return False

    async def ensure_running(self) -> str:
        """
        Ensure the PostgreSQL server is running.

        Installs if needed, starts if not running.

        Returns:
            The connection URI.
        """
        await self.ensure_installed()

        if await self.is_running():
            return await self.get_uri()

        return await self.start()

    def uninstall(self) -> None:
        """Remove the embedded-postgres binary."""
        if self.binary_path.exists():
            self.binary_path.unlink()
            logger.info(f"Removed {self.binary_path}")

    def clear_data(self) -> None:
        """Remove all PostgreSQL data (destructive!)."""
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
            logger.info(f"Removed data directory {self.data_dir}")


# Convenience functions for simple usage

_default_instance: Optional[EmbeddedPostgres] = None


def get_embedded_postgres(
    data_dir: Optional[Path] = None,
    install_dir: Optional[Path] = None,
) -> EmbeddedPostgres:
    """
    Get or create the default EmbeddedPostgres instance.

    Args:
        data_dir: Override default data directory
        install_dir: Override default install directory

    Returns:
        EmbeddedPostgres instance
    """
    global _default_instance

    if _default_instance is None or data_dir or install_dir:
        _default_instance = EmbeddedPostgres(
            data_dir=data_dir,
            install_dir=install_dir,
        )

    return _default_instance


async def start_embedded_postgres(
    data_dir: Optional[Path] = None,
) -> str:
    """
    Quick start function for embedded PostgreSQL.

    Downloads, installs, and starts PostgreSQL in one call.

    Args:
        data_dir: Directory to store PostgreSQL data

    Returns:
        Connection URI string

    Example:
        db_url = await start_embedded_postgres()
        conn = await asyncpg.connect(db_url)
    """
    pg = get_embedded_postgres(data_dir=data_dir)
    return await pg.ensure_running()


async def stop_embedded_postgres() -> None:
    """Stop the default embedded PostgreSQL instance."""
    global _default_instance

    if _default_instance:
        await _default_instance.stop()