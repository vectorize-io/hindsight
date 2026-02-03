"""
Version runner for upgrade tests.

Manages running different git versions of the Hindsight API for upgrade testing.
Handles git checkout, venv creation, dependency installation, and server lifecycle.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """Information about a running server."""

    url: str
    port: int
    version: str


class VersionRunner:
    """
    Manages running a specific git version of the Hindsight API.

    For "HEAD" or "current", uses the current working directory.
    For git tags (e.g., "v0.3.0"), clones the repo at that tag to a temp directory.
    """

    def __init__(
        self,
        version: str,
        db_url: str,
        port: int = 8890,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
    ):
        """
        Initialize a version runner.

        Args:
            version: Git tag (e.g., "v0.3.0") or "HEAD"/"current" for current code
            db_url: PostgreSQL connection URL
            port: Port to run the API on
            llm_provider: LLM provider (defaults to env var)
            llm_api_key: LLM API key (defaults to env var)
            llm_model: LLM model (defaults to env var)
        """
        self.version = version
        self.db_url = db_url
        self.port = port
        self.llm_provider = llm_provider or os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq")
        self.llm_api_key = llm_api_key or os.getenv("HINDSIGHT_API_LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        self.llm_model = llm_model or os.getenv("HINDSIGHT_API_LLM_MODEL", "llama-3.3-70b-versatile")

        self.work_dir: Path | None = None
        self.process: subprocess.Popen | None = None
        self._temp_dir: str | None = None
        self._is_current = version.lower() in ("head", "current")
        self.log_file: Path | None = None
        self._log_handle = None

    def _find_repo_root(self) -> Path:
        """Find the git repository root."""
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())

    def setup(self) -> None:
        """Checkout version and install dependencies."""
        if self._is_current:
            # Use current working directory
            self.work_dir = self._find_repo_root()
            logger.info(f"Using current code at {self.work_dir}")
            return

        # Create temp dir and checkout specific version
        self._temp_dir = tempfile.mkdtemp(prefix=f"hindsight-{self.version}-")
        self.work_dir = Path(self._temp_dir)

        repo_root = self._find_repo_root()
        logger.info(f"Cloning {repo_root} at {self.version} to {self.work_dir}")

        # Shallow clone at specific tag
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", self.version, str(repo_root), str(self.work_dir)],
            check=True,
            capture_output=True,
        )

        # Create venv and install
        venv_path = self.work_dir / ".venv-upgrade-test"
        logger.info(f"Creating venv at {venv_path}")

        subprocess.run(["uv", "venv", str(venv_path)], check=True, capture_output=True)

        api_path = self.work_dir / "hindsight-api"
        logger.info(f"Installing hindsight-api from {api_path}")

        # Install with uv pip - use --index-strategy for pytorch
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-e",
                str(api_path),
                "--python",
                str(venv_path / "bin" / "python"),
                "--index-strategy",
                "unsafe-best-match",
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "UV_INDEX": "pytorch=https://download.pytorch.org/whl/cpu"},
        )

        logger.info(f"Version {self.version} setup complete")

    def _get_venv_path(self) -> Path:
        """Get the path to the venv for this version."""
        if self._is_current:
            # For current code, the venv is at the workspace root (uv workspace layout)
            # Check both possible locations
            workspace_venv = self.work_dir / ".venv"
            api_venv = self.work_dir / "hindsight-api" / ".venv"

            if (workspace_venv / "bin" / "hindsight-api").exists():
                return workspace_venv
            elif (api_venv / "bin" / "hindsight-api").exists():
                return api_venv
            else:
                # Default to workspace root
                return workspace_venv
        return self.work_dir / ".venv-upgrade-test"

    def start(self) -> ServerInfo:
        """
        Start the API server.

        Returns:
            ServerInfo with the URL and port
        """
        venv_path = self._get_venv_path()
        hindsight_api_bin = venv_path / "bin" / "hindsight-api"

        if not hindsight_api_bin.exists():
            raise RuntimeError(f"hindsight-api binary not found at {hindsight_api_bin}")

        env = os.environ.copy()
        env.update(
            {
                "HINDSIGHT_API_PORT": str(self.port),
                "HINDSIGHT_API_DATABASE_URL": self.db_url,
                "HINDSIGHT_API_HOST": "127.0.0.1",
                "HINDSIGHT_API_LLM_PROVIDER": self.llm_provider,
                "HINDSIGHT_API_LLM_API_KEY": self.llm_api_key or "",
                "HINDSIGHT_API_LLM_MODEL": self.llm_model,
                "PYTHONUNBUFFERED": "1",
            }
        )

        logger.info(f"Starting {self.version} API on port {self.port}")
        logger.info(f"Database URL: {self.db_url}")

        # Determine working directory
        # For HEAD/current, use a temp directory to avoid .env file from workspace root
        # (hindsight-api loads .env with override=True which would override our env vars)
        if self._is_current:
            # Create a temp directory for HEAD to avoid workspace .env
            self._head_cwd = tempfile.mkdtemp(prefix="hindsight-head-cwd-")
            cwd = self._head_cwd
        else:
            cwd = str(self.work_dir)
            self._head_cwd = None

        # Create log file for server output
        version_slug = self.version.replace("/", "-").replace(".", "-")
        self.log_file = Path(f"/tmp/upgrade-test-{version_slug}-{self.port}.log")
        self._log_handle = open(self.log_file, "w")
        logger.info(f"Server logs will be written to {self.log_file}")

        # Start the server
        self.process = subprocess.Popen(
            [str(hindsight_api_bin)],
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            cwd=cwd,
        )

        self._wait_healthy()

        url = f"http://127.0.0.1:{self.port}"
        logger.info(f"Server {self.version} ready at {url}")

        return ServerInfo(url=url, port=self.port, version=self.version)

    def _wait_healthy(self, timeout: int = 120) -> None:
        """Wait for /health endpoint to respond."""
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + timeout

        while time.time() < deadline:
            # Check if process is still alive
            if self.process and self.process.poll() is not None:
                logs = self._read_logs()
                raise RuntimeError(f"Server {self.version} exited unexpectedly.\nLogs:\n{logs}")

            try:
                resp = httpx.get(url, timeout=2)
                if resp.status_code == 200:
                    return
            except httpx.RequestError:
                pass

            time.sleep(1)

        # Timeout - dump logs
        if self.process:
            self.process.terminate()
            logs = self._read_logs()
            raise TimeoutError(f"Server {self.version} not healthy after {timeout}s.\nLogs:\n{logs}")

    def _read_logs(self) -> str:
        """Read current logs from the log file."""
        if self.log_file and self.log_file.exists():
            try:
                # Flush the file handle first
                if self._log_handle:
                    self._log_handle.flush()
                return self.log_file.read_text()
            except Exception as e:
                return f"(failed to read logs: {e})"
        return "(no log file found)"

    def stop(self) -> None:
        """Stop the server and cleanup temp directory."""
        if self.process:
            logger.info(f"Stopping {self.version} server")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Server {self.version} did not stop gracefully, killing")
                self.process.kill()
                self.process.wait()
            self.process = None

        # Close log file handle
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

        if self._temp_dir and os.path.exists(self._temp_dir):
            logger.info(f"Cleaning up {self._temp_dir}")
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        # Clean up HEAD's temp cwd
        if hasattr(self, "_head_cwd") and self._head_cwd and os.path.exists(self._head_cwd):
            shutil.rmtree(self._head_cwd, ignore_errors=True)
            self._head_cwd = None

    def get_logs(self) -> str:
        """Get current server logs."""
        return self._read_logs()

    def __enter__(self) -> "VersionRunner":
        self.setup()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
