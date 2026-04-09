"""
Built-in llama.cpp LLM provider for fully offline operation.

Manages a llama-cpp-python server as a subprocess, downloads GGUF models
from HuggingFace on first use, and delegates inference to the OpenAI-compatible API.

Usage:
    HINDSIGHT_API_LLM_PROVIDER=llamacpp
    HINDSIGHT_API_LLAMACPP_MODEL_PATH=~/.hindsight/models/gemma-4-E2B-it-Q4_K_M.gguf
    HINDSIGHT_API_LLAMACPP_GPU_LAYERS=-1  # -1 = all layers on GPU
    HINDSIGHT_API_LLAMACPP_CONTEXT_SIZE=8192
"""

import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hindsight_api.engine.llm_interface import LLMInterface
from hindsight_api.engine.response_models import LLMToolCallResult

logger = logging.getLogger(__name__)

# Default GGUF model for offline mode
DEFAULT_LLAMACPP_HF_REPO = "bartowski/google_gemma-4-E2B-it-GGUF"
DEFAULT_LLAMACPP_HF_FILENAME = "google_gemma-4-E2B-it-Q4_K_M.gguf"
DEFAULT_LLAMACPP_MODEL_ALIAS = "gemma-4-e2b-it"

MODELS_DIR = Path.home() / ".hindsight" / "models"

# Singleton server instance — shared across all LlamaCppLLM instances
# (retain, reflect, consolidation each create their own LLMProvider,
# but they should all share one llama.cpp server process)
_shared_server: "LlamaCppServer | None" = None
_shared_server_lock = asyncio.Lock()


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _download_default_model() -> Path:
    """Download the default GGUF model from HuggingFace if not already cached.

    Returns:
        Path to the downloaded GGUF file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required for automatic model download. "
            "Install with: pip install 'hindsight-api-slim[local-llm]'"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / DEFAULT_LLAMACPP_HF_FILENAME

    if target.exists():
        logger.info(f"Using cached model: {target}")
        return target

    logger.info(
        f"Downloading {DEFAULT_LLAMACPP_HF_FILENAME} from {DEFAULT_LLAMACPP_HF_REPO} (~3.5 GB, first run only)..."
    )

    downloaded = hf_hub_download(
        repo_id=DEFAULT_LLAMACPP_HF_REPO,
        filename=DEFAULT_LLAMACPP_HF_FILENAME,
        local_dir=str(MODELS_DIR),
    )

    logger.info(f"Model downloaded: {downloaded}")
    return Path(downloaded)


def _resolve_model_path(model_path: str | None) -> Path:
    """Resolve the model path, downloading the default if needed.

    Args:
        model_path: Explicit path to a GGUF file, or None to use the default.

    Returns:
        Resolved Path to the GGUF file.
    """
    if model_path:
        p = Path(model_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {p}\n"
                f"Set HINDSIGHT_API_LLAMACPP_MODEL_PATH to a valid .gguf file, "
                f"or remove the setting to auto-download the default model."
            )
        return p

    return _download_default_model()


class LlamaCppServer:
    """Manages a llama-cpp-python OpenAI-compatible server as a subprocess."""

    def __init__(
        self,
        model_path: Path,
        port: int,
        gpu_layers: int = -1,
        context_size: int = 8192,
        chat_format: str | None = None,
        extra_args: str | None = None,
    ):
        self.model_path = model_path
        self.port = port
        self.gpu_layers = gpu_layers
        self.context_size = context_size
        self.chat_format = chat_format
        self.extra_args = extra_args
        self._process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def start(self) -> None:
        """Start the llama.cpp server subprocess."""
        cmd = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--n_gpu_layers",
            str(self.gpu_layers),
            "--n_ctx",
            str(self.context_size),
            "--flash_attn",
            "true",
            "--n_batch",
            "2048",
            # Prompt cache: reuse KV cache for repeated system prompts
            "--cache",
            "true",
        ]
        # Only pass chat_format if explicitly set (most GGUF models have it embedded)
        if self.chat_format:
            cmd.extend(["--chat_format", self.chat_format])
        # User-provided extra args (e.g. "--type_k 1 --type_v 1 --n_threads 8")
        if self.extra_args:
            cmd.extend(self.extra_args.split())

        logger.info(f"Starting llama.cpp server: {' '.join(cmd)}")

        # Write stderr to a log file to avoid pipe buffer deadlock
        # (llama.cpp outputs a lot of model metadata on stderr during loading)
        self._log_path = MODELS_DIR / "llamacpp_server.log"
        self._log_file = open(self._log_path, "w")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
            # Ensure the subprocess is killed when the parent exits
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        # Wait for the server to be ready
        await self._wait_for_ready()

    async def _wait_for_ready(self, timeout: float = 120.0) -> None:
        """Wait for the llama.cpp server to accept connections."""
        import httpx

        start = time.monotonic()
        url = f"http://127.0.0.1:{self.port}/v1/models"
        last_log = start

        while time.monotonic() - start < timeout:
            # Check if process died
            if self._process and self._process.poll() is not None:
                stderr = ""
                try:
                    stderr = self._log_path.read_text()[-2000:]
                except Exception:
                    pass
                raise RuntimeError(f"llama.cpp server exited with code {self._process.returncode}.\nstderr: {stderr}")

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=5.0)
                    if resp.status_code == 200:
                        logger.info(f"llama.cpp server ready on port {self.port}")
                        return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout):
                pass

            # Log progress every 15s
            now = time.monotonic()
            if now - last_log > 15:
                elapsed = int(now - start)
                logger.info(f"Waiting for llama.cpp server to load model... ({elapsed}s)")
                last_log = now

            await asyncio.sleep(1.0)

        # Timeout — read the log to help debug
        stderr = ""
        try:
            stderr = self._log_path.read_text()[-2000:]
        except Exception:
            pass
        raise TimeoutError(
            f"llama.cpp server did not become ready within {timeout}s.\n"
            f"Check model compatibility and available memory.\n"
            f"Server log: {stderr}"
        )

    async def stop(self) -> None:
        """Stop the llama.cpp server subprocess."""
        if self._process is None:
            return

        logger.info("Stopping llama.cpp server...")
        try:
            # Send SIGTERM to the process group
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()

            # Wait up to 10s for graceful shutdown
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
                self._process.wait(timeout=5)
        except (ProcessLookupError, OSError):
            pass  # Process already exited
        finally:
            self._process = None
            if hasattr(self, "_log_file") and self._log_file:
                self._log_file.close()
                self._log_file = None
            logger.info("llama.cpp server stopped")


class LlamaCppLLM(LLMInterface):
    """
    Built-in llama.cpp provider.

    Manages a llama-cpp-python server subprocess and delegates to OpenAICompatibleLLM
    for actual inference calls. Handles model downloading and server lifecycle.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        model_path: str | None = None,
        gpu_layers: int = -1,
        context_size: int = 8192,
        chat_format: str | None = None,
        no_grammar: bool = False,
        extra_args: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            provider=provider,
            api_key=api_key or "llamacpp",
            base_url=base_url or "",
            model=model or DEFAULT_LLAMACPP_MODEL_ALIAS,
            reasoning_effort=reasoning_effort,
        )
        self._model_path_str = model_path
        self._gpu_layers = gpu_layers
        self._context_size = context_size
        self._chat_format = chat_format
        self._no_grammar = no_grammar
        self._extra_args = extra_args
        self._server: LlamaCppServer | None = None
        self._delegate: Any = None  # OpenAICompatibleLLM, created after server starts
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization: download model + start shared server on first use."""
        if self._initialized:
            return

        global _shared_server

        from .openai_compatible_llm import OpenAICompatibleLLM

        async with _shared_server_lock:
            if _shared_server is None:
                # Resolve and potentially download the model
                model_path = _resolve_model_path(self._model_path_str)
                logger.info(f"Using GGUF model: {model_path}")

                # Start the shared llama.cpp server
                port = _find_free_port()
                _shared_server = LlamaCppServer(
                    model_path=model_path,
                    port=port,
                    gpu_layers=self._gpu_layers,
                    context_size=self._context_size,
                    chat_format=self._chat_format,
                    extra_args=self._extra_args,
                )
                await _shared_server.start()

        self._server = _shared_server

        # Create the delegate that talks to the shared server's OpenAI-compatible API
        if self._no_grammar:
            logger.info("Grammar enforcement disabled (HINDSIGHT_API_LLAMACPP_NO_GRAMMAR=true)")
        self._delegate = OpenAICompatibleLLM(
            provider="llamacpp",
            api_key="llamacpp",
            base_url=self._server.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

        self._initialized = True

    async def verify_connection(self) -> None:
        """Verify the llama.cpp server is running and can generate text."""
        await self._ensure_initialized()
        # Make a simple test call to verify the model can actually generate
        await self._delegate.call(
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_completion_tokens=10,
            max_retries=2,
            initial_backoff=0.5,
            max_backoff=2.0,
            scope="verification",
        )
        logger.info("llama.cpp LLM verification passed")

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        return_usage: bool = False,
    ) -> Any:
        """Delegate call to the OpenAI-compatible API."""
        await self._ensure_initialized()
        return await self._delegate.call(
            messages=messages,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            scope=scope,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            skip_validation=skip_validation,
            strict_schema=strict_schema,
            return_usage=return_usage,
        )

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMToolCallResult:
        """Delegate tool calls to the OpenAI-compatible API."""
        await self._ensure_initialized()
        return await self._delegate.call_with_tools(
            messages=messages,
            tools=tools,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            scope=scope,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            tool_choice=tool_choice,
        )

    async def cleanup(self) -> None:
        """Stop the shared llama.cpp server."""
        global _shared_server

        if self._delegate:
            await self._delegate.cleanup()
            self._delegate = None

        # Stop the shared server (only the first cleanup call actually stops it)
        async with _shared_server_lock:
            if _shared_server is not None:
                await _shared_server.stop()
                _shared_server = None

        self._server = None
        self._initialized = False
