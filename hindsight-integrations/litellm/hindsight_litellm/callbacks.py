"""LiteLLM callback handlers for Hindsight memory integration.

This module implements LiteLLM's CustomLogger interface to intercept
LLM calls and integrate with Hindsight for memory injection and storage.

Uses direct HTTP calls via requests/httpx to avoid async event loop conflicts
when the hindsight_client's async methods are called from LiteLLM callbacks.
"""

import logging
import fnmatch
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import asyncio
import threading
import concurrent.futures

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse

from .config import get_config, is_configured, HindsightConfig, MemoryInjectionMode

# Use requests for sync HTTP calls to avoid async event loop issues
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


logger = logging.getLogger(__name__)

# Thread pool for running async operations in background
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="hindsight-")


class HindsightCallback(CustomLogger):
    """LiteLLM custom logger that integrates with Hindsight memory system.

    This callback handler:
    1. Injects relevant memories into prompts before LLM calls
    2. Stores conversations to Hindsight after successful LLM calls

    Features:
    - Works with 100+ LLM providers via LiteLLM
    - Deduplication to avoid storing duplicate conversations
    - Configurable memory injection modes
    - Support for entity observations in recall

    Usage:
        >>> from hindsight_litellm import configure, enable
        >>> configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
        >>> enable()
        >>>
        >>> # Now all LiteLLM calls will have memory integration
        >>> import litellm
        >>> response = litellm.completion(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "What did we discuss?"}]
        ... )
    """

    def __init__(self):
        """Initialize the Hindsight callback handler."""
        super().__init__()
        self._http_session = None
        self._http_lock = threading.Lock()
        # Track recently stored conversation hashes for deduplication
        self._recent_hashes: Set[str] = set()
        self._max_hash_cache = 1000

    def _get_http_session(self):
        """Get or create a requests Session (thread-safe)."""
        if self._http_session is None:
            with self._http_lock:
                if self._http_session is None:
                    if HAS_REQUESTS:
                        self._http_session = requests.Session()
                    elif HAS_HTTPX:
                        self._http_session = httpx.Client(timeout=30.0)
                    else:
                        raise RuntimeError(
                            "Neither 'requests' nor 'httpx' is installed. "
                            "Please install one: pip install requests"
                        )
        return self._http_session

    def _http_post(self, url: str, json_data: dict, config: HindsightConfig) -> Optional[dict]:
        """Make a synchronous HTTP POST request."""
        try:
            session = self._get_http_session()
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            if HAS_REQUESTS:
                response = session.post(url, json=json_data, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            elif HAS_HTTPX:
                response = session.post(url, json=json_data, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            if config.verbose:
                logger.warning(f"HTTP POST failed: {e}")
            return None

    def _should_skip_model(self, model: str, config: HindsightConfig) -> bool:
        """Check if this model should be excluded from interception."""
        for pattern in config.excluded_models:
            if fnmatch.fnmatch(model.lower(), pattern.lower()):
                return True
        return False

    def _extract_user_query(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the user's query from the last user message."""
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle structured content (e.g., vision messages)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        return " ".join(text_parts)
        return None

    def _compute_conversation_hash(
        self,
        user_input: str,
        assistant_output: str,
    ) -> str:
        """Compute a hash for deduplication."""
        content = f"{user_input.strip().lower()}|{assistant_output.strip().lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _is_duplicate(self, conv_hash: str) -> bool:
        """Check if this conversation was recently stored."""
        if conv_hash in self._recent_hashes:
            return True

        # Add to cache, evict oldest if full
        self._recent_hashes.add(conv_hash)
        if len(self._recent_hashes) > self._max_hash_cache:
            # Remove oldest (arbitrary since set, but good enough)
            self._recent_hashes.pop()

        return False

    def _format_memories(
        self,
        results: List[Any],
        config: HindsightConfig
    ) -> str:
        """Format memory recall results into a context string.

        Results can be RecallResult objects (with .text, .type attributes)
        or dicts (with get() method).
        """
        if not results:
            return ""

        # Apply limit if set, otherwise use all results
        results_to_use = results[:config.max_memories] if config.max_memories else results
        memory_lines = []
        for i, result in enumerate(results_to_use, 1):
            # Handle both RecallResult objects and dicts
            if hasattr(result, 'text'):
                text = result.text or ""
                fact_type = getattr(result, 'type', 'world') or "world"
                weight = getattr(result, 'weight', 0.0) or 0.0
            else:
                text = result.get("text", "")
                fact_type = result.get("type", result.get("fact_type", "world"))
                weight = result.get("weight", 0.0)

            if text:
                # Include metadata for context
                type_label = fact_type.upper() if fact_type else "MEMORY"
                line = f"{i}. [{type_label}] {text}"
                if weight > 0 and config.verbose:
                    line += f" (relevance: {weight:.2f})"
                memory_lines.append(line)

        if not memory_lines:
            return ""

        return (
            "# Relevant Memories\n"
            "The following information from memory may be relevant:\n\n"
            + "\n".join(memory_lines)
        )

    def _inject_memories_into_messages(
        self,
        messages: List[Dict[str, Any]],
        memory_context: str,
        config: HindsightConfig,
    ) -> List[Dict[str, Any]]:
        """Inject memory context into the messages list."""
        if not memory_context:
            return messages

        updated_messages = list(messages)  # Make a copy

        if config.injection_mode == MemoryInjectionMode.SYSTEM_MESSAGE:
            # Find existing system message or create new one
            for i, msg in enumerate(updated_messages):
                if msg.get("role") == "system":
                    # Append to existing system message
                    existing_content = msg.get("content", "")
                    updated_messages[i] = {
                        **msg,
                        "content": f"{existing_content}\n\n{memory_context}"
                    }
                    return updated_messages

            # No system message found, prepend one
            updated_messages.insert(0, {
                "role": "system",
                "content": memory_context
            })

        elif config.injection_mode == MemoryInjectionMode.PREPEND_USER:
            # Find the last user message and prepend context
            for i in range(len(updated_messages) - 1, -1, -1):
                if updated_messages[i].get("role") == "user":
                    original_content = updated_messages[i].get("content", "")
                    if isinstance(original_content, str):
                        updated_messages[i] = {
                            **updated_messages[i],
                            "content": f"{memory_context}\n\n---\n\n{original_content}"
                        }
                    break

        return updated_messages

    def _get_bank_id(self, config: HindsightConfig) -> str:
        """Get the bank_id for API calls."""
        return config.bank_id

    def _recall_memories_sync(
        self,
        query: str,
        config: HindsightConfig
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories from Hindsight (sync) using direct HTTP."""
        try:
            bank_id = self._get_bank_id(config)
            url = f"{config.hindsight_api_url}/v1/default/banks/{bank_id}/memories/recall"

            request_data = {
                "query": query,
                "budget": config.recall_budget or "mid",
                "max_tokens": config.max_memory_tokens or 4096,
            }
            if config.fact_types:
                request_data["types"] = config.fact_types

            response = self._http_post(url, request_data, config)
            if response and "results" in response:
                return response["results"]
            return []

        except Exception as e:
            if config.verbose:
                logger.warning(f"Failed to recall memories: {e}")
            return []

    async def _recall_memories_async(
        self,
        query: str,
        config: HindsightConfig
    ) -> List[Any]:
        """Recall relevant memories from Hindsight (async).

        Uses thread pool executor with sync HTTP to avoid event loop conflicts.
        """
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                _executor,
                self._recall_memories_sync,
                query,
                config
            )

            return results if isinstance(results, list) else []

        except Exception as e:
            if config.verbose:
                logger.warning(f"Failed to recall memories: {e}")
            return []

    def _store_conversation_sync(
        self,
        messages: List[Dict[str, Any]],
        response: ModelResponse,
        model: str,
        config: HindsightConfig,
    ) -> None:
        """Store the conversation to Hindsight (sync) using direct HTTP.

        By default, stores the full conversation history passed to the LLM.
        Each message is stored as a separate item, all linked by document_id.

        Hindsight will process the document as a whole for memory extraction.
        """
        try:
            # Extract assistant response from the LLM response
            assistant_output = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and choice.message:
                    assistant_output = choice.message.content or ""

            if not assistant_output:
                return

            # Build conversation items - each message becomes a separate item
            # All linked by document_id for Hindsight to process together
            items = []
            for msg in messages:
                role = msg.get("role", "").upper()
                content = msg.get("content", "")

                # Skip system messages - they're instructions, not conversation
                if role == "SYSTEM":
                    continue

                # Skip if this looks like our injected memory context
                if isinstance(content, str) and content.startswith("# Relevant Memories"):
                    continue

                # Handle structured content (e.g., vision messages)
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = " ".join(text_parts)

                if content:
                    # Map roles to clearer labels
                    label = "USER" if role == "USER" else "ASSISTANT"
                    items.append(f"{label}: {content}")

            # Add the new assistant response
            items.append(f"ASSISTANT: {assistant_output}")

            if not items:
                return

            # Use last user message for deduplication hash
            user_input = self._extract_user_query(messages) or ""

            # Deduplication check
            conv_hash = self._compute_conversation_hash(user_input, assistant_output)
            if self._is_duplicate(conv_hash):
                if config.verbose:
                    logger.debug(f"Skipping duplicate conversation: {conv_hash}")
                return

            # Build the full conversation as a single item for now
            # (Future: could store each message as separate item in same document)
            conversation_text = "\n\n".join(items)

            # Build metadata
            metadata = {
                "source": "litellm",
                "model": model,
            }

            # Add token usage if available
            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    metadata["tokens"] = str(response.usage.total_tokens)

            bank_id = self._get_bank_id(config)
            url = f"{config.hindsight_api_url}/v1/default/banks/{bank_id}/memories"

            request_data = {
                "items": [
                    {
                        "content": conversation_text,
                        "context": f"conversation:litellm:{model}",
                        "metadata": metadata,
                        "document_id": config.document_id,  # Group by document
                    }
                ],
            }

            self._http_post(url, request_data, config)

            if config.verbose:
                logger.info(f"Stored conversation to Hindsight bank: {config.bank_id}")

        except Exception as e:
            if config.verbose:
                logger.warning(f"Failed to store conversation: {e}")

    async def _store_conversation_async(
        self,
        messages: List[Dict[str, Any]],
        response: ModelResponse,
        model: str,
        config: HindsightConfig,
    ) -> None:
        """Store the conversation to Hindsight (async).

        Uses thread pool executor with sync HTTP to avoid event loop conflicts.
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                _executor,
                self._store_conversation_sync,
                messages,
                response,
                model,
                config
            )
        except Exception as e:
            if config.verbose:
                logger.warning(f"Failed to store conversation: {e}")

    # ========== LiteLLM CustomLogger Interface ==========

    def log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """Called before making the API call (sync).

        This is where we inject memories into the messages.
        """
        if not is_configured():
            return

        config = get_config()
        if not config or not config.enabled or not config.inject_memories:
            return

        if self._should_skip_model(model, config):
            return

        # Extract user query
        user_query = self._extract_user_query(messages)
        if not user_query:
            return

        # Recall relevant memories
        memories = self._recall_memories_sync(user_query, config)
        if not memories:
            return

        # Format and inject memories
        memory_context = self._format_memories(memories, config)
        updated_messages = self._inject_memories_into_messages(
            messages, memory_context, config
        )

        # Modify messages list IN-PLACE (don't just reassign kwargs)
        messages.clear()
        messages.extend(updated_messages)

        if config.verbose:
            logger.info(f"Injected {len(memories)} memories into prompt")

    async def async_log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """Called before making the API call (async).

        This is where we inject memories into the messages.
        """
        if not is_configured():
            return

        config = get_config()
        if not config or not config.enabled or not config.inject_memories:
            return

        if self._should_skip_model(model, config):
            return

        # Extract user query
        user_query = self._extract_user_query(messages)
        if not user_query:
            return

        # Recall relevant memories
        memories = await self._recall_memories_async(user_query, config)
        if not memories:
            return

        # Format and inject memories
        memory_context = self._format_memories(memories, config)
        updated_messages = self._inject_memories_into_messages(
            messages, memory_context, config
        )

        # Modify messages list IN-PLACE (don't just reassign kwargs)
        messages.clear()
        messages.extend(updated_messages)

        if config.verbose:
            logger.info(f"Injected {len(memories)} memories into prompt")

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after successful API call (sync).

        This is where we store the conversation.
        """
        if not is_configured():
            return

        config = get_config()
        if not config or not config.enabled or not config.store_conversations:
            return

        model = kwargs.get("model", "unknown")
        if self._should_skip_model(model, config):
            return

        messages = kwargs.get("messages", [])
        if not messages:
            return

        # Store the conversation
        self._store_conversation_sync(messages, response_obj, model, config)

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after successful API call (async).

        This is where we store the conversation.
        """
        if not is_configured():
            return

        config = get_config()
        if not config or not config.enabled or not config.store_conversations:
            return

        model = kwargs.get("model", "unknown")
        if self._should_skip_model(model, config):
            return

        messages = kwargs.get("messages", [])
        if not messages:
            return

        # Store the conversation
        await self._store_conversation_async(messages, response_obj, model, config)

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after failed API call (sync)."""
        # We don't store failed conversations
        pass

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after failed API call (async)."""
        # We don't store failed conversations
        pass

    def close(self) -> None:
        """Clean up resources."""
        with self._http_lock:
            if self._http_session is not None:
                try:
                    if HAS_REQUESTS:
                        self._http_session.close()
                    elif HAS_HTTPX:
                        self._http_session.close()
                except Exception:
                    pass
                self._http_session = None
        self._recent_hashes.clear()


# Global callback instance
_callback: Optional[HindsightCallback] = None


def get_callback() -> HindsightCallback:
    """Get the global callback instance, creating it if necessary."""
    global _callback
    if _callback is None:
        _callback = HindsightCallback()
    return _callback


def cleanup_callback() -> None:
    """Clean up the global callback instance."""
    global _callback
    if _callback is not None:
        _callback.close()
        _callback = None
