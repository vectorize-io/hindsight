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

from .config import (
    get_config,
    get_defaults,
    is_configured,
    HindsightConfig,
    HindsightDefaults,
    MemoryInjectionMode,
)

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


class HindsightError(Exception):
    """Exception raised when a Hindsight operation fails.

    This is raised when inject_memories=True and recall fails,
    or when store_conversations=True and store fails.
    """
    pass


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

    def _get_effective_settings(self, kwargs: Dict[str, Any]) -> HindsightDefaults:
        """Get effective per-call settings from kwargs with fallback to defaults.

        Per-call kwargs (hindsight_*) override defaults. Supported kwargs:
        - hindsight_bank_id: Override bank_id
        - hindsight_document_id: Override document_id
        - hindsight_budget: Override budget
        - hindsight_fact_types: Override fact_types
        - hindsight_max_memories: Override max_memories
        - hindsight_max_memory_tokens: Override max_memory_tokens
        - hindsight_use_reflect: Override use_reflect
        - hindsight_reflect_include_facts: Override reflect_include_facts
        - hindsight_context: Override reflect_context
        - hindsight_response_schema: Override reflect_response_schema
        - hindsight_include_entities: Override include_entities
        - hindsight_trace: Override trace

        Note: hindsight_query is handled separately in log_pre_api_call since it's
        always per-call (no sensible default for dynamic queries).
        """
        defaults = get_defaults() or HindsightDefaults()

        return HindsightDefaults(
            bank_id=kwargs.get("hindsight_bank_id", defaults.bank_id),
            document_id=kwargs.get("hindsight_document_id", defaults.document_id),
            budget=kwargs.get("hindsight_budget", defaults.budget),
            fact_types=kwargs.get("hindsight_fact_types", defaults.fact_types),
            max_memories=kwargs.get("hindsight_max_memories", defaults.max_memories),
            max_memory_tokens=kwargs.get("hindsight_max_memory_tokens", defaults.max_memory_tokens),
            use_reflect=kwargs.get("hindsight_use_reflect", defaults.use_reflect),
            reflect_include_facts=kwargs.get("hindsight_reflect_include_facts", defaults.reflect_include_facts),
            reflect_context=kwargs.get("hindsight_context", defaults.reflect_context),
            reflect_response_schema=kwargs.get("hindsight_response_schema", defaults.reflect_response_schema),
            include_entities=kwargs.get("hindsight_include_entities", defaults.include_entities),
            trace=kwargs.get("hindsight_trace", defaults.trace),
        )

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

    def _http_post(self, url: str, json_data: dict, config: HindsightConfig) -> dict:
        """Make a synchronous HTTP POST request.

        Raises:
            HindsightError: If the request fails for any reason.
        """
        session = self._get_http_session()
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        try:
            if HAS_REQUESTS:
                response = session.post(url, json=json_data, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            elif HAS_HTTPX:
                response = session.post(url, json=json_data, headers=headers)
                response.raise_for_status()
                return response.json()
            else:
                raise HindsightError("No HTTP client available (install requests or httpx)")
        except HindsightError:
            raise
        except Exception as e:
            if config.verbose:
                logger.error(f"HTTP POST failed: {e}")
            raise HindsightError(f"Hindsight API request failed: {e}") from e

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

    def _messages_to_query(self, messages: List[Dict[str, Any]]) -> str:
        """Concatenate all message contents into a single query string."""
        message_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                message_parts.append(content)
            elif isinstance(content, list):
                # Handle structured content (e.g., vision messages)
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        message_parts.append(item.get("text", ""))
        return "\n".join(message_parts)

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
        settings: HindsightDefaults,
        config: HindsightConfig
    ) -> str:
        """Format memory recall results into a context string.

        Results can be RecallResult objects (with .text, .type attributes)
        or dicts (with get() method).
        """
        if not results:
            return ""

        # Apply limit if set, otherwise use all results
        results_to_use = results[:settings.max_memories] if settings.max_memories else results
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

    def _recall_memories_sync(
        self,
        query: str,
        settings: HindsightDefaults,
        config: HindsightConfig
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories from Hindsight (sync) using direct HTTP.

        Raises:
            HindsightError: If inject_memories=True and recall fails.
        """
        bank_id = settings.bank_id
        if not bank_id:
            raise HindsightError(
                "No bank_id configured. Call set_defaults(bank_id=...) "
                "or pass hindsight_bank_id=... to the completion call."
            )

        url = f"{config.hindsight_api_url}/v1/default/banks/{bank_id}/memories/recall"

        request_data = {
            "query": query,
            "budget": settings.budget or "mid",
            "max_tokens": settings.max_memory_tokens or 4096,
        }
        if settings.fact_types:
            request_data["types"] = settings.fact_types

        # Add trace parameter for debugging
        if settings.trace:
            request_data["trace"] = True

        # Add include options for entity observations
        # include_entities=True -> include: {entities: {}}
        # include_entities=False -> include: {entities: null}
        if settings.include_entities:
            request_data["include"] = {"entities": {}}
        else:
            request_data["include"] = {"entities": None}

        try:
            response = self._http_post(url, request_data, config)
            if response and "results" in response:
                return response["results"]
            return []
        except HindsightError as e:
            if config.verbose:
                logger.error(f"Failed to recall memories: {e}")
            raise HindsightError(f"Memory recall failed: {e}") from e

    async def _recall_memories_async(
        self,
        query: str,
        settings: HindsightDefaults,
        config: HindsightConfig
    ) -> List[Any]:
        """Recall relevant memories from Hindsight (async).

        Uses thread pool executor with sync HTTP to avoid event loop conflicts.

        Raises:
            HindsightError: If inject_memories=True and recall fails.
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            _executor,
            lambda: self._recall_memories_sync(query, settings, config)
        )

        return results if isinstance(results, list) else []

    def _reflect_sync(
        self,
        query: str,
        settings: HindsightDefaults,
        config: HindsightConfig
    ) -> Optional[str]:
        """Generate a reflection response from Hindsight (sync) using direct HTTP.

        Returns:
            The reflect response text, or None if no response.

        Raises:
            HindsightError: If inject_memories=True and reflect fails.
        """
        bank_id = settings.bank_id
        if not bank_id:
            raise HindsightError(
                "No bank_id configured. Call set_defaults(bank_id=...) "
                "or pass hindsight_bank_id=... to the completion call."
            )

        url = f"{config.hindsight_api_url}/v1/default/banks/{bank_id}/reflect"

        request_data: Dict[str, Any] = {
            "query": query,
            "budget": settings.budget or "mid",
            "max_tokens": settings.max_memory_tokens or 4096,
        }

        # Add context if provided (shapes reasoning but not retrieval)
        if settings.reflect_context:
            request_data["context"] = settings.reflect_context

        # Add response_schema for structured output
        if settings.reflect_response_schema:
            request_data["response_schema"] = settings.reflect_response_schema

        # Add include options for facts if requested
        if settings.reflect_include_facts:
            request_data["include"] = {"facts": {}}

        try:
            response = self._http_post(url, request_data, config)
            if response:
                # Handle structured output if schema was provided
                if settings.reflect_response_schema and "structured_output" in response:
                    # Return structured output as JSON string for injection
                    import json
                    return json.dumps(response["structured_output"], indent=2)
                # Otherwise return text response
                return response.get("text", "")
            return None
        except HindsightError as e:
            if config.verbose:
                logger.error(f"Failed to reflect: {e}")
            raise HindsightError(f"Reflect failed: {e}") from e

    async def _reflect_async(
        self,
        query: str,
        settings: HindsightDefaults,
        config: HindsightConfig
    ) -> Optional[str]:
        """Generate a reflection response from Hindsight (async).

        Uses thread pool executor with sync HTTP to avoid event loop conflicts.

        Returns:
            The reflect response text, or None if no response.

        Raises:
            HindsightError: If inject_memories=True and reflect fails.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: self._reflect_sync(query, settings, config)
        )
        return result

    def _store_conversation_sync(
        self,
        messages: List[Dict[str, Any]],
        response: ModelResponse,
        model: str,
        settings: HindsightDefaults,
        config: HindsightConfig,
    ) -> None:
        """Store the conversation to Hindsight (sync) using direct HTTP.

        IMPORTANT: This intentionally sends the FULL conversation history each call,
        not just the new messages. This is required because Hindsight's retain API
        with document_id performs an UPSERT (replace), not an append.

        If we only sent deltas (new messages), Hindsight would only have the latest
        fragment and lose all prior context. By sending the full conversation each
        time, Hindsight always has the complete context to extract meaningful facts.

        Example with delta-only (WRONG):
            Call 1: "USER: deliver to Alex\\nASSISTANT_TOOL_CALLS: look_at_business"
            Call 2: "TOOL_RESULT: TechStart Labs\\nASSISTANT_TOOL_CALLS: go_up"  # Lost context!

        Example with full conversation (CORRECT):
            Call 1: "USER: deliver to Alex\\nASSISTANT_TOOL_CALLS: look_at_business"
            Call 2: "USER: deliver to Alex\\nASSISTANT_TOOL_CALLS: look_at_business\\n
                     TOOL_RESULT: TechStart Labs\\nASSISTANT_TOOL_CALLS: go_up"  # Full context!

        Each upsert replaces the previous, so the final stored document contains
        the complete conversation for Hindsight to process.

        Raises:
            HindsightError: If store_conversations=True and store fails.
        """
        bank_id = settings.bank_id
        if not bank_id:
            raise HindsightError(
                "No bank_id configured. Call set_defaults(bank_id=...) "
                "or pass hindsight_bank_id=... to the completion call."
            )

        # Extract assistant response from the LLM response
        assistant_output = ""
        assistant_tool_calls = []
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                assistant_output = choice.message.content or ""
                # Also capture tool calls
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        if hasattr(tc, "function"):
                            assistant_tool_calls.append(f"{tc.function.name}({tc.function.arguments})")

        # Skip if no content AND no tool calls - nothing to store
        if not assistant_output and not assistant_tool_calls:
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

            # Handle tool messages (results from tool calls)
            if role == "TOOL":
                tool_call_id = msg.get("tool_call_id", "")
                items.append(f"TOOL_RESULT: {content}")
                continue

            # Handle assistant messages with tool calls
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_strs = []
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        tc_strs.append(f"{tc.function.name}({tc.function.arguments})")
                    elif isinstance(tc, dict) and "function" in tc:
                        func = tc["function"]
                        tc_strs.append(f"{func.get('name', '')}({func.get('arguments', '')})")
                if tc_strs:
                    items.append(f"ASSISTANT_TOOL_CALLS: {'; '.join(tc_strs)}")
                if content:
                    items.append(f"ASSISTANT: {content}")
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

        # Add the new assistant response (text or tool calls)
        if assistant_output:
            items.append(f"ASSISTANT: {assistant_output}")
        if assistant_tool_calls:
            items.append(f"ASSISTANT_TOOL_CALLS: {'; '.join(assistant_tool_calls)}")

        if not items:
            return

        # Use last user message for deduplication hash
        user_input = self._extract_user_query(messages) or ""

        # Deduplication check - include tool calls if no text content
        dedup_output = assistant_output or ";".join(assistant_tool_calls)
        conv_hash = self._compute_conversation_hash(user_input, dedup_output)
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

        url = f"{config.hindsight_api_url}/v1/default/banks/{bank_id}/memories"

        request_data = {
            "items": [
                {
                    "content": conversation_text,
                    "context": f"conversation:litellm:{model}",
                    "metadata": metadata,
                    "document_id": settings.document_id,  # Group by document
                }
            ],
        }

        try:
            self._http_post(url, request_data, config)
            if config.verbose:
                logger.info(f"Stored conversation to Hindsight bank: {bank_id}")
        except HindsightError as e:
            if config.verbose:
                logger.error(f"Failed to store conversation: {e}")
            raise HindsightError(f"Memory storage failed: {e}") from e

    async def _store_conversation_async(
        self,
        messages: List[Dict[str, Any]],
        response: ModelResponse,
        model: str,
        settings: HindsightDefaults,
        config: HindsightConfig,
    ) -> None:
        """Store the conversation to Hindsight (async).

        Uses thread pool executor with sync HTTP to avoid event loop conflicts.

        Raises:
            HindsightError: If store_conversations=True and store fails.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _executor,
            lambda: self._store_conversation_sync(
                messages, response, model, settings, config
            )
        )

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
        config = get_config()
        if not config or not config.inject_memories:
            return

        # Get effective settings (kwargs override defaults)
        settings = self._get_effective_settings(kwargs)
        if not settings.bank_id:
            raise ValueError(
                "No bank_id configured. Either call set_defaults(bank_id=...) "
                "or pass hindsight_bank_id=... to the completion call."
            )

        if self._should_skip_model(model, config):
            return

        # hindsight_query is required when inject_memories=True
        custom_query = kwargs.get("hindsight_query")
        if not custom_query:
            raise ValueError(
                "hindsight_query is required when inject_memories=True. "
                "Pass hindsight_query='your query' to specify what to search for in memory. "
                "Example: hindsight_query=recipient_name or hindsight_query='What do I know about Alice?'"
            )

        user_query = custom_query

        # Use reflect or recall based on settings
        if settings.use_reflect:
            # Use reflect API for disposition-aware reasoning
            reflect_response = self._reflect_sync(user_query, settings, config)
            if not reflect_response:
                return

            # Format reflect response as context
            memory_context = (
                "# Relevant Context from Memory\n"
                f"{reflect_response}"
            )
        else:
            # Use recall API for raw fact retrieval
            memories = self._recall_memories_sync(user_query, settings, config)
            if not memories:
                return

            # Format and inject memories
            memory_context = self._format_memories(memories, settings, config)

        updated_messages = self._inject_memories_into_messages(
            messages, memory_context, config
        )

        # Modify messages list IN-PLACE (don't just reassign kwargs)
        messages.clear()
        messages.extend(updated_messages)

        if config.verbose:
            mode = "reflect" if settings.use_reflect else "recall"
            logger.info(f"Injected memory context via {mode}")

    async def async_log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """Called before making the API call (async).

        This is where we inject memories into the messages.
        """
        config = get_config()
        if not config or not config.inject_memories:
            return

        # Get effective settings (kwargs override defaults)
        settings = self._get_effective_settings(kwargs)
        if not settings.bank_id:
            raise ValueError(
                "No bank_id configured. Either call set_defaults(bank_id=...) "
                "or pass hindsight_bank_id=... to the completion call."
            )

        if self._should_skip_model(model, config):
            return

        # hindsight_query is required when inject_memories=True
        custom_query = kwargs.get("hindsight_query")
        if not custom_query:
            raise ValueError(
                "hindsight_query is required when inject_memories=True. "
                "Pass hindsight_query='your query' to specify what to search for in memory. "
                "Example: hindsight_query=recipient_name or hindsight_query='What do I know about Alice?'"
            )

        user_query = custom_query

        # Use reflect or recall based on settings
        if settings.use_reflect:
            # Use reflect API for disposition-aware reasoning
            reflect_response = await self._reflect_async(user_query, settings, config)
            if not reflect_response:
                return

            # Format reflect response as context
            memory_context = (
                "# Relevant Context from Memory\n"
                f"{reflect_response}"
            )
        else:
            # Use recall API for raw fact retrieval
            memories = await self._recall_memories_async(user_query, settings, config)
            if not memories:
                return

            # Format and inject memories
            memory_context = self._format_memories(memories, settings, config)

        updated_messages = self._inject_memories_into_messages(
            messages, memory_context, config
        )

        # Modify messages list IN-PLACE (don't just reassign kwargs)
        messages.clear()
        messages.extend(updated_messages)

        if config.verbose:
            mode = "reflect" if settings.use_reflect else "recall"
            logger.info(f"Injected memory context via {mode}")

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
        config = get_config()
        if not config or not config.store_conversations:
            return

        # Get effective settings (kwargs override defaults)
        settings = self._get_effective_settings(kwargs)
        if not settings.bank_id:
            # bank_id validation already done in log_pre_api_call
            return

        model = kwargs.get("model", "unknown")
        if self._should_skip_model(model, config):
            return

        messages = kwargs.get("messages", [])
        if not messages:
            return

        # Store the conversation
        self._store_conversation_sync(messages, response_obj, model, settings, config)

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
        config = get_config()
        if not config or not config.store_conversations:
            return

        # Get effective settings (kwargs override defaults)
        settings = self._get_effective_settings(kwargs)
        if not settings.bank_id:
            # bank_id validation already done in async_log_pre_api_call
            return

        model = kwargs.get("model", "unknown")
        if self._should_skip_model(model, config):
            return

        messages = kwargs.get("messages", [])
        if not messages:
            return

        # Store the conversation
        await self._store_conversation_async(messages, response_obj, model, settings, config)

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
