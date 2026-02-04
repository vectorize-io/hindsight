"""
Hindsight - All-in-one semantic memory system for AI agents.

This package provides a simple way to run Hindsight locally with embedded PostgreSQL.

Easiest way - Embedded client (recommended):
    ```python
    from hindsight import HindsightEmbedded

    # Server starts automatically on first use
    client = HindsightEmbedded(
        profile="myapp",
        llm_provider="groq",
        llm_api_key="your-api-key",
    )

    # Use immediately - no manual server management needed
    client.retain(bank_id="alice", content="Alice loves AI")
    results = client.recall(bank_id="alice", query="What does Alice like?")
    ```

Manual server management:
    ```python
    from hindsight import start_server, HindsightClient

    # Start server with embedded PostgreSQL (pg0)
    server = start_server(
        llm_provider="groq",
        llm_api_key="your-api-key",
        llm_model="openai/gpt-oss-120b"
    )

    # Create client
    client = HindsightClient(base_url=server.url)

    # Store memories
    client.retain(bank_id="assistant", content="User prefers Python for data analysis")

    # Search memories
    results = client.recall(bank_id="assistant", query="programming preferences")

    # Generate contextual response
    response = client.reflect(bank_id="assistant", query="What are my interests?")

    # Stop server when done
    server.stop()
    ```

Using context manager:
    ```python
    from hindsight import HindsightServer, HindsightClient

    with HindsightServer(llm_provider="groq", llm_api_key="...") as server:
        client = HindsightClient(base_url=server.url)
        # ... use client ...
    # Server automatically stops
    ```
"""

from .client_wrapper import HindsightClient
from .embedded import HindsightEmbedded
from .server import Server as HindsightServer, start_server

__all__ = [
    "HindsightServer",
    "start_server",
    "HindsightClient",
    "HindsightEmbedded",
]
