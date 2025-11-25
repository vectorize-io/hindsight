"""
Hindsight - All-in-one semantic memory system for AI agents.

This package provides a simple way to run Hindsight locally with embedded PostgreSQL.

Example:
    ```python
    from hindsight import start_server, HindsightClient

    # Start server with embedded PostgreSQL (pg0)
    server = start_server(
        llm_provider="groq",
        llm_api_key="your-api-key",
        llm_model="llama-3.3-70b-versatile"
    )

    # Create client
    client = HindsightClient(base_url=server.url)

    # Store memories
    client.put(agent_id="assistant", content="User prefers Python for data analysis")

    # Search memories
    results = client.search(agent_id="assistant", query="programming preferences")

    # Generate contextual response
    response = client.think(agent_id="assistant", query="What languages should I recommend?")

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

from .server import Server as HindsightServer, start_server

# Re-export Client from hindsight-client
from hindsight_client import Hindsight as HindsightClient

__all__ = [
    "HindsightServer",
    "start_server",
    "HindsightClient",
]
