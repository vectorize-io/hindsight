"""Manual integration test — requires running Hindsight API on localhost:8888."""

import asyncio

from hindsight_client import Hindsight
from hindsight_langgraph import create_hindsight_tools


async def main():
    client = Hindsight(base_url="http://localhost:8888")
    try:
        # Create a test bank
        await client.acreate_bank("langgraph-test", name="LangGraph Test")

        tools = create_hindsight_tools(client=client, bank_id="langgraph-test")
        retain, recall, reflect = tools

        # Test retain
        print("--- Retain ---")
        result = await retain.ainvoke("The user's favorite language is Python")
        print(result)

        result = await retain.ainvoke("The user lives in San Francisco")
        print(result)

        # Give the engine a moment to process
        await asyncio.sleep(2)

        # Test recall
        print("\n--- Recall ---")
        result = await recall.ainvoke("What programming language does the user like?")
        print(result)

        # Test reflect
        print("\n--- Reflect ---")
        result = await reflect.ainvoke("What do you know about the user?")
        print(result)

        # Cleanup
        await client.adelete_bank("langgraph-test")
        print("\n--- Done, bank cleaned up ---")
    finally:
        # Close the client so aiohttp doesn't warn about unclosed sessions.
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
