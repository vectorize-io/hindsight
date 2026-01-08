#!/usr/bin/env python3
"""
LLM Comparison Benchmark for Hindsight
Compares fact extraction and recall quality between different LLM providers.
"""

import asyncio
import httpx
import json
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

API_URL = "http://localhost:8888"

# Test memories - diverse content to test extraction quality
TEST_MEMORIES = [
    "The Camarilla trading system uses pivot points calculated from the previous day's high, low, and close prices. The key levels are H3, H4, L3, and L4 which serve as breakout points.",
    "User prefers to trade EUR/USD during the London session between 8:00 AM and 12:00 PM GMT. They avoid trading during news events.",
    "The backtesting showed that the H4 breakout strategy has a 62% win rate when combined with RSI confirmation above 70 for shorts and below 30 for longs.",
    "Python was used to build the trading bot. The main dependencies are pandas for data analysis, ccxt for exchange connectivity, and ta-lib for technical indicators.",
    "User experienced significant losses on March 15th, 2024 due to unexpected Fed announcement. Lesson learned: always check economic calendar before opening positions.",
    "The optimal position size is 2% of account balance per trade. Risk-reward ratio should be minimum 1:2 for all trades.",
    "User's trading journal shows best performance on Tuesdays and Wednesdays. Mondays tend to have more false breakouts.",
    "The mobile app integration uses React Native with Expo. Push notifications are sent via Firebase Cloud Messaging.",
    "Database schema includes tables for trades, signals, user_preferences, and historical_data. PostgreSQL is used with TimescaleDB extension for time-series data.",
    "Authentication is handled via JWT tokens with 24-hour expiration. Refresh tokens are stored securely and rotated on each use.",
]

# Test recall queries
TEST_QUERIES = [
    "What is the win rate of the H4 breakout strategy?",
    "When does the user prefer to trade EUR/USD?",
    "What happened on March 15th 2024?",
    "What programming language was used for the trading bot?",
    "What is the recommended position size?",
    "Which days show the best trading performance?",
    "How is authentication handled in the system?",
    "What are the Camarilla pivot levels used for breakouts?",
]


@dataclass
class BenchmarkResult:
    llm_provider: str
    llm_model: str
    bank_id: str
    retain_results: list = field(default_factory=list)
    recall_results: list = field(default_factory=list)
    total_retain_time: float = 0.0
    total_recall_time: float = 0.0
    retain_success_count: int = 0
    retain_failure_count: int = 0
    facts_extracted: int = 0
    entities_extracted: int = 0
    recall_avg_relevance: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


async def get_llm_config(client: httpx.AsyncClient) -> tuple[str, str]:
    """Get current LLM provider and model from a test call."""
    # We'll infer this from the container env
    return "unknown", "unknown"


async def create_bank(client: httpx.AsyncClient, bank_id: str) -> bool:
    """Create a new memory bank using PUT (upsert)."""
    try:
        resp = await client.put(
            f"{API_URL}/v1/default/banks/{bank_id}",
            json={
                "name": bank_id,
                "background": "Benchmark test bank for LLM comparison",
            },
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        print(f"Error creating bank: {e}")
        return False


async def delete_bank(client: httpx.AsyncClient, bank_id: str) -> bool:
    """Delete a memory bank."""
    try:
        resp = await client.delete(f"{API_URL}/v1/default/banks/{bank_id}")
        return resp.status_code in (200, 204, 404)
    except Exception as e:
        print(f"Error deleting bank: {e}")
        return False


async def retain_memory(
    client: httpx.AsyncClient, bank_id: str, content: str, timeout: float = 180.0
) -> dict:
    """Store a memory using synchronous mode (waits for completion)."""
    start = time.time()
    result = {"content": content[:50] + "...", "success": False, "time": 0, "error": None}

    try:
        # Submit retain request synchronously (async=false waits for completion)
        resp = await client.post(
            f"{API_URL}/v1/default/banks/{bank_id}/memories",
            json={"items": [{"content": content}], "async": False},
            timeout=timeout,
        )

        if resp.status_code in (200, 201):
            data = resp.json()
            result["success"] = data.get("success", True)
            result["facts"] = data.get("facts_count", 0)
            result["entities"] = data.get("entities_count", 0)
        else:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"

    except httpx.TimeoutException:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        result["error"] = str(e)[:200]

    result["time"] = time.time() - start
    return result


async def recall_memories(
    client: httpx.AsyncClient, bank_id: str, query: str, timeout: float = 60.0
) -> dict:
    """Query memories and measure response."""
    start = time.time()
    result = {"query": query, "success": False, "time": 0, "memories_count": 0, "error": None}

    try:
        resp = await client.post(
            f"{API_URL}/v1/default/banks/{bank_id}/memories/recall",
            json={"query": query, "limit": 5},
            timeout=timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            result["success"] = True
            result["memories_count"] = len(results)
            result["memories"] = [
                {
                    "fact": m.get("text", "")[:100],
                    "fact_type": m.get("type", ""),
                }
                for m in results[:3]
            ]
        else:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"

    except Exception as e:
        result["error"] = str(e)[:200]

    result["time"] = time.time() - start
    return result


async def get_bank_stats(client: httpx.AsyncClient, bank_id: str) -> dict:
    """Get bank statistics."""
    try:
        resp = await client.get(f"{API_URL}/v1/default/banks/{bank_id}/stats", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


async def run_benchmark(bank_id: str, llm_info: tuple[str, str]) -> BenchmarkResult:
    """Run the full benchmark suite."""
    result = BenchmarkResult(
        llm_provider=llm_info[0],
        llm_model=llm_info[1],
        bank_id=bank_id,
    )

    async with httpx.AsyncClient() as client:
        # Create fresh bank
        print(f"\n{'='*60}")
        print(f"Creating bank: {bank_id}")
        await delete_bank(client, bank_id)
        await asyncio.sleep(1)
        if not await create_bank(client, bank_id):
            print("Failed to create bank!")
            return result

        # Phase 1: Retain memories
        print(f"\n--- RETAIN PHASE ({len(TEST_MEMORIES)} memories) ---")
        for i, memory in enumerate(TEST_MEMORIES, 1):
            print(f"  [{i}/{len(TEST_MEMORIES)}] Storing: {memory[:40]}...")
            retain_result = await retain_memory(client, bank_id, memory)
            result.retain_results.append(retain_result)

            if retain_result["success"]:
                result.retain_success_count += 1
                result.facts_extracted += retain_result.get("facts", 0)
                result.entities_extracted += retain_result.get("entities", 0)
                print(f"         ✓ {retain_result['time']:.1f}s - {retain_result.get('facts', 0)} facts")
            else:
                result.retain_failure_count += 1
                print(f"         ✗ {retain_result['time']:.1f}s - {retain_result.get('error', 'Unknown')[:50]}")

            result.total_retain_time += retain_result["time"]

        # Get stats
        stats = await get_bank_stats(client, bank_id)
        print(f"\n  Bank stats: {stats.get('total_nodes', 0)} nodes, {stats.get('total_links', 0)} links")

        # Phase 2: Recall queries
        print(f"\n--- RECALL PHASE ({len(TEST_QUERIES)} queries) ---")
        relevance_scores = []
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"  [{i}/{len(TEST_QUERIES)}] Query: {query[:40]}...")
            recall_result = await recall_memories(client, bank_id, query)
            result.recall_results.append(recall_result)

            if recall_result["success"]:
                count = recall_result["memories_count"]
                if count > 0:
                    relevance_scores.append(count)
                print(f"         ✓ {recall_result['time']:.1f}s - {count} results")
            else:
                print(f"         ✗ {recall_result['time']:.1f}s - {recall_result.get('error', 'Unknown')[:50]}")

            result.total_recall_time += recall_result["time"]

        if relevance_scores:
            result.recall_avg_relevance = sum(relevance_scores) / len(relevance_scores)

    return result


def print_summary(result: BenchmarkResult):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY: {result.llm_provider} / {result.llm_model}")
    print(f"{'='*60}")
    print(f"\nRETAIN:")
    print(f"  Success rate: {result.retain_success_count}/{len(result.retain_results)} ({100*result.retain_success_count/max(1, len(result.retain_results)):.0f}%)")
    print(f"  Total time: {result.total_retain_time:.1f}s")
    print(f"  Avg time per memory: {result.total_retain_time/max(1, len(result.retain_results)):.1f}s")
    print(f"  Facts extracted: {result.facts_extracted}")
    print(f"  Entities extracted: {result.entities_extracted}")

    print(f"\nRECALL:")
    successful_recalls = sum(1 for r in result.recall_results if r["success"])
    print(f"  Success rate: {successful_recalls}/{len(result.recall_results)} ({100*successful_recalls/max(1, len(result.recall_results)):.0f}%)")
    print(f"  Total time: {result.total_recall_time:.1f}s")
    print(f"  Avg time per query: {result.total_recall_time/max(1, len(result.recall_results)):.1f}s")
    print(f"  Avg results per query: {result.recall_avg_relevance:.1f}")


async def main():
    parser = argparse.ArgumentParser(description="LLM Comparison Benchmark")
    parser.add_argument("--bank-prefix", default="benchmark", help="Bank ID prefix")
    parser.add_argument("--provider", default="", help="LLM provider name for labeling")
    parser.add_argument("--model", default="", help="LLM model name for labeling")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    # Get LLM info
    llm_info = (args.provider or "unknown", args.model or "unknown")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bank_id = f"{args.bank_prefix}-{timestamp}"

    print(f"Starting LLM Comparison Benchmark")
    print(f"Provider: {llm_info[0]}")
    print(f"Model: {llm_info[1]}")
    print(f"Bank: {bank_id}")
    print(f"Memories: {len(TEST_MEMORIES)}")
    print(f"Queries: {len(TEST_QUERIES)}")

    result = await run_benchmark(bank_id, llm_info)
    print_summary(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "provider": result.llm_provider,
                "model": result.llm_model,
                "bank_id": result.bank_id,
                "timestamp": result.timestamp,
                "retain": {
                    "success_rate": result.retain_success_count / max(1, len(result.retain_results)),
                    "total_time": result.total_retain_time,
                    "avg_time": result.total_retain_time / max(1, len(result.retain_results)),
                    "facts_extracted": result.facts_extracted,
                    "entities_extracted": result.entities_extracted,
                    "failures": result.retain_failure_count,
                },
                "recall": {
                    "success_rate": sum(1 for r in result.recall_results if r["success"]) / max(1, len(result.recall_results)),
                    "total_time": result.total_recall_time,
                    "avg_time": result.total_recall_time / max(1, len(result.recall_results)),
                    "avg_relevance": result.recall_avg_relevance,
                },
                "details": {
                    "retain_results": result.retain_results,
                    "recall_results": result.recall_results,
                }
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
