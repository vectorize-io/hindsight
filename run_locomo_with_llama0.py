#!/usr/bin/env python3
"""
Run LoComo benchmark with llama0 (gemma-3-1b).
"""

import os
import sys
import asyncio
import subprocess
import time
from pathlib import Path

# Add llama0 to path
llama0_path = Path("/Users/nicoloboschi/dev/locallm/llama0/src")
sys.path.insert(0, str(llama0_path.parent))

from llama0 import Llama0

async def main():
    print("=" * 70)
    print("Running LoComo Benchmark with llama0 (gemma-3-1b)")
    print("=" * 70)

    # Start llama0 server
    print("\n📡 Starting llama0 server with gemma-3-1b...")
    llm = Llama0(
        "gemma-3-1b",
        port=9400,
        n_ctx=131072,  # 128K context
        n_parallel=8,
        verbose=False
    )
    llm.start(timeout=90)
    print(f"✓ Server ready at {llm.url}\n")

    try:
        # Set environment variables for hindsight to use llama0
        env = os.environ.copy()
        env["HINDSIGHT_API_ANSWER_LLM_PROVIDER"] = "ollama"  # OpenAI-compatible
        env["HINDSIGHT_API_ANSWER_LLM_BASE_URL"] = f"{llm.url}/v1"
        env["HINDSIGHT_API_ANSWER_LLM_MODEL"] = "gemma-3-1b"
        env["HINDSIGHT_API_ANSWER_LLM_API_KEY"] = "not-needed"

        # Also set max concurrent to 1 for local LLM
        env["HINDSIGHT_API_LLM_MAX_CONCURRENT"] = "1"

        print("🧪 Running LoComo benchmark (1 conversation)...")
        print(f"   Answer LLM: llama0 @ {llm.url}/v1")
        print(f"   Model: gemma-3-1b")
        print()

        # Run the benchmark
        benchmark_dir = Path("/Users/nicoloboschi/dev/hindsight-wt4/hindsight-dev/benchmarks/locomo")

        proc = subprocess.run(
            [
                sys.executable,
                "locomo_benchmark.py",
                "--max-conversations", "1",
            ],
            cwd=benchmark_dir,
            env=env,
            capture_output=False
        )

        if proc.returncode == 0:
            print("\n✓ Benchmark completed successfully!")
        else:
            print(f"\n✗ Benchmark failed with exit code {proc.returncode}")

    finally:
        print("\n🛑 Stopping llama0 server...")
        llm.stop()
        print("✓ Done")

if __name__ == "__main__":
    asyncio.run(main())
