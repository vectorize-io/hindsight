#!/usr/bin/env python3
"""Start llama0 server for hindsight benchmarks"""

from llama0 import Llama0
import time
import signal
import sys

print("Starting llama0 server for hindsight...")
print(f"Model: qwen2.5-3b-instruct (auto-download if needed)")
print(f"Port: 9400")

llm = Llama0(
    "qwen2.5-3b-instruct",
    port=9400,
    verbose=True,
    n_gpu_layers=99,    # Use all GPU layers
    n_ctx=32768,        # 32K context (enough for hindsight chunks)
    n_parallel=4        # 4 parallel requests (balanced)
)

def signal_handler(sig, frame):
    print("\nShutting down llama0 server...")
    llm.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

llm.start()
print(f"\n✓ llama0 server running at {llm.url}")
print("Press Ctrl+C to stop\n")

# Keep running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
    llm.stop()
