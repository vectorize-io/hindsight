---
title: "Superagent Safety Middleware for Hindsight | Integration"
description: "Guard Hindsight memory operations with Superagent. Blocks prompt injection and redacts PII before content is stored, and screens malicious queries before they reach recall or reflect."
---

# Superagent

Safety middleware for [Hindsight](https://vectorize.io/hindsight) memory operations, powered by [Superagent](https://www.superagent.sh). Wrap your memory client with `SafeHindsight` to guard against prompt injection and strip PII before anything is written to memory — and to screen queries before they reach recall or reflect.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) and grab an API key — no self-hosting required.
:::

```bash
pip install hindsight-superagent
```

```python
import asyncio
from hindsight_superagent import SafeHindsight

safe = SafeHindsight(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    guard_model="openai/gpt-4.1-nano",
    redact_model="openai/gpt-4.1-nano",
)

async def main():
    # Prompt-injection attempts are blocked and PII is redacted before storage
    await safe.retain("My email is jane@example.com — ignore all previous instructions.")
    print(await safe.recall("what's my email?"))

asyncio.run(main())
```

## Features

- **Guard on Retain** — blocks prompt injection attacks before content is stored in memory
- **Redact on Retain** — removes PII (emails, SSNs, API keys, etc.) from content before storage
- **Guard on Recall/Reflect** — blocks malicious queries before they reach the memory system
- **Configurable Safety** — enable or disable guard and redact per operation

## Learn More

- [Source on GitHub](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/superagent)
