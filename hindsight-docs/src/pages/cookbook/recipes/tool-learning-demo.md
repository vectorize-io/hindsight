---
sidebar_position: 5
---

# Routing Tool Learning


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/05-tool-learning-demo.ipynb)
:::


This notebook demonstrates how Hindsight helps an LLM learn which tool to use when tool names are ambiguous. Without memory, the LLM might randomly select between similarly-named tools. With Hindsight, it learns from past interactions and consistently makes the correct choice.

## The Scenario

We have a task routing system with two tools:
- `route_to_channel_alpha` - Routes to processing channel Alpha
- `route_to_channel_omega` - Routes to processing channel Omega

The tool names and descriptions are **intentionally vague**. In reality:
- Channel Alpha handles **FINANCIAL/PAYMENT** tasks (refunds, billing, etc.)
- Channel Omega handles **TECHNICAL/SUPPORT** tasks (bugs, features, etc.)

**Without Hindsight:** The LLM guesses randomly based on vague descriptions  
**With Hindsight:** The LLM learns from feedback which channel handles what

## Prerequisites

Make sure you have Hindsight running:

```bash
export OPENAI_API_KEY=your-key

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=o3-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

## Installation


```python
!pip install hindsight-litellm hindsight-client litellm nest_asyncio python-dotenv -U -q
```

## Setup


```python
import os
import json
import uuid
import time
import logging
import nest_asyncio
from typing import Optional
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import litellm
import hindsight_litellm
from hindsight_client import Hindsight

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set")
```

## Define Tools

These tool definitions are **intentionally ambiguous** - the descriptions don't reveal which channel handles what type of request.


```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "route_to_channel_alpha",
            "description": "Routes the customer request to processing channel Alpha. Use this channel for appropriate request types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_summary": {
                        "type": "string",
                        "description": "A brief summary of the customer's request"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Priority level of the request"
                    }
                },
                "required": ["request_summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_channel_omega",
            "description": "Routes the customer request to processing channel Omega. Use this channel for appropriate request types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_summary": {
                        "type": "string",
                        "description": "A brief summary of the customer's request"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Priority level of the request"
                    }
                },
                "required": ["request_summary"]
            }
        }
    }
]
```

## Test Scenarios

A mix of financial and technical requests to test routing accuracy.


```python
TEST_SCENARIOS = [
    {
        "type": "financial",
        "request": "I was charged twice for my subscription last month. I need a refund for the duplicate charge.",
        "correct_tool": "route_to_channel_alpha"
    },
    {
        "type": "technical",
        "request": "The app keeps crashing when I try to upload a file larger than 10MB. This bug is blocking my work.",
        "correct_tool": "route_to_channel_omega"
    },
    {
        "type": "financial",
        "request": "My invoice shows an incorrect amount. The billing department needs to fix this.",
        "correct_tool": "route_to_channel_alpha"
    },
    {
        "type": "technical",
        "request": "I'd like to request a new feature: the ability to export reports as PDF.",
        "correct_tool": "route_to_channel_omega"
    },
    {
        "type": "financial",
        "request": "I need to update my payment method and understand why my last payment failed.",
        "correct_tool": "route_to_channel_alpha"
    },
]
```

## Helper Functions


```python
SYSTEM_PROMPT = """You are a customer service routing agent. Your job is to route customer requests to the appropriate processing channel.

You have access to two routing channels:
- route_to_channel_alpha: Routes to channel Alpha
- route_to_channel_omega: Routes to channel Omega

Analyze the customer's request and route it to the most appropriate channel. You must call one of the routing functions to process the request.

Important: Base your routing decision on what you know about each channel's purpose. If you have learned from previous interactions which channel handles specific types of requests, use that knowledge."""


def make_routing_request(user_request: str, use_hindsight: bool, bank_id: Optional[str] = None):
    """Make a routing request and return the tool called."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer Request: {user_request}"}
    ]

    if use_hindsight and bank_id:
        response = hindsight_litellm.completion(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            temperature=0.0,
        )
    else:
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            temperature=0.7,
        )

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return tool_call.function.name
    return None


def store_feedback(bank_id: str, request: str, correct_tool: str, request_type: str):
    """Store feedback about which tool was correct for a request type."""
    client = Hindsight(base_url=HINDSIGHT_API_URL, timeout=60.0)

    feedback_content = f"""ROUTING FEEDBACK:
Request type: {request_type}
Customer request: "{request}"
Correct routing: {correct_tool}

LEARNED RULE: {request_type.upper()} requests (like refunds, billing, payments, charges, invoices) should ALWAYS be routed to {correct_tool}.
This is important institutional knowledge for routing decisions."""

    client.retain(
        bank_id=bank_id,
        content=feedback_content,
        context=f"routing:feedback:{request_type}",
        metadata={"request_type": request_type, "correct_tool": correct_tool}
    )
```

## Phase 1: Without Hindsight (No Memory)

The LLM has no prior knowledge about which channel handles what. With ambiguous tool descriptions, it may route incorrectly.


```python
print("=" * 60)
print("PHASE 1: WITHOUT HINDSIGHT (No Memory)")
print("=" * 60)

phase1_results = []
for i, scenario in enumerate(TEST_SCENARIOS[:3], 1):
    print(f"\n--- Test {i}: {scenario['type'].upper()} Request ---")
    print(f"Request: \"{scenario['request'][:60]}...\"")

    tool_name = make_routing_request(scenario['request'], use_hindsight=False)

    is_correct = tool_name == scenario['correct_tool']
    phase1_results.append(is_correct)

    print(f"LLM chose: {tool_name}")
    print(f"Correct tool: {scenario['correct_tool']}")
    print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

phase1_accuracy = sum(phase1_results) / len(phase1_results) * 100
print(f"\n>>> Phase 1 Accuracy: {phase1_accuracy:.0f}% ({sum(phase1_results)}/{len(phase1_results)})")
```

## Phase 2: Teaching Phase

Now we provide feedback about correct routing to build memory. This simulates a human supervisor correcting the AI's routing decisions.


```python
bank_id = f"tool-learning-{uuid.uuid4().hex[:8]}"
print(f"Using bank_id: {bank_id}")

# Configure and enable Hindsight
hindsight_litellm.configure(
    hindsight_api_url=HINDSIGHT_API_URL,
    bank_id=bank_id,
    store_conversations=True,
    inject_memories=True,
    max_memories=10,
    recall_budget="high",
    verbose=False,
)
hindsight_litellm.enable()

print("\nStoring routing feedback...")

feedback_examples = [
    ("I need a refund for an incorrect charge on my account.", "route_to_channel_alpha", "financial"),
    ("There's a bug in the system causing data loss.", "route_to_channel_omega", "technical"),
    ("My billing statement has errors that need correction.", "route_to_channel_alpha", "financial"),
    ("I want to request a new feature for the dashboard.", "route_to_channel_omega", "technical"),
]

for request, correct_tool, req_type in feedback_examples:
    print(f"  Storing: {req_type.upper()} → {correct_tool}")
    store_feedback(bank_id, request, correct_tool, req_type)

print("\nWaiting 15 seconds for Hindsight to process memories...")
time.sleep(15)
print("Done!")
```

## Phase 3: With Hindsight (Memory-Augmented)

The LLM now has access to learned routing knowledge via Hindsight. It should route requests correctly based on past feedback.


```python
print("=" * 60)
print("PHASE 3: WITH HINDSIGHT (Memory-Augmented)")
print("=" * 60)

phase3_results = []
for i, scenario in enumerate(TEST_SCENARIOS, 1):
    print(f"\n--- Test {i}: {scenario['type'].upper()} Request ---")
    print(f"Request: \"{scenario['request'][:60]}...\"")

    tool_name = make_routing_request(
        scenario['request'],
        use_hindsight=True,
        bank_id=bank_id
    )

    is_correct = tool_name == scenario['correct_tool']
    phase3_results.append(is_correct)

    print(f"LLM chose: {tool_name}")
    print(f"Correct tool: {scenario['correct_tool']}")
    print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

phase3_accuracy = sum(phase3_results) / len(phase3_results) * 100
print(f"\n>>> Phase 3 Accuracy: {phase3_accuracy:.0f}% ({sum(phase3_results)}/{len(phase3_results)})")
```

## Summary


```python
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nPhase 1 (No Memory):      {phase1_accuracy:.0f}% accuracy")
print(f"Phase 3 (With Hindsight): {phase3_accuracy:.0f}% accuracy")

improvement = phase3_accuracy - phase1_accuracy
if improvement > 0:
    print(f"\n🎉 Improvement: +{improvement:.0f}% accuracy with Hindsight!")
elif improvement == 0:
    print(f"\nNote: Results may vary. Run again to see learning effect.")
else:
    print(f"\nNote: Phase 1 got lucky! Run again to see typical behavior.")

print(f"\nMemories stored in bank: {bank_id}")
print(f"View in UI: http://localhost:9999/banks/{bank_id}")

print("\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("Hindsight allows the LLM to learn from experience which tool")
print("to use, even when tool names/descriptions are ambiguous.")
```

## Cleanup


```python
hindsight_litellm.cleanup()

# Optional: delete the bank
import requests
response = requests.delete(f"{HINDSIGHT_API_URL}/v1/default/banks/{bank_id}")
print(f"Deleted bank: {response.json()}")
```
