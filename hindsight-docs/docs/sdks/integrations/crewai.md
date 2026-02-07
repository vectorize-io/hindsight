---
sidebar_position: 5
---

# CrewAI

Persistent memory for AI agent crews via [CrewAI](https://github.com/crewAIInc/crewAI). Give your crews long-term memory with fact extraction, entity tracking, and temporal awareness.

## Features

- **Drop-in Storage Backend** - Implements CrewAI's `Storage` interface for `ExternalMemory`
- **Automatic Memory Flow** - CrewAI automatically stores task outputs and retrieves relevant memories
- **Per-Agent Banks** - Optionally give each agent its own isolated memory bank
- **Reflect Tool** - Agents can explicitly reason over memories with disposition-aware synthesis
- **Simple Configuration** - Configure once, use everywhere

## Installation

```bash
pip install hindsight-crewai
```

## Quick Start

```python
from hindsight_crewai import configure, HindsightStorage
from crewai.memory.external.external_memory import ExternalMemory
from crewai import Agent, Crew, Task

configure(hindsight_api_url="http://localhost:8888")

crew = Crew(
    agents=[Agent(role="Researcher", goal="Find information", backstory="...")],
    tasks=[Task(description="Research AI trends", expected_output="Report")],
    external_memory=ExternalMemory(
        storage=HindsightStorage(bank_id="my-crew")
    ),
)

crew.kickoff()
```

That's it. CrewAI will automatically:
- **Query memories** at the start of each task
- **Store task outputs** to Hindsight after each task completes

Memories persist across crew runs, so your crew learns over time.

## How It Works

The integration maps CrewAI's 3-method `Storage` interface to Hindsight's API:

| CrewAI | Hindsight | What happens |
|--------|-----------|--------------|
| `save(value, metadata, agent)` | `retain(bank_id, content, ...)` | Task output is stored. Hindsight extracts facts, entities, and relationships from the raw text. |
| `search(query, limit)` | `recall(bank_id, query, ...)` | CrewAI constructs a query from the task description. Hindsight runs semantic search, BM25, graph traversal, and reranking. |
| `reset()` | `delete_bank(bank_id)` | Wipes the bank and optionally recreates it with its original mission. |

CrewAI calls `search()` automatically at the start of each task and `save()` after each task completes.

## Configuration Options

```python
from hindsight_crewai import configure

configure(
    hindsight_api_url="http://localhost:8888",  # Hindsight API URL
    api_key="your-api-key",                     # Or set HINDSIGHT_API_KEY env var
    budget="mid",                               # Recall budget: "low", "mid", "high"
    max_tokens=4096,                            # Max tokens for recall results
    tags=["env:prod"],                          # Tags for stored memories
    recall_tags=["scope:global"],               # Tags to filter recall
    recall_tags_match="any",                    # Tag match: any/all/any_strict/all_strict
    verbose=True,                               # Enable logging
)
```

### Per-Storage Overrides

Constructor arguments override global configuration:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    budget="high",
    max_tokens=8192,
    tags=["team:alpha"],
)
```

## Bank Missions

Set a mission to guide how Hindsight processes and organizes memories:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    mission="Track software architecture decisions, technical debt, and team preferences.",
)
```

## Per-Agent Memory Banks

Give each agent its own isolated memory bank:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    per_agent_banks=True,
    # Researcher -> "my-crew-researcher"
    # Writer     -> "my-crew-writer"
)
```

Or use a custom bank resolver for full control:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    bank_resolver=lambda base, agent: f"{base}-{agent.lower()}" if agent else base,
)
```

:::info
When `per_agent_banks=True`, the automatic `search()` at task start queries the base bank (shared context), since CrewAI's `search()` method does not receive the agent parameter. For per-agent search isolation, create separate `HindsightStorage` instances per agent.
:::

## Reflect Tool

CrewAI's storage interface only supports save/search/reset. To give agents access to Hindsight's `reflect` (disposition-aware memory synthesis), add it as a tool:

```python
from hindsight_crewai import HindsightReflectTool

reflect_tool = HindsightReflectTool(
    bank_id="my-crew",
    budget="mid",
    reflect_context="You are helping a software team track decisions.",
)

agent = Agent(
    role="Analyst",
    goal="Analyze project history",
    backstory="...",
    tools=[reflect_tool],
)
```

When the agent calls this tool, it gets a synthesized, contextual answer based on all relevant memories rather than raw fact snippets.

## Full Example

A research crew that remembers findings across runs:

```python
from hindsight_crewai import configure, HindsightStorage, HindsightReflectTool
from crewai.memory.external.external_memory import ExternalMemory
from crewai import Agent, Crew, Task

configure(hindsight_api_url="http://localhost:8888")

storage = HindsightStorage(
    bank_id="research-crew",
    mission="Track technology research findings and comparisons.",
)

reflect_tool = HindsightReflectTool(bank_id="research-crew", budget="mid")

researcher = Agent(
    role="Researcher",
    goal="Research topics, building on prior knowledge.",
    backstory="Before starting, use hindsight_reflect to check what you already know.",
    tools=[reflect_tool],
)

writer = Agent(
    role="Writer",
    goal="Write summaries incorporating prior findings.",
    backstory="Use hindsight_reflect to recall prior research.",
    tools=[reflect_tool],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[
        Task(description="Research the benefits of Rust", expected_output="Analysis", agent=researcher),
        Task(description="Write an executive summary", expected_output="Summary", agent=writer),
    ],
    external_memory=ExternalMemory(storage=storage),
)

# Run 1: researches Rust, stores findings
crew.kickoff()

# Run 2: recalls Rust research when comparing with Go
crew.tasks[0].description = "Compare Rust with Go"
crew.kickoff()
```

## API Reference

### Configuration

| Function | Description |
|----------|-------------|
| `configure(...)` | Set global connection and default settings |
| `get_config()` | Get current configuration |
| `reset_config()` | Reset configuration to None |

### Storage

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bank_id` | required | Hindsight memory bank ID |
| `hindsight_api_url` | from config | Override API URL |
| `api_key` | from config | Override API key |
| `budget` | `"mid"` | Recall budget (low/mid/high) |
| `max_tokens` | `4096` | Max tokens for recall results |
| `tags` | `None` | Tags applied when storing |
| `recall_tags` | `None` | Tags to filter when searching |
| `recall_tags_match` | `"any"` | Tag matching mode |
| `per_agent_banks` | `False` | Give each agent its own bank |
| `bank_resolver` | `None` | Custom `(bank_id, agent) -> bank_id` |
| `mission` | `None` | Bank mission for memory organization |
| `verbose` | `False` | Enable verbose logging |

### Reflect Tool

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bank_id` | required | Hindsight memory bank ID |
| `budget` | `"mid"` | Reflect budget (low/mid/high) |
| `reflect_context` | `None` | Additional context for reasoning |
| `hindsight_api_url` | from config | Override API URL |
| `api_key` | from config | Override API key |

## Requirements

- Python >= 3.10
- crewai >= 0.86.0
- A running Hindsight API server
