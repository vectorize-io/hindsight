# hindsight-crewai

Persistent memory for AI agent crews via Hindsight. Give your CrewAI crews long-term memory with fact extraction, entity tracking, and temporal awareness.

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

# Step 1: Configure connection
configure(hindsight_api_url="http://localhost:8888")

# Step 2: Create crew with Hindsight-backed memory
crew = Crew(
    agents=[
        Agent(role="Researcher", goal="Find information", backstory="..."),
        Agent(role="Writer", goal="Write reports", backstory="..."),
    ],
    tasks=[
        Task(description="Research AI trends", expected_output="Report"),
    ],
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

## Per-Agent Memory Banks

Give each agent its own isolated memory bank:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    per_agent_banks=True,  # Researcher -> "my-crew-researcher", Writer -> "my-crew-writer"
)
```

Or use a custom bank resolver for full control:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    bank_resolver=lambda base, agent: f"{base}-{agent.lower()}" if agent else base,
)
```

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

When the agent calls this tool, it gets a synthesized, contextual answer based on all relevant memories — not just raw facts.

## Bank Missions

Set a mission to guide how Hindsight processes and organizes memories:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    mission="Track software architecture decisions, technical debt, and team preferences.",
)
```

## Configuration

### Global Configuration

```python
from hindsight_crewai import configure

configure(
    hindsight_api_url="http://localhost:8888",  # Default: production API
    api_key="your-api-key",                     # Or set HINDSIGHT_API_KEY env var
    budget="mid",                               # Recall budget: low/mid/high
    max_tokens=4096,                            # Max tokens for recall results
    tags=["env:prod"],                          # Tags for stored memories
    recall_tags=["scope:global"],               # Tags to filter recall
    recall_tags_match="any",                    # Tag match mode: any/all/any_strict/all_strict
    verbose=True,                               # Enable logging
)
```

### Per-Storage Overrides

Constructor arguments override global configuration:

```python
storage = HindsightStorage(
    bank_id="my-crew",
    budget="high",       # Override global budget
    max_tokens=8192,     # Override global max_tokens
    tags=["team:alpha"], # Override global tags
)
```

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `hindsight_api_url` | Production API | Hindsight API URL |
| `api_key` | `HINDSIGHT_API_KEY` env | API key for authentication |
| `budget` | `"mid"` | Recall budget level (low/mid/high) |
| `max_tokens` | `4096` | Maximum tokens for recall results |
| `tags` | `None` | Tags applied when storing memories |
| `recall_tags` | `None` | Tags to filter when searching |
| `recall_tags_match` | `"any"` | Tag matching mode |
| `per_agent_banks` | `False` | Give each agent its own bank |
| `bank_resolver` | `None` | Custom (bank_id, agent) -> bank_id function |
| `mission` | `None` | Bank mission for memory organization |
| `verbose` | `False` | Enable verbose logging |
