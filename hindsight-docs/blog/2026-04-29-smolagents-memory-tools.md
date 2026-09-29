---
title: "Agent Memory in SmolAgents: Retain, Recall, and Reflect"
authors: [benfrank241]
date: 2026-04-29T15:00:00Z
tags: [integrations, agents, smolagents, memory, guide, tutorial]
description: "Add persistent memory to SmolAgents with Hindsight. Retain facts, recall context, and reflect on agent decisions across sessions."
image: /img/blog/smolagents-memory-tools.png
hide_table_of_contents: true
---

![Agent Memory in SmolAgents: Retain, Recall, and Reflect](/img/blog/smolagents-memory-tools.png)

SmolAgents builds lightweight, practical agents that reason step-by-step. But agents without memory are stuck repeating work. Hindsight now integrates with SmolAgents as a set of tools that let your agents **retain observations, recall relevant context, and reflect on their own reasoning**—across sessions and tasks. This turns single-shot agents into learning systems.

<!-- truncate -->

## Why Memory Matters for SmolAgents

SmolAgents excels at building minimal, composable agents that reason about code, data, and systems. But traditional agents operate in isolation:
- Each task starts fresh, with no knowledge of previous runs
- Agents can't learn from past mistakes or successes
- Insights from one task don't inform the next
- Users repeat explanations and context constantly

Hindsight solves this by providing three memory tools: **Retain** (store facts), **Recall** (retrieve context), and **Reflect** (synthesize learnings). Together, they create agents that remember, learn, and improve over time.

## How Hindsight Integrates with SmolAgents

The Hindsight SmolAgents integration exposes three tools your agents can use:

1. **Retain Tool** - Store facts, decisions, and observations
   - Called automatically or explicitly within agent code
   - Extracts structured facts from unstructured reasoning
   - Tags facts by topic, date, or context

2. **Recall Tool** - Retrieve relevant context from memory
   - Query by question ("What did I learn about database indexing?")
   - Returns ranked, relevant facts with confidence scores
   - Integrates seamlessly into agent reasoning loops

3. **Reflect Tool** - Synthesize learnings and patterns
   - Analyze stored facts to extract high-level insights
   - Generate summaries of what the agent has learned
   - Identify gaps or repeated mistakes

All three tools are registered as SmolAgents tools, so your agent can call them like any other tool.

## Setting Up Hindsight with SmolAgents

First, install the Hindsight SmolAgents integration:

```bash
pip install hindsight-smolagents
```

Then configure it in your agent:

```python
from hindsight_smolagents import configure, create_hindsight_tools
from smolagents import CodeAgent, DuckDuckGoSearchTool

# Configure Hindsight (once, globally)
configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

# Create the three memory tools
memory_tools = create_hindsight_tools(bank_id="my-agent-memory")

# Create agent with memory tools
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()] + memory_tools,
    model_id="openai/gpt-4"
)

# Now your agent has memory
result = agent.run("Search for Python async patterns and remember what you learn")
```

The `create_hindsight_tools()` function returns three Tool instances: `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect`. The agent can call any of them like built-in tools.

## Real-World Use Cases

### Use Case 1: Code Review Agent

A code review agent that learns from past reviews:

```python
from hindsight_smolagents import configure, create_hindsight_tools
from smolagents import CodeAgent

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

memory_tools = create_hindsight_tools(bank_id="code-reviews")

agent = CodeAgent(
    tools=[CodeAnalysisTool()] + memory_tools,
    system_prompt="""
    You are a code review expert. Before reviewing, use hindsight_recall to 
    recall what you've learned about common issues in this codebase. 
    After reviewing, use hindsight_retain to save your findings so future reviews improve.
    """
)

# Each review feeds the next
for pr in pull_requests:
    result = agent.run(f"Review PR {pr.number}: {pr.diff}")
```

The agent builds institutional knowledge across reviews, recognizing patterns and catching recurring bugs.

### Use Case 2: Data Analysis Agent

An agent that analyzes datasets and remembers what it discovers:

```python
from hindsight_smolagents import configure, create_hindsight_tools
from smolagents import CodeAgent

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

memory_tools = create_hindsight_tools(bank_id="data-analysis")

agent = CodeAgent(
    tools=[PandasTool(), SQLQueryTool()] + memory_tools
)

# Agent queries the data, retains insights
result = agent.run("""
Analyze the sales database. Find trends. Use hindsight_retain to save what you learn.
Then answer: What does our data tell us about customer behavior?
""")

# Later, the agent recalls these insights
follow_up = agent.run("""
Use hindsight_recall to remember what you learned about customer behavior. 
Then answer: How should we adjust pricing based on what you've learned?
""")
```

The agent's analysis compounds over time—each new query can reference prior discoveries.

### Use Case 3: Research Assistant

An agent that reads papers and builds a knowledge base:

```python
from hindsight_smolagents import configure, create_hindsight_tools
from smolagents import CodeAgent

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

memory_tools = create_hindsight_tools(bank_id="research-papers")

agent = CodeAgent(
    tools=[DocumentReaderTool()] + memory_tools
)

# Process a series of papers
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
for paper in papers:
    agent.run(f"Read {paper}. Extract key findings and use hindsight_retain to save them.")

# Then synthesize
summary = agent.run("Use hindsight_reflect to analyze everything you've learned. What are the major themes?")
```

The agent synthesizes across multiple papers, extracting patterns that wouldn't be obvious from any single source.

## The Three Memory Tools

The `create_hindsight_tools()` function returns three SmolAgents Tool instances that your agent can use:

**hindsight_retain** - Store facts and observations
- Takes a string of content to remember
- Stores it in the memory bank with optional tags
- Called when the agent discovers something important
- Example: `hindsight_retain("Python async/await allows concurrent execution without threads")`

**hindsight_recall** - Retrieve relevant context
- Takes a query string about what you want to know
- Returns ranked facts from memory that match the query
- Example: `hindsight_recall("What have I learned about Python async patterns?")`

**hindsight_reflect** - Synthesize and extract patterns
- Takes a query about what to synthesize
- Analyzes stored facts to produce high-level insights
- Example: `hindsight_reflect("What are the main themes in everything I've learned about ML?")`

All three are registered as tools, so the agent calls them using their natural names within its reasoning.

## Example: Interactive Learning Loop

Here's a complete example showing an agent that learns over multiple interactions:

```python
from hindsight_smolagents import configure, create_hindsight_tools
from smolagents import CodeAgent, DuckDuckGoSearchTool

# Configure once globally
configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

# Create agent with memory tools
memory_tools = create_hindsight_tools(bank_id="learning-agent")
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()] + memory_tools
)

# First task: Research machine learning
print("=== Task 1: ML Research ===")
result1 = agent.run("""
Search for the latest machine learning breakthroughs in 2026.
Focus on: transformer improvements, efficiency, novel architectures.
Use hindsight_retain to save your findings.
""")

# Second task: React to new research
print("\n=== Task 2: Build on Prior Learning ===")
result2 = agent.run("""
Use hindsight_recall to remember what you learned about ML breakthroughs.
Now research how these apply to resource-constrained environments.
Use hindsight_retain to save what you discover.
""")

# Third task: Synthesize learnings
print("\n=== Task 3: Synthesize ===")
result3 = agent.run("""
Use hindsight_reflect to synthesize everything you've learned about ML in 2026.
What are the patterns? What's most important?
What should developers focus on?
""")

print(result3)  # Agent's high-level synthesis
```

Each task builds on the previous one. The agent recalls what it learned, applies it to new questions, and synthesizes across all interactions.

## Best Practices

**Explicit Retention:** Call retain() when the agent discovers something valuable, not on every step. This keeps memory focused.

**Contextual Recall:** Use specific queries. "What have I learned about database performance?" is better than "Tell me everything."

**Reflect Periodically:** After multiple tasks, use reflect() to synthesize what was learned and identify patterns.

**Tag for Organization:** Use tags like `["ml", "2026"]` or `["database", "performance"]` to organize facts by topic.

**Version Your Agent:** Different versions of your agent may have different memory needs. Use separate banks for different agent roles.

## Troubleshooting

**Recall Returns Empty:** The agent hasn't retained anything yet, or the query doesn't match stored facts. Check that retain() is being called and review what was stored.

**Memory Feels Redundant:** Facts get stored multiple times. Use tags and be selective about what you retain—quality over quantity.

**Reflect Output is Generic:** Provide context when calling reflect(). "Reflect on what you've learned about X" is more useful than open-ended reflection.

**API Errors:** Verify your bank_id exists and your API key is valid. Check the Hindsight Cloud dashboard to inspect stored facts.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [SmolAgents Documentation](https://github.com/huggingface/smolagents)
- [Hindsight Recall API](/developer/api/recall)
- [Hindsight Retain API](/developer/api/retain)
- [Hindsight SmolAgents Integration README](/sdks/integrations/smolagents)
