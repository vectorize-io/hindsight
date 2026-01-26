---
sidebar_position: 1
---

# OpenAI Agent + Hindsight Memory Integration


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/openai-fitness-coach)
:::


A fitness coach example demonstrating how to use **OpenAI Agents** with **Hindsight as a memory backend**.

## What This Demonstrates

This example showcases:

- **OpenAI Assistants** handling conversation logic
- **Hindsight** providing sophisticated memory storage & retrieval
- **Function calling** to bridge them together
- **Streaming responses** for real-time interaction (enabled by default)
- **Bidirectional memory** - both user data AND coach observations stored
- **System-level post-processing** - automatic opinion storage for reliability
- **Temporal-semantic memory** queries via function tools
- **Enhanced preference learning** - coach learns and respects user likes/dislikes
- **Real-world integration pattern** for adding memory to AI agents

## Architecture

```
User: "I ran 5K today, don't like tempo runs"
    |
OpenAI Assistant
    |
Function Call: store_memory(workout + preference)
    |
Hindsight API (stores as world/agent)
    |
OpenAI Assistant: "What should I focus on?"
    |
Function Call: retrieve_memories("workouts and preferences")
    |
Hindsight API (returns workouts + preferences)
    |
OpenAI Assistant (analyzes, gives advice)
    |
Function Call: store_memory(advice as opinion)
    |
Hindsight API (stores coach's observation)
    |
Personalized Answer
```

## Key Difference from Standard Demo

| Component | Standard Demo | OpenAI Integration |
|-----------|---------------|-------------------|
| **Conversation** | Hindsight `/think` endpoint | OpenAI Assistant API |
| **Memory** | Hindsight (built-in) | Hindsight (via function calling) |
| **LLM** | Configured in Hindsight | OpenAI GPT-4 |
| **Opinion Formation** | Automatic in `/think` | Explicit via `store_memory(type="opinion")` |
| **Best For** | Hindsight-native apps | Integrating memory into existing OpenAI agents |

## Quick Start

### Prerequisites

1. **OpenAI API Key**
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

2. **Hindsight API running**
   ```bash
   # Follow Hindsight setup instructions to start the API
   # Default: http://localhost:8888
   ```

3. **Install dependencies**
   ```bash
   pip install openai requests
   ```

### Run the Conversational Demo

```bash
cd openai-fitness-coach
export OPENAI_API_KEY=your_key_here
python demo_conversational.py
```

The demo showcases:
1. **Natural language workout logging** - Tell the coach what you did conversationally
2. **Preference learning** - Express likes/dislikes and watch the coach adapt
3. **Goal tracking** - Set goals, track progress, achieve milestones
4. **Bidirectional memory** - Both your activities AND coach's advice are stored
5. **Streaming responses** - See responses appear in real-time
6. **7 interactive phases** - From goal setting to achievement recognition

The demo uses a separate agent (`fitness-coach-demo`) to avoid mixing with real data.

## Usage

### Chat with Your Coach

**Interactive mode:**
```bash
python openai_coach.py
```

**Single question:**
```bash
python openai_coach.py "What did I do for training this week?"
```

## How It Works

### 1. Memory Tools (`memory_tools.py`)

Defines function tools that the OpenAI Agent can call:

```python
retrieve_memories(query, fact_types, top_k)
search_workouts(after_date, before_date, workout_type)
get_nutrition_summary(after_date, before_date)
get_user_goals()
get_coach_opinions(about)
```

Each function makes API calls to Hindsight to fetch relevant memories.

### 2. OpenAI Agent (`openai_coach.py`)

Creates an OpenAI Assistant with:
- Fitness coaching instructions
- Access to memory function tools
- Conversation management

When you ask a question:
1. User message is sent to OpenAI Assistant
2. Assistant decides which memory functions to call
3. Functions fetch data from Hindsight
4. Assistant generates response using retrieved context

### 3. Function Calling Flow

```python
# User asks: "What did I run this week?"

# OpenAI Assistant decides to call:
search_workouts(
    after_date="2024-11-18",
    workout_type="running"
)

# Function retrieves from Hindsight:
{
  "results": [
    {"text": "User completed 45-minute cardio workout: running..."},
    {"text": "User completed 60-minute cardio workout: running..."}
  ]
}

# OpenAI Assistant generates response:
"This week you've done two runs: a 45-minute run on Monday
and a longer 60-minute run on Wednesday. Great consistency!"
```

## Example Questions

Try asking:

```bash
python openai_coach.py "What does my training look like this week?"
python openai_coach.py "Based on my workouts, should I rest today?"
python openai_coach.py "How is my nutrition supporting my goals?"
python openai_coach.py "What's my progress toward my goal?"
python openai_coach.py "Compare my training this month to last month"
```

The agent will automatically:
1. Identify what memories it needs
2. Call the appropriate function tools
3. Retrieve data from Hindsight
4. Generate a personalized response

## Memory Types Retrieved

The OpenAI Agent can retrieve different memory types from Hindsight:

- **World Facts** (`fact_type: "world"`): Workouts, meals, activities
- **Agent Facts** (`fact_type: "agent"`): Goals, intentions
- **Opinions** (`fact_type: "opinion"`): Coach's observations about patterns

## Customization

### Add New Function Tools

Edit `memory_tools.py` to add new capabilities:

```python
def get_weekly_summary(week_offset: int = 0):
    """Get a summary of a specific week."""
    # Implementation
    pass

# Add to MEMORY_TOOLS list
MEMORY_TOOLS.append({
    "type": "function",
    "function": {
        "name": "get_weekly_summary",
        "description": "Get training summary for a specific week",
        # ... parameters
    }
})

# Add to FUNCTION_MAP
FUNCTION_MAP["get_weekly_summary"] = get_weekly_summary
```

### Modify Assistant Instructions

Edit `openai_coach.py` to change the coach's personality or behavior:

```python
assistant = client.beta.assistants.create(
    name="Your Custom Coach",
    instructions="Your custom instructions here...",
    model="gpt-4o-mini",
    tools=MEMORY_TOOLS
)
```

## Use Cases

This pattern works for any application that needs memory:

1. **Customer Support Agents** - Remember past conversations and issues
2. **Personal Assistants** - Remember preferences, schedules, past decisions
3. **Educational Tutors** - Track learning progress over time
4. **Health Coaches** - Monitor habits, progress, goals (like this example)
5. **Sales Assistants** - Remember customer interactions and preferences

## Integration Pattern

**To add Hindsight memory to your own OpenAI Agent:**

1. Define function tools that call Hindsight API
2. Register them with your OpenAI Assistant
3. Implement function handlers to execute Hindsight queries
4. Let OpenAI Assistant decide when to retrieve memories

The key benefit: **Separation of concerns**
- OpenAI = Conversation logic
- Hindsight = Memory storage, retrieval, temporal queries, entity linking

## When to Use This vs. Standard Hindsight

**Use OpenAI + Hindsight (this example) when:**
- You want OpenAI's conversation capabilities
- You're already using OpenAI Agents
- You want explicit control over when to retrieve memories
- You want to combine Hindsight with other OpenAI features

**Use Hindsight directly when:**
- You want a complete memory-first solution
- You want automatic memory retrieval and opinion formation
- You want to use different LLM providers (not just OpenAI)
- You want the `/think` endpoint's integrated approach

## Learning Points

After running this demo, you'll understand:

1. How to add sophisticated memory to any OpenAI Agent
2. How function calling bridges LLMs and memory systems
3. How temporal-semantic queries work via function tools
4. Real-world pattern for LLM + memory integration

## Core Files

- `demo_conversational.py` - Conversational demo showcasing preference learning and goal tracking
- `openai_coach.py` - OpenAI Assistant wrapper with streaming and memory integration
- `memory_tools.py` - Function calling tools that bridge to Hindsight API
- `.openai_assistant_id` - Saved assistant ID (auto-generated, gitignored)

## Common Issues

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY=your_api_key_here
```

**"Agent not found"**
- Make sure the Hindsight fitness-coach agent exists

**"Connection refused"**
- Make sure Hindsight API is running on localhost:8888

## Next Steps

1. Run the demo to see it in action
2. Try chatting with the coach: `python openai_coach.py`
3. Log your own workouts and meals
4. Experiment with different questions
5. Add custom function tools for your use case

---

**Built with:**
- OpenAI Assistants API
- Hindsight (temporal-semantic memory)
- Function calling for integration
