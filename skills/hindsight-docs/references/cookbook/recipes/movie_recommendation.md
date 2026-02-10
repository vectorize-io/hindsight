---
sidebar_position: 8
---

# Movie Recommendation Assistant with Hindsight Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/movie_recommendation.ipynb)
:::


A personalized movie recommender that remembers your preferences, watch history, and tastes to give better suggestions over time.

## Features
- Remembers favorite genres, directors, and actors
- Tracks movies you've watched and enjoyed
- Provides contextual recommendations based on mood

## Prerequisites
- OpenAI API key
- Hindsight running locally via Docker (see setup below)

## Start Hindsight Locally

Before running this notebook, start Hindsight in a terminal:

```bash
export OPENAI_API_KEY="your-openai-api-key"

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

## 1. Install Dependencies


```python
!pip install -q hindsight-client openai nest-asyncio
```

## 2. Configure OpenAI API Key

Enter your OpenAI API key when prompted (used by both Hindsight and the demo).


```python
import getpass
import os

# Set OpenAI API key (used by both Hindsight and the demo)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

print("API key configured!")
```

## 3. Initialize Clients


```python
import nest_asyncio
nest_asyncio.apply()

from openai import OpenAI
from hindsight_client import Hindsight

# Initialize Hindsight client (connects to local Docker instance)
hindsight = Hindsight(
    base_url=os.getenv("HINDSIGHT_BASE_URL", "http://localhost:8888"),
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Unique identifier for this user's memory bank
USER_ID = "movie-fan-demo"

print("Clients initialized!")
```

## 4. Define Helper Functions

These functions demonstrate the three core Hindsight operations:
- **retain()**: Store memories
- **recall()**: Retrieve relevant memories
- **reflect()**: Synthesize insights from memories


```python
def get_recommendation(user_query: str) -> str:
    """
    Get a movie recommendation based on user query and remembered preferences.
    """
    # Recall relevant memories about this user's movie preferences
    memories = hindsight.recall(
        bank_id=USER_ID,
        query=f"movie preferences tastes genres {user_query}",
        budget="mid",
    )

    # Build context from memories
    memory_context = ""
    if memories and memories.results:
        memory_context = "\n".join(
            f"- {m.text}" for m in memories.results[:5]
        )

    # Generate recommendation with context
    system_prompt = f"""You are a helpful movie recommendation assistant.
You remember the user's preferences and past conversations to give personalized suggestions.

What you know about this user:
{memory_context if memory_context else "No previous preferences recorded yet."}

Give thoughtful, personalized recommendations based on their tastes.
If they mention new preferences, acknowledge them."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    recommendation = response.choices[0].message.content

    # Store this interaction for future context
    hindsight.retain(
        bank_id=USER_ID,
        content=f"User asked: {user_query}\nRecommendation given: {recommendation}",
        metadata={"category": "movie_recommendation"},
    )

    return recommendation


def store_preference(preference: str) -> None:
    """Store an explicit user preference."""
    hindsight.retain(
        bank_id=USER_ID,
        content=f"User preference: {preference}",
        metadata={"category": "preference"},
    )
    print(f"Stored preference: {preference}")


def get_preference_summary() -> str:
    """Get a summary of what we know about the user's movie tastes."""
    summary = hindsight.reflect(
        bank_id=USER_ID,
        query="Summarize this user's movie preferences, favorite genres, actors they like, and movies they've mentioned enjoying or disliking.",
        budget="high",
    )
    return summary.text if hasattr(summary, 'text') else str(summary)

print("Helper functions defined!")
```

## 5. Run the Demo

Watch how the assistant learns and remembers preferences across conversations.


```python
import time

print("=" * 60)
print("  Movie Recommendation Assistant with Memory")
print("=" * 60)
print()

# Simulate a conversation over time
conversations = [
    "I'm looking for a movie to watch tonight. Any suggestions?",
    "I really loved Inception and Interstellar. Christopher Nolan is amazing!",
    "Can you suggest something similar to those? I like mind-bending plots.",
    "Actually, I'm not in the mood for something heavy. Something lighter?",
    "I watched The Grand Budapest Hotel last week and loved it!",
    "What should I watch tonight? Remember what I like!",
]

for i, query in enumerate(conversations, 1):
    print(f"\n[Conversation {i}]")
    print(f"User: {query}")
    print("-" * 40)

    response = get_recommendation(query)
    print(f"Assistant: {response}")
    print()

    time.sleep(1)
```

## 6. View Learned Preferences

Use `reflect()` to synthesize what Hindsight has learned about your movie tastes.


```python
print("=" * 60)
print("  What I've learned about your movie tastes:")
print("=" * 60)
print(get_preference_summary())
```

## 7. Try Your Own Queries

Experiment with your own movie preferences!


```python
# Try your own query!
your_query = "I'm in the mood for a sci-fi thriller"  # Change this!

print(f"You: {your_query}")
print("-" * 40)
print(f"Assistant: {get_recommendation(your_query)}")
```

## 8. Cleanup

Close the Hindsight client connection.


```python
hindsight.close()
print("Client connection closed.")
```


```python

```


```python

```
