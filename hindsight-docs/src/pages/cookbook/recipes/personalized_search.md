---
sidebar_position: 10
---

# Personalized Search Agent with Hindsight Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/personalized_search.ipynb)
:::


A search assistant that learns your preferences, location, dietary needs, and lifestyle to provide contextually relevant search results.

## Features
- Learns location, dietary restrictions, and lifestyle
- Personalizes search queries based on context
- Remembers past searches and preferences
- Integrates with Tavily for real web search (optional)

## Prerequisites
- OpenAI API key
- Hindsight running locally via Docker (see setup below)
- Tavily API key (optional, for real web search)

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
# Tavily is optional - demo works with simulated results if not installed
!pip install -q hindsight-client openai tavily-python nest-asyncio
```

## 2. Configure API Keys

Enter your API keys when prompted. Tavily is optional - press Enter to skip for simulated search results.


```python
import getpass
import os

# Set OpenAI API key (used by both Hindsight and the demo)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Tavily is optional - for real web search
if not os.getenv("TAVILY_API_KEY"):
    tavily_key = getpass.getpass("Enter your Tavily API key (or press Enter to skip): ")
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

print("API keys configured!")
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

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Optional: Tavily for real web search
try:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    HAS_TAVILY = True
    print("Tavily configured - using real web search!")
except (ImportError, Exception) as e:
    HAS_TAVILY = False
    print("Note: Using simulated search results (Tavily not configured)")

USER_ID = "search-user-demo"

print("Clients initialized!")
```

## 4. Define Helper Functions


```python
def store_preference(preference: str) -> str:
    """Store a user preference."""
    hindsight.retain(
        bank_id=USER_ID,
        content=f"User preference: {preference}",
        metadata={"category": "preference"},
    )
    return f"Learned: {preference}"


def store_interaction(query: str, response: str) -> None:
    """Store a search interaction."""
    hindsight.retain(
        bank_id=USER_ID,
        content=f"Search query: {query}\nResult highlights: {response[:200]}",
        metadata={"category": "search_history"},
    )


def get_user_context(query: str) -> str:
    """Retrieve relevant user context."""
    memories = hindsight.recall(
        bank_id=USER_ID,
        query=f"preferences location dietary lifestyle {query}",
        budget="mid",
    )

    if memories and memories.results:
        return "\n".join(f"- {m.text}" for m in memories.results[:6])
    return ""


def personalized_search(query: str) -> str:
    """Perform a personalized search."""
    user_context = get_user_context(query)

    enhancement_prompt = f"""Given this user's preferences and the search query, suggest how to enhance the search.

User preferences:
{user_context if user_context else "No preferences recorded yet."}

Search query: {query}

Return a JSON object with:
- "enhanced_query": The improved search query incorporating relevant preferences
- "filters": Any specific filters to apply (e.g., "vegetarian", "within 5 miles")
- "reasoning": Brief explanation of personalizations applied"""

    enhancement = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": enhancement_prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    enhanced_info = enhancement.choices[0].message.content

    # Perform the search
    if HAS_TAVILY:
        search_results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )
        results_text = "\n".join(
            f"- {r['title']}: {r['content'][:150]}..."
            for r in search_results.get('results', [])
        )
    else:
        results_text = f"[Simulated search results for: {query}]"

    response_prompt = f"""Based on the search results and user preferences, provide a personalized summary.

User preferences:
{user_context if user_context else "No preferences recorded yet."}

Query: {query}

Search enhancement applied:
{enhanced_info}

Search results:
{results_text}

Provide a helpful, personalized response that takes into account their preferences."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": response_prompt}],
        temperature=0.7,
        max_tokens=500,
    )

    answer = response.choices[0].message.content
    store_interaction(query, answer)

    return answer


def get_preference_profile() -> str:
    """Get a summary of the user's preference profile."""
    profile = hindsight.reflect(
        bank_id=USER_ID,
        query="""Summarize what we know about this user:
        - Location and neighborhood
        - Dietary preferences and restrictions
        - Work style and schedule
        - Hobbies and interests
        - Family situation
        - Shopping preferences""",
        budget="high",
    )
    return profile.text if hasattr(profile, 'text') else str(profile)

print("Helper functions defined!")
```

## 5. Build User Profile


```python
print("Learning user preferences...")

preferences = [
    "Lives in San Francisco, Mission District",
    "Works remotely as a software engineer",
    "Vegetarian, prefers organic food when possible",
    "Has a 5-year-old daughter named Emma",
    "Enjoys hiking and outdoor activities on weekends",
    "Prefers quiet coffee shops for remote work",
    "Lactose intolerant, uses oat milk",
    "Interested in sustainable and eco-friendly products",
    "Usually free on Tuesday and Thursday afternoons",
    "Husband is allergic to nuts",
]

for pref in preferences:
    result = store_preference(pref)
    print(f"  {result}")
```

## 6. Personalized Search Results


```python
import time

print("=" * 60)
print("  Personalized Search Results")
print("=" * 60)

searches = [
    "Find a good coffee shop for working remotely",
    "Restaurant recommendations for a family dinner",
    "Birthday gift ideas for a 5-year-old",
]

for query in searches:
    print(f"\nSearch: {query}")
    print("-" * 40)
    result = personalized_search(query)
    print(result)
    time.sleep(1)
```

## 7. View Preference Profile


```python
print("=" * 60)
print("  User Preference Profile")
print("=" * 60)
print(get_preference_profile())
```

## 8. Try Your Own Search


```python
your_search = "Best hiking trails near me"  # Change this!

print(f"Search: {your_search}")
print("-" * 40)
print(personalized_search(your_search))
```

## 9. Cleanup


```python
hindsight.close()
print("Client connection closed.")
```
