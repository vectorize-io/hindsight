---
sidebar_position: 6
---

# Fitness Coach with Hindsight Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/fitness_tracker.ipynb)
:::


A personalized fitness assistant that tracks your workouts, diet, recovery, and progress over time to give contextual advice.

## Features
- Logs workout sessions with exercises and weights
- Tracks meals and dietary preferences
- Monitors recovery and sleep patterns
- Provides personalized training advice

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

from datetime import datetime
from openai import OpenAI
from hindsight_client import Hindsight

# Initialize Hindsight client (connects to local Docker instance)
hindsight = Hindsight(
    base_url=os.getenv("HINDSIGHT_BASE_URL", "http://localhost:8888"),
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USER_ID = "fitness-user-demo"

print("Clients initialized!")
```

## 4. Define Helper Functions


```python
def log_workout(workout_details: str) -> str:
    """Log a workout session with timestamp."""
    today = datetime.now().strftime("%B %d, %Y")
    hindsight.retain(
        bank_id=USER_ID,
        content=f"{today} - WORKOUT LOG: {workout_details}",
        metadata={"category": "workout", "date": today},
    )
    return f"Logged workout for {today}: {workout_details}"


def log_meal(meal_details: str) -> str:
    """Log a meal with timestamp."""
    today = datetime.now().strftime("%B %d, %Y")
    hindsight.retain(
        bank_id=USER_ID,
        content=f"{today} - MEAL LOG: {meal_details}",
        metadata={"category": "nutrition", "date": today},
    )
    return f"Logged meal for {today}: {meal_details}"


def log_recovery(recovery_details: str) -> str:
    """Log recovery information (sleep, soreness, etc.)."""
    today = datetime.now().strftime("%B %d, %Y")
    hindsight.retain(
        bank_id=USER_ID,
        content=f"{today} - RECOVERY LOG: {recovery_details}",
        metadata={"category": "recovery", "date": today},
    )
    return f"Logged recovery for {today}: {recovery_details}"


def store_user_profile(profile_info: str) -> str:
    """Store user profile information."""
    hindsight.retain(
        bank_id=USER_ID,
        content=f"USER PROFILE: {profile_info}",
        metadata={"category": "profile"},
    )
    return f"Stored profile info: {profile_info}"


def fitness_coach(user_query: str) -> str:
    """Get personalized fitness advice based on query and user history."""
    memories = hindsight.recall(
        bank_id=USER_ID,
        query=f"fitness workout diet recovery goals {user_query}",
        budget="high",
    )

    memory_context = ""
    if memories and memories.results:
        memory_context = "\n".join(f"- {m.text}" for m in memories.results[:10])

    system_prompt = f"""You are a knowledgeable and supportive fitness coach.
You have access to the user's workout history, diet logs, recovery notes, and personal profile.

What you know about this user:
{memory_context if memory_context else "No history recorded yet."}

Provide personalized, actionable advice based on their:
- Training history and progress
- Dietary preferences and restrictions
- Recovery patterns
- Personal goals

Be encouraging but realistic. Reference their specific history when relevant."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.7,
        max_tokens=600,
    )

    advice = response.choices[0].message.content

    hindsight.retain(
        bank_id=USER_ID,
        content=f"User asked: {user_query}\nCoach advised: {advice[:200]}...",
        metadata={"category": "coaching"},
    )

    return advice


def get_progress_report() -> str:
    """Generate a progress report based on workout history."""
    report = hindsight.reflect(
        bank_id=USER_ID,
        query="""Analyze this user's fitness journey:
        1. How consistent have they been with workouts?
        2. What progress have they made (weight lifted, exercises)?
        3. How is their recovery and sleep?
        4. What dietary patterns do you notice?
        5. What should they focus on next?""",
        budget="high",
    )
    return report.text if hasattr(report, 'text') else str(report)

print("Helper functions defined!")
```

## 5. Set Up User Profile


```python
print("Setting up user profile...")

profile_data = [
    "Name: Anish, Age: 26, Height: 5'10\", Weight: 72kg",
    "Goal: Building lean muscle, started gym 6 months ago",
    "Routine: Push-pull-legs split, 5x per week",
    "Rest days: Wednesday and Sunday",
    "Dietary restriction: Mild lactose intolerance, uses almond milk",
    "Health note: Occasional knee pain, avoids deep squats",
    "Supplements: Whey protein (lactose-free), magnesium",
    "Sleep: Aims for 7+ hours, performance drops under 6 hours",
]

for info in profile_data:
    store_user_profile(info)
    print(f"  Stored: {info[:50]}...")
```

## 6. Log Workout History


```python
print("Logging workout history...")

workouts = [
    "Push day: Bench press 3x8 @ 60kg, overhead press 4x12, tricep dips 3x10. Felt strong.",
    "Pull day: Deadlift 3x5 @ 80kg, barbell rows 4x10, bicep curls 3x12. Good session.",
    "Leg day: Leg press 4x12, hamstring curls 3x12, glute bridges 3x15. Knee felt okay.",
]

for workout in workouts:
    print(f"  {log_workout(workout)[:60]}...")

print("\nLogging recent meals...")
meals = [
    "Post-workout: Whey shake with almond milk, banana, oats",
    "Dinner: Grilled chicken, brown rice, steamed vegetables",
    "Snack: Greek yogurt (lactose-free) with berries",
]

for meal in meals:
    print(f"  {log_meal(meal)[:60]}...")

print("\nLogging recovery notes...")
recovery = [
    "Slept 7.5 hours, feeling well rested",
    "Some DOMS in legs from yesterday, using turmeric milk",
]

for note in recovery:
    print(f"  {log_recovery(note)[:60]}...")
```

## 7. Talk to Your Fitness Coach


```python
import time

print("=" * 60)
print("  Talking to your fitness coach...")
print("=" * 60)

queries = [
    "How much was I lifting for bench press recently?",
    "I slept poorly last night (only 5 hours). What should I do for today's workout?",
    "Suggest a post-workout meal that works with my dietary restrictions.",
    "My knee has been bothering me more. Any exercise modifications?",
]

for query in queries:
    print(f"\nUser: {query}")
    print("-" * 40)
    response = fitness_coach(query)
    print(f"Coach: {response}")
    time.sleep(1)
```

## 8. Generate Progress Report


```python
print("=" * 60)
print("  Progress Report")
print("=" * 60)
print(get_progress_report())
```

## 9. Try Your Own Query


```python
your_query = "What exercises should I do today?"  # Change this!

print(f"You: {your_query}")
print("-" * 40)
print(f"Coach: {fitness_coach(your_query)}")
```

## 10. Cleanup


```python
hindsight.close()
print("Client connection closed.")
```
