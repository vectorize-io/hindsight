"""Manual integration test for hindsight-crewai.

Prerequisites:
  1. Hindsight API running on localhost:8888 (./scripts/dev/start-api.sh)
  2. OPENAI_API_KEY set (or configure CrewAI for another LLM provider)
  3. uv pip install -e . (from this directory)

Usage:
  uv run python test_manual.py
"""

from hindsight_crewai import configure, HindsightStorage, HindsightReflectTool
from crewai.memory.external.external_memory import ExternalMemory
from crewai import Agent, Crew, Task

BANK_ID = "crewai-test"
HINDSIGHT_URL = "http://localhost:8888"

# --- Configure ---

configure(hindsight_api_url=HINDSIGHT_URL, verbose=True)

storage = HindsightStorage(
    bank_id=BANK_ID,
    mission="Track research findings and summaries for a software team.",
)

reflect_tool = HindsightReflectTool(bank_id=BANK_ID, budget="mid")

# --- Smoke test (no LLM needed) ---

print("=== SMOKE TEST: save/search/reset ===\n")

storage.save("Python is great for data science", metadata={"task": "research"}, agent="Tester")
print("Saved memory.")

results = storage.search("What programming languages are useful?")
print(f"Search returned {len(results)} result(s):")
for r in results:
    print(f"  - [{r['score']}] {r['context']}")

print()

# --- Full crew test ---

print("=== RUN 1: Initial research ===\n")

researcher = Agent(
    role="Researcher",
    goal="Research topics and remember findings",
    backstory="You are a diligent researcher who remembers everything.",
    tools=[reflect_tool],
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Write summaries based on research",
    backstory="You write clear, concise summaries.",
    tools=[reflect_tool],
    verbose=True,
)

research_task = Task(
    description=(
        "Research the benefits of functional programming. "
        "List at least 3 key benefits with examples."
    ),
    expected_output="A list of functional programming benefits with examples.",
    agent=researcher,
)

summary_task = Task(
    description="Write a one-paragraph summary of the research findings.",
    expected_output="A concise summary paragraph.",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, summary_task],
    external_memory=ExternalMemory(storage=storage),
    verbose=True,
)

result = crew.kickoff()
print(f"\nRun 1 result:\n{result}\n")

# --- Second run: recall from memory ---

print("=== RUN 2: Recall from memory ===\n")

recall_task = Task(
    description=(
        "What do you already know about functional programming from previous research? "
        "Use the hindsight_reflect tool to check your memories."
    ),
    expected_output="A summary of what was previously learned.",
    agent=researcher,
)

crew2 = Crew(
    agents=[researcher],
    tasks=[recall_task],
    external_memory=ExternalMemory(storage=storage),
    verbose=True,
)

result2 = crew2.kickoff()
print(f"\nRun 2 result:\n{result2}\n")

# --- Cleanup ---

print("=== CLEANUP ===\n")
storage.reset()
print("Bank reset. Done.")
