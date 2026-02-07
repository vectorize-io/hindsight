"""
Research crew with persistent memory via Hindsight.

This crew has a Researcher and a Writer. Run it multiple times —
the crew remembers what it learned in previous runs.

Usage:
    # First run — the crew has no memories yet
    uv run python examples/research_crew.py "What are the benefits of Rust?"

    # Second run — the crew remembers the Rust research
    uv run python examples/research_crew.py "Compare Rust with Go"

    # Third run — the crew has context from both prior runs
    uv run python examples/research_crew.py "Which language should I pick for a CLI tool?"

    # Reset memory and start fresh
    uv run python examples/research_crew.py --reset

Prerequisites:
    - Hindsight API running: ./scripts/dev/start-api.sh
    - OpenAI key: export OPENAI_API_KEY=sk-...
    - Install: cd hindsight-integrations/crewai && uv pip install -e .
"""

import sys

from hindsight_crewai import configure, HindsightStorage, HindsightReflectTool
from crewai.memory.external.external_memory import ExternalMemory
from crewai import Agent, Crew, Task

BANK_ID = "research-crew"
HINDSIGHT_URL = "http://localhost:8888"

# --- Setup ---

configure(hindsight_api_url=HINDSIGHT_URL, verbose=True)

storage = HindsightStorage(
    bank_id=BANK_ID,
    mission="Track technology research findings, comparisons, and recommendations.",
)

# Handle --reset
if "--reset" in sys.argv:
    storage.reset()
    print(f"Memory bank '{BANK_ID}' has been reset.")
    sys.exit(0)

# Get the research topic from the command line
topic = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--"))
if not topic:
    print("Usage: uv run python examples/research_crew.py \"your research topic\"")
    print("       uv run python examples/research_crew.py --reset")
    sys.exit(1)

# --- Agents ---

reflect_tool = HindsightReflectTool(bank_id=BANK_ID, budget="mid")

researcher = Agent(
    role="Researcher",
    goal="Research topics thoroughly, building on what you already know from past research.",
    backstory=(
        "You are a senior technology researcher. Before starting new research, "
        "always use the hindsight_reflect tool to check what you already know "
        "about the topic from previous sessions."
    ),
    tools=[reflect_tool],
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Write clear, concise summaries that incorporate both new and prior findings.",
    backstory=(
        "You write excellent technical summaries. Use the hindsight_reflect "
        "tool to recall prior research so your summaries build on previous work."
    ),
    tools=[reflect_tool],
    verbose=True,
)

# --- Tasks ---

research_task = Task(
    description=f"Research the following topic: {topic}. "
    "First check your memories for any prior research on related topics. "
    "Then provide a thorough analysis with at least 3 key points.",
    expected_output="A detailed analysis with key points and examples.",
    agent=researcher,
)

summary_task = Task(
    description="Write a one-paragraph executive summary of the research findings. "
    "Reference any connections to prior research if relevant.",
    expected_output="A concise executive summary paragraph.",
    agent=writer,
)

# --- Run ---

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, summary_task],
    external_memory=ExternalMemory(storage=storage),
    verbose=True,
)

print(f"\n{'='*60}")
print(f"  Research topic: {topic}")
print(f"  Memory bank: {BANK_ID}")
print(f"{'='*60}\n")

result = crew.kickoff()

print(f"\n{'='*60}")
print("  FINAL OUTPUT")
print(f"{'='*60}\n")
print(result)
