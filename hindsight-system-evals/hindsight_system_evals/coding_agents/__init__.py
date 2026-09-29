"""Does the coding-agents plugin actually get the agent to READ memory?

Measured on real Claude Code, in a throwaway HOME, against a real Hindsight
server. The metric is one number: **searches per user turn** — how often a turn
makes the agent call ``hindsight_search_knowledge_pages`` (and its sibling
retrieval tools).

It exists because the observed rate on a developer's own machine was 0.5% of
turns, while the same sessions wrote memory 37 times. The plugin's prompt
changes are cheap to make and impossible to judge by reading them, so this
harness is the judge.

Read ``README.md`` for how the sandbox is built and what the numbers mean.
"""

from hindsight_system_evals.coding_agents.metric import UsageStats, read_usage_stats
from hindsight_system_evals.coding_agents.sandbox import Sandbox, build_sandbox
from hindsight_system_evals.coding_agents.scenario import BASELINE_PROMPTS, write_fixture_repo
from hindsight_system_evals.coding_agents.session import SessionResult, run_session

__all__ = [
    "BASELINE_PROMPTS",
    "Sandbox",
    "SessionResult",
    "UsageStats",
    "build_sandbox",
    "read_usage_stats",
    "run_session",
    "write_fixture_repo",
]
