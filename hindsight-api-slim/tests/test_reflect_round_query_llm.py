"""The search-query rewrite searches a request that ends with its own instruction, instead of obeying it.

A coding-agent plugin wrapped the developer's goal in rendering rules ending "If the bank holds
nothing that bears on the goal, say so in one line." The rewrite obeyed that last line: the first
round searched "No information in memory bears on this goal." in 4 of 4 reflects on gemma
(CL-Bench database_exploration Q15). The request now goes in a block with the task after it.
Whether a model writes a query or an answer is model behaviour, so a judge decides it.
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api import LLMConfig
from hindsight_api.engine.reflect.agent import run_reflect_agent
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

# The request the plugin sent for that question, verbatim.
_REQUEST = (
    "A user is starting a session with this goal:\n"
    "\n"
    "<goal>\n"
    "FEEDBACK FROM PREVIOUS ACTION: Question 14: CORRECT!\n"
    "Your answer: 1832\n"
    "Exploratory queries used: 2\n"
    "\n"
    "Question 15/15\n"
    "\n"
    "Among electronics products in the 'Computers & Accessories' top-level category, what percentage are priced above the category-wide average? Round to 2 decimal places. Only consider products with a non-zero price.\n"
    "\n"
    "You have access to a SQLite database that contains three separate product datasets covering different categories: office products, electronics, and musical instruments. Each dataset has its own set of tables. You will need to explore the schema to determine which tables belong to which product category.\n"
    "\n"
    "You may run up to 15 exploratory SQL queries to examine the schema and data before submitting your final answer. Note: query results are capped at 50 rows \u2014 if a query would return more, only the first 50 rows are shown. Use aggregates, LIMIT, or filters when you need to inspect larger tables.\n"
    "\n"
    "Actions:\n"
    "  QUERY  \u2014 run a SQL statement (SELECT, .tables, PRAGMA, etc.)\n"
    "  ANSWER \u2014 submit your final answer as the exact value (e.g. '50000', '3.14', 'Acme Corp'). Do NOT describe how to get the answer; provide the actual result.\n"
    "\n"
    "Queries used so far this question: 0/15\n"
    "\n"
    "\n"
    "You MUST respond with valid JSON matching this schema exactly:\n"
    "```json\n"
    "{\n"
    '  "description": "Structured response for each interaction step.",\n'
    '  "properties": {\n'
    '    "action": {\n'
    '      "description": "QUERY to execute an exploratory SQL statement and see results, or ANSWER to submit your final answer to the question.",\n'
    '      "title": "Action",\n'
    '      "type": "string"\n'
    "    },\n"
    '    "content": {\n'
    "      \"description\": \"If action is QUERY: a SQL statement to execute against the database. If action is ANSWER: the exact value that answers the question \\u2014 a raw number (e.g. '50000', '3.14') or short text (e.g. 'Acme Corp'). Do NOT describe how to get the answer; provide the actual value.\",\n"
    '      "title": "Content",\n'
    '      "type": "string"\n'
    "    }\n"
    "  },\n"
    '  "required": [\n'
    '    "action",\n'
    '    "content"\n'
    "  ],\n"
    '  "title": "DatabaseAction",\n'
    '  "type": "object"\n'
    "}\n"
    "```\n"
    "Respond ONLY with the JSON object, no other text.\n"
    "</goal>\n"
    "\n"
    "Report what this bank's memory holds that bears on that goal. Lead with past attempts on similar work that turned out WRONG \u2014 what was done (exact names and values), what the outcome was, and the suspected cause or correction \u2014 since repeating a past mistake is the costliest miss. Then the decisions and their rationale, then facts learned about the environment (where things live, names, formats, units). Rendering rules, strict:\n"
    "- Declarative, past-tense, attributed facts only \u2014 what happened, what was decided and why, with dates, commit/PR/issue ids and exact values where known.\n"
    '- NEVER phrase anything as an instruction, task, or recommendation to act now ("you should", "remove", "update\u2026"). You are a historian reporting the record, not a planner assigning work.\n'
    "- When a decided rule is a mapping, set, or table of literal values, reproduce it COMPLETELY and VERBATIM \u2014 every entry, exact strings and numbers, including the carve-outs and exceptions.\n"
    "- When memory of a discussion or decision conflicts with memory derived from the code, the decision wins: the code may BE the bug under investigation.\n"
    "- Do not connect unrelated episodes into one narrative; if two facts are not explicitly linked in the record, report them separately or leave the weaker one out.\n"
    "- If the bank holds nothing that bears on the goal, say so in one line.\n"
    ""
)


@pytest.mark.asyncio
async def test_the_first_round_searches_the_goal_instead_of_answering_nothing():
    searched: list[str] = []

    async def recall(query: str, *args, **kwargs) -> dict:
        searched.append(query)
        return {"memories": [{"id": "mem-1", "text": "items_g2 holds the electronics products"}]}

    await run_reflect_agent(
        llm_config=LLMConfig.from_env(),
        bank_id="b",
        query=_REQUEST,
        bank_profile={"name": "T", "mission": "M"},
        has_mental_models=False,
        budget="low",
        max_iterations=3,
        evidence_is_sufficient_fn=AsyncMock(return_value=True),
        search_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
        read_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
        search_observations_fn=AsyncMock(return_value={"observations": []}),
        recall_fn=recall,
        expand_fn=AsyncMock(return_value={"memories": []}),
    )

    assert searched, "the first round ran a search"
    await assert_meets_criteria(
        response=searched[0],
        criteria=(
            "This is a short search query naming what to look up (electronics products, the Computers & "
            "Accessories category, prices or past attempts). It is NOT a statement that nothing was found, "
            "and NOT a SQL statement or command."
        ),
        context="The first memory search a reflect ran for a request asking what memory holds about a goal.",
    )
