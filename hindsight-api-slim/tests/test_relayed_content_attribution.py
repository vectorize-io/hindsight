"""Content an agent relays from a tool is not the agent's or the user's own experience (#6818).

A Hermes user asked their agent to summarise another creator's video. The reply kept the
creator's first person ("I tested 65/75/85% hydration") and extraction stored it as the agent's
own experiment. Nothing in a plain ``User:`` / ``Assistant:`` payload says the reply is relayed,
and a rule in the extraction prompt alone did not fix it (tried on #6818: it only worked when its
example matched the input).

The Hermes plugin now sends what the agent fetched as a ``tool`` message (the tool's name and
arguments, never its output) and adds a sentence to the retain context. These tests pin that the
model reads that payload as relayed, and still keeps the agent's own work as its own.

Real-LLM tests: attribution is the model's call, MockLLM cannot simulate it.
"""

import json
from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

# The context the Hermes plugin sends: its default label plus `_RELAYED_CONTENT_NOTE`
# (hindsight-integrations/hermes/__init__.py). Keep the two in step.
_CONTEXT = (
    "conversation between Hermes Agent and the User. Tool messages show what the assistant fetched "
    "(pages, videos, files), written by other people; when the Assistant repeats or summarises them, their "
    "claims and first person belong to that source, not to the User or the Assistant."
)


def _turn(user: str, tool_call: str, reply: str) -> str:
    """One turn in the shape the Hermes plugin retains it."""
    return json.dumps(
        [
            {"role": "user", "content": f"User: {user}"},
            {"role": "tool", "content": f"Tool call: {tool_call}"},
            {"role": "assistant", "content": f"Assistant: {reply}"},
        ]
    )


async def _facts_summary(text: str) -> str:
    facts, _, _ = await extract_facts_from_text(
        text=text,
        event_date=datetime(2026, 11, 20),
        llm_config=LLMConfig.from_env(),
        context=_CONTEXT,
        config=_get_raw_config(),
    )
    return "\n".join(f"- [{f.fact_type}] {f.fact}" for f in facts)


_PARTICIPANTS = "the user, 'User', the assistant, 'Assistant' or 'Hermes Agent'"
_FORMAT = "Each fact is shown as '- [fact_type] text'; 'experience' means something the assistant itself did."


@pytest.mark.asyncio
async def test_an_article_relayed_in_its_authors_first_person_is_credited_to_the_author():
    summary = await _facts_summary(
        _turn(
            "what does this article say about cold plunges? https://coldwellness.example/90-days",
            "web_extract(url=https://coldwellness.example/90-days)",
            "I did a cold plunge every morning for 90 days. You will feel awful the first week, but my resting "
            "heart rate dropped 6 bpm and I sleep better. I'd never go back.",
        )
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            f"No fact says that {_PARTICIPANTS} did cold plunges, had their heart rate drop, sleeps better, will "
            "feel awful, or would never go back, and no such fact is an 'experience'. Those are the article "
            "author's: attributed to the article or its author, or left out. Facts that the user asked about "
            "the article, or that the assistant summarised it, are fine. An empty list also passes."
        ),
        context=(
            "The user asked what an article says. The assistant, after fetching the page, answered in the "
            f"article author's own first person. Neither the user nor the assistant (an AI) did any of it. {_FORMAT}"
        ),
        msg=f"Relayed article content was attributed to a conversation participant. Facts:\n{summary}",
    )


@pytest.mark.asyncio
async def test_a_video_summarised_in_its_creators_first_person_is_credited_to_the_creator():
    summary = await _facts_summary(
        _turn(
            "summarise this sourdough video for me https://youtube.com/watch?v=abc123",
            "youtube_transcript(url=https://youtube.com/watch?v=abc123)",
            "I tested 65%, 75% and 85% hydration side by side. At 85% the dough was impossible to shape, so I "
            "settled on 75%. You can see the crumb is much more open.",
        )
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            f"No fact says that {_PARTICIPANTS} tested hydration levels, shaped dough, or settled on 75%, and no "
            "such fact is an 'experience'. Those are the video creator's: attributed to the video or its "
            "creator, or left out. Facts that the user asked for a summary are fine. An empty list also passes."
        ),
        context=(
            "The user asked for a summary of a video. The assistant, after fetching its transcript, answered in "
            f"the creator's own first person. Neither the user nor the assistant (an AI) baked anything. {_FORMAT}"
        ),
        msg=f"Relayed video content was attributed to a conversation participant. Facts:\n{summary}",
    )


@pytest.mark.asyncio
async def test_work_the_agent_did_with_a_tool_stays_the_agents_own():
    """The tool message and context must not make the agent's real work read as relayed."""
    summary = await _facts_summary(
        _turn(
            "the timeout test is failing, can you fix it?",
            "read_file(path=tests/test_api.py)",
            "I read tests/test_api.py, found the 1s timeout was too short for the CI runner, raised it to 5s and "
            "the test passes now.",
        )
    )
    await assert_meets_criteria(
        response=summary,
        criteria=(
            "At least one 'experience' fact says the assistant (or Hermes Agent) raised the test timeout from 1s "
            "to 5s or fixed the failing test. No fact credits that fix to a file, an author or another source."
        ),
        context=f"The assistant read a test file and fixed the test itself. {_FORMAT}",
        msg=f"The agent's own fix was not kept as its experience. Facts:\n{summary}",
    )
