"""
Test suite for fact extraction output size validation.

Ensures that fact extraction doesn't produce excessively verbose output
relative to input size.
"""

import json
from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


class TestFactExtractionOutputRatio:
    """Tests for output size relative to input."""

    @pytest.mark.asyncio
    async def test_output_ratio_simple_text(self):
        """
        Test that output size is reasonable for simple text.

        The total output (all fact texts combined) should not be excessively
        larger than the input text.
        """
        text = """
I went to the grocery store yesterday and bought some apples and oranges.
The weather was really nice, sunny with a light breeze.
I ran into my neighbor Sarah who mentioned she's planning a trip to Italy next month.
"""

        context = "Personal diary entry"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 6, 15),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        input_length = len(text)
        output_length = sum(len(f.fact) for f in facts)
        ratio = output_length / input_length if input_length > 0 else 0

        print(f"\nSimple text test:")
        print(f"  Input length: {input_length} chars")
        print(f"  Output length: {output_length} chars")
        print(f"  Number of facts: {len(facts)}")
        print(f"  Output/Input ratio: {ratio:.2f}")
        print(f"  Facts:")
        for i, f in enumerate(facts):
            print(f"    [{i}] ({len(f.fact)} chars): {f.fact[:100]}...")

        # Output should not be more than 5x the input
        assert ratio < 5.0, (
            f"Output/input ratio {ratio:.2f} is too high! "
            f"Input: {input_length} chars, Output: {output_length} chars. "
            f"Facts: {[f.fact for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_output_ratio_conversation(self):
        """
        Test output ratio for a typical conversation.
        """
        text = """
User: Hey, I'm looking for a good restaurant for my anniversary dinner.
Assistant: I'd recommend La Maison for a romantic atmosphere. They have excellent French cuisine.
User: That sounds great! We love French food. What's the price range?
Assistant: It's upscale, around $100-150 per person. They also have a great wine selection.
User: Perfect, I'll make a reservation for Saturday at 7pm.
"""

        context = "Restaurant recommendation conversation"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 6, 15),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        input_length = len(text)
        output_length = sum(len(f.fact) for f in facts)
        ratio = output_length / input_length if input_length > 0 else 0

        print(f"\nConversation test:")
        print(f"  Input length: {input_length} chars")
        print(f"  Output length: {output_length} chars")
        print(f"  Number of facts: {len(facts)}")
        print(f"  Output/Input ratio: {ratio:.2f}")
        print(f"  Facts:")
        for i, f in enumerate(facts):
            print(f"    [{i}] ({len(f.fact)} chars): {f.fact[:100]}...")

        # Output should not be more than 5x the input
        assert ratio < 5.0, (
            f"Output/input ratio {ratio:.2f} is too high! Input: {input_length} chars, Output: {output_length} chars"
        )

    @pytest.mark.asyncio
    async def test_output_ratio_longer_text(self):
        """
        Test output ratio for a longer piece of text.
        """
        text = """
Last weekend was incredible. On Saturday morning, I woke up early and went for a 5-mile run
through the park near my house. The cherry blossoms were in full bloom, which made the whole
experience magical. After the run, I met up with my college friend Mike at our favorite cafe
downtown. We hadn't seen each other in about six months, so we had a lot to catch up on.

Mike told me about his new job at a tech startup in San Francisco. He's working as a senior
engineer there and seems really excited about the projects they're building. Something about
AI-powered healthcare solutions. He mentioned they're looking for more engineers and asked if
I'd be interested in applying. I told him I'd think about it, but honestly, I'm pretty happy
with my current position.

In the afternoon, we went to see a movie - the new sci-fi thriller that everyone's been talking
about. I thought it was okay, maybe a 7 out of 10. Mike loved it though. He's always been more
into action-heavy films than I am.

Sunday was more relaxed. I spent most of the day working on my photography hobby. I've been
learning to use Lightroom to edit my photos, and I finally feel like I'm getting the hang of it.
I edited about 20 photos from my recent trip to the mountains.
"""

        context = "Personal blog post"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 4, 15),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        input_length = len(text)
        output_length = sum(len(f.fact) for f in facts)
        ratio = output_length / input_length if input_length > 0 else 0

        print(f"\nLonger text test:")
        print(f"  Input length: {input_length} chars")
        print(f"  Output length: {output_length} chars")
        print(f"  Number of facts: {len(facts)}")
        print(f"  Output/Input ratio: {ratio:.2f}")
        print(f"  Avg fact length: {output_length / len(facts):.0f} chars" if facts else "N/A")
        print(f"  Facts:")
        for i, f in enumerate(facts):
            print(f"    [{i}] ({len(f.fact)} chars): {f.fact[:100]}...")

        # Output should not be more than 4x the input for longer texts
        # (ratio should decrease as input grows)
        assert ratio < 4.0, (
            f"Output/input ratio {ratio:.2f} is too high! Input: {input_length} chars, Output: {output_length} chars"
        )

        # Also check that individual facts aren't excessively long
        max_fact_length = max(len(f.fact) for f in facts) if facts else 0
        assert max_fact_length < 1000, f"Individual fact too long: {max_fact_length} chars. Facts should be concise."

    @pytest.mark.asyncio
    async def test_token_ratio_with_locomo_conversation(self):
        """
        Test output ratio with a realistic locomo conversation.

        The user reported: input_tokens=4714, output_tokens=24824, ratio=5.27
        This test uses real conversation data to check for excessive output.
        """
        import os

        # Load locomo conversation
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "locomo_conversation_sample.json")
        with open(fixture_path, "r") as f:
            data = json.load(f)

        # Use session_1 (a realistic conversation between Caroline and Melanie)
        session = data["conversation"]["session_1"]

        # Convert to text format
        text = "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in session])

        context = f"Conversation between {data['conversation']['speaker_a']} and {data['conversation']['speaker_b']}"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2023, 5, 8),  # Date from locomo dataset
            context=context,
            llm_config=llm_config,
            agent_name=data["conversation"]["speaker_a"],
            config=_get_raw_config(),
        )

        # Calculate ratios
        input_length = len(text)
        output_length = sum(len(f.fact) for f in facts)
        text_to_output_ratio = output_length / input_length if input_length > 0 else 0

        print(f"\nLocomo conversation test:")
        print(f"  Input text: {input_length} chars (~{input_length // 4} tokens)")
        print(f"  Output text: {output_length} chars (~{output_length // 4} tokens)")
        print(f"  Number of facts: {len(facts)}")
        print(f"  Output/Input text ratio: {text_to_output_ratio:.2f}")
        print(f"  Sample facts:")
        for i, f in enumerate(facts[:5]):  # Show first 5
            print(f"    [{i}] ({len(f.fact)} chars): {f.fact[:80]}...")
        if len(facts) > 5:
            print(f"    ... and {len(facts) - 5} more")

        # The output should not be more than 4x the input TEXT
        # This catches the extreme 5.27x case reported by the user
        assert text_to_output_ratio < 4.0, (
            f"Output/input text ratio {text_to_output_ratio:.2f} is too high! "
            f"Input text: {input_length} chars, Output: {output_length} chars. "
            f"Number of facts: {len(facts)}"
        )

        # Sanity check on number of facts
        # A conversation shouldn't produce an unreasonable number of facts
        num_turns = len(session)
        max_expected_facts = num_turns * 2  # At most 2 facts per conversation turn

        assert len(facts) <= max_expected_facts, (
            f"Too many facts: {len(facts)} for {num_turns} conversation turns. Expected at most {max_expected_facts}."
        )

    @pytest.mark.asyncio
    async def test_number_of_facts_reasonable(self):
        """
        Test that the number of extracted facts is reasonable.

        We shouldn't extract way more facts than there are sentences/statements
        in the input.
        """
        text = """
I love coffee in the morning.
My favorite restaurant is Olive Garden.
I work as a software engineer at Google.
My dog's name is Max.
I'm planning to visit Japan next year.
"""

        context = "Personal info"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 6, 15),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        # Count approximate number of statements (sentences)
        num_statements = len([s for s in text.split(".") if s.strip()])

        print(f"\nNumber of facts test:")
        print(f"  Input statements: ~{num_statements}")
        print(f"  Extracted facts: {len(facts)}")
        print(f"  Facts:")
        for i, f in enumerate(facts):
            print(f"    [{i}]: {f.fact[:80]}...")

        # Should not extract more than 2x the number of input statements
        assert len(facts) <= num_statements * 2, (
            f"Too many facts extracted: {len(facts)} for ~{num_statements} input statements"
        )
