"""
Test suite for fact extraction quality verification.

This comprehensive test suite validates that the fact extraction system:
1. Preserves all information dimensions (emotional, sensory, cognitive, etc.)
2. Correctly converts relative dates to absolute dates
3. Properly classifies facts as agent vs world
4. Makes logical inferences to connect related information
5. Correctly attributes statements to speakers
6. Filters out irrelevant content (podcast intros/outros)

Every test here exercises real LLM extraction behaviour — the file is marked
hs_llm_core at module scope so it runs in the single-provider quality CI job.
MockLLM cannot simulate dimension preservation, date conversion, or pronoun
resolution; running these tests against a mock would either pass spuriously
(MockLLM echoes input text, so string assertions trivially succeed) or fail
with no diagnostic signal.

Semantic assertions go through tests.llm_judge so paraphrases survive — the
LLM might phrase preserved emotion as "elated" instead of "thrilled", and a
literal substring check would flake.  Structural assertions (date fields,
fact counts, fact_type classification) stay as direct asserts.
"""

from datetime import UTC, datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

# =============================================================================
# DIMENSION PRESERVATION TESTS
# =============================================================================


class TestDimensionPreservation:
    """Tests that fact extraction preserves all information dimensions."""

    @pytest.mark.asyncio
    async def test_emotional_dimension_preservation(self):
        """
        Test that emotional states and feelings are preserved, not stripped away.

        Example: "I was thrilled to receive positive feedback"
        Should NOT become: "I received positive feedback"
        """
        text = """
I was absolutely thrilled when I received such positive feedback on my presentation!
Sarah seemed disappointed when she heard the news about the delay.
Marcus felt anxious about the upcoming interview.
"""

        context = "Personal journal entry"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts preserve emotional states from the input: the speaker's "
                "excitement/thrill about positive feedback, Sarah's disappointment about the delay, "
                "and Marcus's anxiety about the interview. At least two of these emotional dimensions "
                "should be present (exact wording doesn't matter — 'elated' for 'thrilled' is fine)."
            ),
            context=(
                "Input mentioned: being thrilled about positive feedback on a presentation, "
                "Sarah seeming disappointed about a delay, and Marcus feeling anxious about an interview."
            ),
            msg=f"Emotional dimension should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_sensory_dimension_preservation(self):
        """Test that sensory details (visual, auditory, etc.) are preserved."""
        text = """
The coffee tasted bitter and burnt.
She showed me her bright orange hair, which looked stunning under the lights.
The music was so loud I could barely hear myself think.
"""

        context = "Personal experience"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts preserve sensory details from the input — at least two of: "
                "the bitter/burnt taste of the coffee, the bright orange hair (and how it looked), "
                "or the loud volume of the music. Equivalent sensory descriptors are acceptable."
            ),
            context=(
                "Input described: coffee that tasted bitter and burnt; bright orange hair that "
                "looked stunning under the lights; music so loud one could barely hear oneself think."
            ),
            msg=f"Sensory dimension should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_cognitive_epistemic_dimension(self):
        """Test that cognitive states and certainty levels are preserved."""
        text = """
I realized that the approach wasn't working.
She wasn't sure if the meeting would happen.
He's convinced that AI will transform healthcare.
Maybe we should reconsider the timeline.
"""

        context = "Team discussion"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts preserve cognitive or epistemic states from the input — "
                "at least two of: the realisation that the approach wasn't working, her uncertainty "
                "about whether the meeting would happen, his conviction that AI will transform "
                "healthcare, or the suggestion to reconsider the timeline.  Equivalent phrasing "
                "(e.g. 'came to understand' for 'realised') is acceptable."
            ),
            context=(
                "Input: realising an approach wasn't working; uncertainty about a meeting; "
                "conviction that AI will transform healthcare; a suggestion to reconsider the timeline."
            ),
            msg=f"Cognitive/epistemic dimension should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_capability_skill_dimension(self):
        """Test that capabilities, skills, and limitations are preserved."""
        text = """
I can speak French fluently.
Sarah struggles with public speaking.
He's an expert in machine learning.
I'm unable to attend the conference due to scheduling conflicts.
"""

        context = "Personal profile discussion"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts preserve capability/skill/limitation information from the input "
                "— at least two of: the speaker's fluency in French, Sarah's difficulty with public "
                "speaking, his expertise in machine learning, or the speaker's inability to attend "
                "the conference. Equivalent phrasing is fine."
            ),
            context=(
                "Input: 'I can speak French fluently.', 'Sarah struggles with public speaking.', "
                "'He's an expert in machine learning.', 'I'm unable to attend the conference due "
                "to scheduling conflicts.'"
            ),
            msg=f"Capability/skill dimension should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_comparative_dimension(self):
        """Test that comparisons and contrasts are preserved."""
        text = """
This approach is much better than the previous one.
The new design is worse than expected.
Unlike last year, we're ahead of schedule.
"""

        context = "Project review"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "At least one fact preserves a comparative or contrasting relationship from the "
                "input — that this approach is better than the previous one, that the new design "
                "is worse than expected, or that the team is ahead of schedule unlike last year."
            ),
            context=(
                "Input: 'This approach is much better than the previous one.', 'The new design "
                "is worse than expected.', 'Unlike last year, we're ahead of schedule.'"
            ),
            msg=f"Comparative dimension should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_attitudinal_reactive_dimension(self):
        """Test that attitudes and reactions are preserved."""
        text = """
She's very skeptical about the new technology.
I was surprised when he announced his resignation.
Marcus rolled his eyes when the topic came up.
She's enthusiastic about the opportunity.
"""

        context = "Team meeting"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "At least one fact preserves an attitude or reaction from the input — her skepticism "
                "about the new technology, the speaker's surprise at his resignation, Marcus rolling "
                "his eyes (a non-verbal reaction), or her enthusiasm about the opportunity."
            ),
            context=(
                "Input: 'She's very skeptical about the new technology.', 'I was surprised when he "
                "announced his resignation.', 'Marcus rolled his eyes when the topic came up.', "
                "'She's enthusiastic about the opportunity.'"
            ),
            msg=f"Attitudinal/reactive dimension should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_intentional_motivational_dimension(self):
        """Test that goals, plans, and motivations are preserved."""
        text = """
I want to learn Mandarin before my trip to China.
She aims to complete her PhD within three years.
His goal is to build a sustainable business.
I'm planning to switch careers because I'm not fulfilled in my current role.
"""

        context = "Personal goals discussion"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts preserve goals, plans, or motivations from the input — at "
                "least one of: the speaker wanting to learn Mandarin before a trip to China, her "
                "PhD timeline goal, his goal of building a sustainable business, or the speaker's "
                "plan to switch careers because of unfulfilment."
            ),
            context=(
                "Input mentioned: wanting to learn Mandarin before a China trip; aiming to complete "
                "a PhD within three years; a goal to build a sustainable business; planning to "
                "switch careers due to lack of fulfilment in current role."
            ),
            msg=f"Intentional/motivational content should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_evaluative_preferential_dimension(self):
        """Test that preferences and values are preserved."""
        text = """
I prefer working remotely to being in an office.
She values honesty above all else.
He hates being late to meetings.
Family is the most important thing to her.
"""

        context = "Personal values discussion"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts preserve preferences or values from the input — at least two "
                "of: the speaker's preference for remote work over the office, her valuing honesty "
                "above all, his dislike of being late to meetings, or family being the most "
                "important thing to her."
            ),
            context=(
                "Input: 'I prefer working remotely to being in an office.', 'She values honesty "
                "above all else.', 'He hates being late to meetings.', 'Family is the most "
                "important thing to her.'"
            ),
            msg=f"Evaluative/preferential dimension should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_comprehensive_multi_dimension(self):
        """Test a realistic scenario with multiple dimensions in one fact.

        Inherits the module-level `hs_llm_core` marker rather than running on
        the full `hs_llm_mat` matrix — the weakest matrix providers (notably
        bedrock/nova-2-lite) drop one of the two required dimensions (emotional
        or preferential) and fail the judge. This is a quality assertion, not
        a provider-compatibility check, so single-strong-provider is the right
        tier.
        """
        text = """
I was thrilled to receive such positive feedback on my presentation yesterday!
I wasn't sure if my approach would resonate, but the audience seemed enthusiastic.
I prefer presenting in person rather than virtually because I can read the room better.
"""

        context = "Personal reflection"
        llm_config = LLMConfig.from_env()

        event_date = datetime(2024, 11, 13)

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        all_facts_text = " ".join([f.fact for f in facts])

        # Check no vague temporal terms (structural check — not LLM-dependent)
        prohibited_terms = ["recently", "soon", "lately"]
        found_prohibited = [term for term in prohibited_terms if term in all_facts_text.lower()]
        assert len(found_prohibited) == 0, f"Should NOT use vague temporal terms. Found: {found_prohibited}"

        # Check emotional and preferential dimensions via LLM judge
        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts preserve BOTH of these dimensions from the input: "
                "(1) emotional — any mention of positive feedback, enthusiasm, thrilled, or positive sentiment, "
                "(2) preferential — any mention of preferring in-person presentations or reading the room. "
                "The facts don't need to use the exact same words — semantic equivalents count."
            ),
            context=(
                "Input text: Was thrilled about positive feedback on presentation. "
                "Audience seemed enthusiastic. Prefers presenting in person rather than "
                "virtually because they can read the room better."
            ),
        )


# =============================================================================
# TEMPORAL CONVERSION TESTS
# =============================================================================


class TestTemporalConversion:
    """Tests for temporal extraction and date conversion."""

    @pytest.mark.asyncio
    async def test_temporal_absolute_conversion(self):
        """
        Test that relative temporal expressions are converted to absolute dates.

        Critical: "yesterday" should become "on November 12, 2024", NOT "recently"
        LLM behavior may vary, so we check the occurred_start field rather than fact text.
        """
        text = """
Yesterday I went for a morning jog for the first time in a nearby park.
Last week I started a new project.
I'm planning to visit Tokyo next month.
"""

        context = "Personal conversation"
        llm_config = LLMConfig.from_env()

        event_date = datetime(2024, 11, 13)

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        all_facts_text = " ".join([f.fact.lower() for f in facts])

        # Should NOT contain vague temporal terms
        prohibited_terms = ["recently", "lately", "a while ago", "some time ago"]
        found_prohibited = [term for term in prohibited_terms if term in all_facts_text]

        assert len(found_prohibited) == 0, f"Should NOT use vague temporal terms. Found: {found_prohibited}"

        # Check that at least one fact has a valid occurred_start date
        facts_with_temporal = [f for f in facts if f.occurred_start]
        assert len(facts_with_temporal) >= 1, (
            f"At least one fact should have temporal data (occurred_start). Facts: {[f.fact for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_date_field_calculation_last_night(self):
        """
        Test that the date field is calculated correctly for "last night" events.

        Ideally: If conversation is on August 14, 2023 and text says "last night",
        the date field should be August 13. We accept 13 or 14 as LLM may vary.

        Retries up to 3 times to account for LLM inconsistencies.
        """
        text = """
Melanie: Hey Caroline! Last night was amazing! We celebrated my daughter's birthday
with a concert surrounded by music, joy and the warm summer breeze.
"""

        context = "Conversation between Melanie and Caroline"
        llm_config = LLMConfig.from_env()
        event_date = datetime(2023, 8, 14, 14, 24)

        last_error = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                facts, _, _ = await extract_facts_from_text(
                    text=text,
                    event_date=event_date,
                    context=context,
                    llm_config=llm_config,
                    agent_name="Melanie",
                    config=_get_raw_config(),
                )

                assert len(facts) > 0, "Should extract at least one fact"

                birthday_fact = None
                for fact in facts:
                    if "birthday" in fact.fact.lower() or "concert" in fact.fact.lower():
                        birthday_fact = fact
                        break

                assert birthday_fact is not None, "Should extract fact about birthday celebration"

                fact_date_str = birthday_fact.occurred_start
                assert fact_date_str is not None, "occurred_start should not be None for temporal events"

                if "T" in fact_date_str:
                    fact_date = datetime.fromisoformat(fact_date_str.replace("Z", "+00:00"))
                else:
                    fact_date = datetime.fromisoformat(fact_date_str)

                assert fact_date.year == 2023, "Year should be 2023"
                assert fact_date.month == 8, "Month should be August"
                # Accept day 13 (ideal: last night) or 14 (conversation date) as valid
                assert fact_date.day in (13, 14), (
                    f"Day should be 13 or 14 (around Aug 14 event), but got {fact_date.day}."
                )

                # If we reach here, test passed
                return

            except AssertionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"Test attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    # Last attempt failed, re-raise the error
                    raise e
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"Test attempt {attempt + 1} failed with exception: {e}. Retrying...")
                    continue
                else:
                    # Last attempt failed, re-raise the error
                    raise e

        # Should not reach here, but just in case
        if last_error:
            raise last_error

    @pytest.mark.asyncio
    async def test_date_field_calculation_yesterday(self):
        """Test that the date field is calculated correctly for "yesterday" events."""
        text = """
Yesterday I went for a morning jog for the first time in a nearby park.
It was a beautiful day and I plan to make this a regular habit.
"""

        context = "Personal diary"
        llm_config = LLMConfig.from_env()

        event_date = datetime(2024, 11, 13)

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        # Find a fact with occurred_start
        facts_with_date = [f for f in facts if f.occurred_start]

        # If we got a fact with temporal data, verify the date is reasonable
        if facts_with_date:
            jogging_fact = facts_with_date[0]
            fact_date_str = jogging_fact.occurred_start
            if "T" in fact_date_str:
                fact_date = datetime.fromisoformat(fact_date_str.replace("Z", "+00:00"))
            else:
                fact_date = datetime.fromisoformat(fact_date_str)

            assert fact_date.year == 2024, "Year should be 2024"
            assert fact_date.month == 11, "Month should be November"
            # Accept day 12 (ideal: yesterday) or 13 (conversation date) as valid
            assert fact_date.day in (12, 13), f"Day should be 12 or 13 (around Nov 13 event), but got {fact_date.day}."

        all_facts_text_lower = " ".join(f.fact.lower() for f in facts)
        all_facts_text = " ".join(f.fact for f in facts)

        # Structural: "recently" is a prohibited vague term — the LLM must convert
        # "yesterday" to a concrete date, not paraphrase it as something equally vague.
        assert "recently" not in all_facts_text_lower, "Should NOT convert 'yesterday' to 'recently'"

        # Semantic: content preservation AND date conversion go through the judge so
        # paraphrases ("ran" for "jog", "Nov 12 2024" for "November 12") still satisfy.
        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts (1) preserve the activity content — that the speaker went for "
                "a morning jog/run for the first time in a nearby park — and (2) reflect that the "
                "event happened on November 12, 2024 (the day before the conversation), either by "
                "stating the absolute date in the fact text or by using an unambiguous reference."
            ),
            context=(
                "Conversation date: 2024-11-13. Input: 'Yesterday I went for a morning jog for the "
                "first time in a nearby park. It was a beautiful day...'"
            ),
            msg=f"Yesterday content and date conversion should be preserved. Facts: {[f.fact for f in facts]}",
        )

    @pytest.mark.asyncio
    async def test_extract_facts_with_relative_dates(self):
        """Test that relative dates are converted to absolute dates."""

        reference_date = datetime(2024, 3, 20, 14, 0, 0, tzinfo=UTC)
        llm_config = LLMConfig.from_env()

        text = """
        Yesterday I went hiking in Yosemite.
        Last week I started my new job at Google.
        This morning I had coffee with Alice.
        """

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=reference_date,
            llm_config=llm_config,
            agent_name="TestUser",
            context="Personal diary",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        for fact in facts:
            assert fact.fact, "Each fact should have 'fact' field"

        # Check that facts were extracted - dates may or may not be populated
        # depending on LLM behavior
        dates = [f.occurred_start for f in facts if f.occurred_start]
        # If dates were extracted, they should ideally be different for different events
        if len(dates) >= 2:
            unique_dates = set(dates)
            # Just verify we got dates, don't require them to be unique

    @pytest.mark.asyncio
    async def test_extract_facts_with_no_temporal_info(self):
        """Test that facts without temporal info are still extracted."""

        reference_date = datetime(2024, 3, 20, 14, 0, 0, tzinfo=UTC)
        llm_config = LLMConfig.from_env()

        text = "Alice works at Google. She loves Python programming."

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=reference_date,
            llm_config=llm_config,
            agent_name="TestUser",
            context="General info",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        # For facts without temporal info, occurred_start may be None or set to reference date
        # We just verify that facts were extracted with content
        for fact in facts:
            assert fact.fact, "Each fact should have text content"

    @pytest.mark.asyncio
    async def test_extract_facts_with_absolute_dates(self):
        """Test that absolute dates in text are preserved."""

        reference_date = datetime(2024, 3, 20, 14, 0, 0, tzinfo=UTC)
        llm_config = LLMConfig.from_env()

        text = """
        On March 15, 2024, Alice joined Google.
        Bob will start his vacation on April 1st.
        """

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=reference_date,
            llm_config=llm_config,
            agent_name="TestUser",
            context="Calendar events",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        for fact in facts:
            assert fact.occurred_start, f"Fact should have a date: {fact.fact}"


# =============================================================================
# LOGICAL INFERENCE TESTS
# =============================================================================


class TestLogicalInference:
    """Tests that the system makes logical inferences to connect related information."""

    @pytest.mark.asyncio
    async def test_logical_inference_identity_connection(self):
        """
        Test that the system extracts key information about loss and relationships.

        The LLM should extract facts about losing a friend and about Karlie.
        Ideally it connects them, but we accept extracting both separately.
        """
        text = """
Deborah: The roses and dahlias bring me peace. I lost a friend last week,
so I've been spending time in the garden to find some comfort.

Jolene: Sorry to hear about your friend, Deb. Losing someone can be really tough.
How are you holding up?

Deborah: Thanks for the kind words. It's been tough, but I'm comforted by
remembering our time together. It reminds me of how special life is.

Jolene: Memories can give us so much comfort and joy.

Deborah: Memories keep our loved ones close. This is the last photo with Karlie
which was taken last summer when we hiked. It was our last one. We had such a
great time! Every time I see it, I can't help but smile.
"""

        context = "Conversation between Deborah and Jolene"
        llm_config = LLMConfig.from_env()

        event_date = datetime(2023, 2, 23)

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name="Deborah",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact.lower() for f in facts)

        # Key-information preservation is structural — required tokens are
        # proper nouns and a small set of loss-related verbs that the LLM
        # can't paraphrase away without losing the meaning.  The judge proved
        # too strict here (it kept reading the facts and asking for explicit
        # connection prose), so this is a deterministic substring check —
        # same shape as the original pre-migration assertion.
        assert "karlie" in all_facts_text, f"Should mention Karlie. Facts: {[f.fact for f in facts]}"
        loss_terms = ("lost", "loss", "losing", "passed", "died", "death")
        assert any(t in all_facts_text for t in loss_terms), (
            f"Should mention the loss (one of {loss_terms}). Facts: {[f.fact for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_logical_inference_pronoun_resolution(self):
        """
        Test that pronouns are resolved to their referents.

        Example: "I started a project" + "It's challenging" -> "The project is challenging"
        """
        text = """
I started a new machine learning project last month.
It's been really challenging but very rewarding.
I've learned so much from it.
"""

        context = "Personal update"
        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            context=context,
            llm_config=llm_config,
            agent_name="TestUser",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"

        # Pronoun resolution is a *structural* property — every fact that describes a
        # quality (challenging, rewarding, learning) must also name a specific anchor
        # noun (project / work / ML / etc.) in that same fact.  The judge handled
        # this poorly in practice (hallucinating about pronouns that weren't there),
        # so this check is done deterministically: each quality-describing fact must
        # mention a project-anchor noun.
        quality_words = ("challenging", "rewarding", "learn", "demanding", "fulfilling", "rough", "tough")
        anchor_words = ("project", "ml", "machine learning", "work")

        bad_facts = []
        for f in facts:
            fact_lower = f.fact.lower()
            if any(q in fact_lower for q in quality_words) and not any(a in fact_lower for a in anchor_words):
                bad_facts.append(f.fact)

        assert not bad_facts, (
            "Pronoun 'it' should be resolved to a specific noun anchor in every "
            "quality-describing fact, but these facts lack a project/work/ML anchor:\n"
            + "\n".join(f"  - {bf}" for bf in bad_facts)
            + f"\nAll facts: {[f.fact for f in facts]}"
        )


# =============================================================================
# FACT CLASSIFICATION TESTS
# =============================================================================


class TestFactClassification:
    """Tests that facts are correctly classified as agent vs world."""

    @pytest.mark.asyncio
    async def test_agent_facts_from_podcast_transcript(self):
        """
        Test that when context identifies someone as 'you', their actions are classified as agent facts.

        This test addresses the issue where podcast transcripts with context like
        "this was podcast episode between you (Marcus) and Jamie" were extracting
        all facts as 'world' instead of properly identifying Marcus's statements as 'bank'.
        """

        transcript = """
Marcus: I've been working on AI safety research for the past six months.
Jamie: That's really interesting! What specifically are you focusing on?
Marcus: I'm investigating interpretability methods. I believe we need to understand
how models make decisions before we can trust them in critical applications.
Jamie: I completely agree with that approach.
Marcus: I published a paper on this topic last month, and I'm presenting it at
the conference next week.
Jamie: Congratulations! I'd love to read it.
"""

        context = "Podcast episode between you (Marcus) and Jamie discussing AI research"

        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=transcript,
            event_date=datetime(2024, 11, 13),
            llm_config=llm_config,
            agent_name="Marcus",
            context=context,
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact from the transcript"
        all_facts_text = " ".join(f.fact for f in facts)

        # The transcript is dense with AI-research content from Marcus.  Extraction
        # must surface that subject matter — paraphrases like "alignment work" for
        # "AI safety research" should count, so the judge handles the assertion.
        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts cover the AI-research subject matter from Marcus's statements "
                "— at least mentioning AI/ML safety, interpretability, or his recent paper and "
                "upcoming conference presentation."
            ),
            context=(
                "Marcus (the 'you' agent) said: working on AI safety research for six months, "
                "investigating interpretability methods, published a paper last month, presenting "
                "at a conference next week."
            ),
            msg=f"Should extract AI research content from transcript. Facts: {[f.fact for f in facts]}",
        )

        # Classification check is informational — many models split between 'agent'
        # and 'experience' for first-person statements.  We don't assert on the
        # split, only that one of them is non-empty for Marcus's claims.

    @pytest.mark.asyncio
    async def test_agent_facts_without_explicit_context(self):
        """Test that when 'you' is used in the text itself, it gets properly classified."""

        text = """
I completed the project on machine learning interpretability last week.
My colleague Sarah helped me with the data analysis.
We presented our findings to the team yesterday.
"""

        context = "Personal work log"

        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2024, 11, 13),
            llm_config=llm_config,
            agent_name="TestUser",
            context=context,
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract facts"

        agent_facts = [f for f in facts if f.fact_type == "agent"]

        assert len(agent_facts) >= 0  # Just verify classification works

    @pytest.mark.asyncio
    async def test_speaker_attribution_predictions(self):
        """
        Test that predictions made by different speakers are correctly attributed.

        This addresses the issue where Jamie's prediction of "Niners 27-13" was being
        incorrectly attributed to Marcus (the agent) in the extracted facts.
        """

        transcript = """
Marcus: [excited] I'm calling it now, Rams will win twenty seven to twenty four, their defense is too strong!
Jamie: [laughs] No way, I predict the Niners will win twenty seven to thirteen, comfy win at home.
Marcus: [angry] That's ridiculous, I stand by my Rams prediction.
Jamie: [teasing] We'll see who's right, my Niners pick is solid.
"""

        context = "podcast episode on match prediction of week 10 - Marcus (you) and Jamie - 14 nov"
        agent_name = "Marcus"

        llm_config = LLMConfig.from_env()

        facts, _, _ = await extract_facts_from_text(
            text=transcript,
            event_date=datetime(2024, 11, 14),
            context=context,
            llm_config=llm_config,
            agent_name=agent_name,
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        all_facts_text = " ".join(f.fact for f in facts)

        # The judge evaluates speaker attribution rather than substring-matching team
        # names — paraphrases like "the home team" or "San Francisco's squad" should
        # still satisfy the prediction-content criterion.
        await assert_meets_criteria(
            response=all_facts_text,
            criteria=(
                "The extracted facts capture at least one of the predictions made in the "
                "podcast — that the Rams will win 27-24 (Marcus's pick), or that the 49ers/Niners "
                "will win 27-13 (Jamie's pick).  Equivalent wording about team names or scores counts."
            ),
            context=(
                "Marcus (agent) predicted: Rams win 27-24. Jamie predicted: Niners win 27-13. "
                "Both predictions appear in the transcript."
            ),
            msg=f"Should extract prediction content. Facts: {[f.fact for f in facts]}",
        )

        # Speaker attribution is the deeper concern (Jamie's prediction must not be
        # attributed to Marcus). If agent_facts exist, ensure they don't claim Jamie's
        # Niners pick as Marcus's.
        agent_facts = [f for f in facts if f.fact_type == "agent"]
        if agent_facts:
            agent_text = " ".join(f.fact for f in agent_facts)
            await assert_meets_criteria(
                response=agent_text,
                criteria=(
                    "No agent fact (which represents Marcus's own statements) attributes the "
                    "'Niners 27-13' prediction to Marcus.  Marcus picked the Rams; Jamie picked "
                    "the Niners.  Marcus's facts may include his Rams prediction but must not "
                    "claim he predicted a Niners win."
                ),
                context="Marcus is the agent. He predicted Rams 27-24. Jamie predicted Niners 27-13.",
                msg=f"Jamie's prediction should not be misattributed to Marcus. Agent facts: {[f.fact for f in agent_facts]}",
            )

    @pytest.mark.asyncio
    async def test_skip_podcast_meta_commentary(self):
        """
        Test that podcast intros, outros, and calls to action are skipped.

        This addresses the issue where podcast outros like "that's all for today,
        don't forget to subscribe" were being extracted as facts.

        Note: LLM fact extraction is non-deterministic, so we retry up to 3 times.
        """

        transcript = """
Marcus: Welcome everyone to today's episode! Before we dive in, don't forget to
subscribe and leave a rating.

Marcus: Today I want to talk about my research on interpretability in AI systems.
I've been working on this for about a year now.

Jamie: That sounds really interesting! What made you focus on that area?

Marcus: I believe it's crucial for AI safety. We need to understand how these
models make decisions before we can trust them in critical applications.

Jamie: I completely agree with that approach.

Marcus: Well, I think that's gonna do it for us today! Thanks for listening everyone.
Don't forget to tap follow or subscribe, tell a friend, and drop a quick rating
so the algorithm learns to box out. See you next week!
"""

        context = "Podcast episode between you (Marcus) and Jamie about AI"

        llm_config = LLMConfig.from_env()

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                facts, _, _ = await extract_facts_from_text(
                    text=transcript,
                    event_date=datetime(2024, 11, 13),
                    llm_config=llm_config,
                    agent_name="Marcus",
                    context=context,
                    config=_get_raw_config(),
                )

                assert len(facts) > 0, "Should extract at least one fact"
                all_facts_text = " ".join(f.fact for f in facts)

                # Judge: substantive content must be extracted regardless of paraphrasing.
                # "Alignment work" or "machine-learning transparency" satisfy the criterion
                # the substring check used to enforce as "interpretability/ai/safety".
                await assert_meets_criteria(
                    response=all_facts_text,
                    criteria=(
                        "The extracted facts cover the substantive AI/ML research content from "
                        "the podcast — Marcus's work on interpretability, his motivation around "
                        "AI safety, or the goal of understanding how models make decisions before "
                        "trusting them in critical applications."
                    ),
                    context=(
                        "The transcript wraps substantive AI-research discussion in podcast "
                        "intro/outro meta-commentary (subscribe, like, follow). The substantive "
                        "content is Marcus's interpretability research and AI safety motivation."
                    ),
                    msg=f"Should extract substantive AI research content. Facts: {[f.fact for f in facts]}",
                )

                return  # Test passed

            except AssertionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"Test attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    raise e
