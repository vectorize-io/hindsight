"""Tests for observation trend computation and evidence-grounded models."""

from datetime import datetime, timedelta, timezone

import pytest

from hindsight_api.engine.reflect.observations import (
    CandidateObservation,
    Observation,
    ObservationEvidence,
    Trend,
    compute_trend,
    verify_evidence_quotes,
)


class TestComputeTrend:
    """Tests for the compute_trend function."""

    def test_empty_evidence_returns_stale(self):
        """No evidence should return STALE trend."""
        trend = compute_trend([])
        assert trend == Trend.STALE

    def test_all_recent_evidence_returns_new(self):
        """All evidence within recent window (30 days) should return NEW trend.

        Scenario: User just started using the app and mentioned they like coffee twice.
        Both mentions are within the last 2 weeks, so this is a NEW observation.
        """
        now = datetime.now(timezone.utc)
        evidence = [
            ObservationEvidence(
                memory_id="mem-coffee-morning",
                quote="I always start my day with a large black coffee",
                relevance="Shows preference for coffee and morning routine",
                timestamp=now - timedelta(days=5),
            ),
            ObservationEvidence(
                memory_id="mem-coffee-meeting",
                quote="grabbed coffee before the standup meeting",
                relevance="Confirms regular coffee consumption",
                timestamp=now - timedelta(days=10),
            ),
        ]

        trend = compute_trend(evidence, now=now)
        assert trend == Trend.NEW

    def test_no_recent_evidence_returns_stale(self):
        """No evidence in recent window should return STALE trend.

        Scenario: User mentioned running 3 months ago but hasn't mentioned it since.
        The observation about running as a hobby may no longer be accurate.
        """
        now = datetime.now(timezone.utc)
        evidence = [
            ObservationEvidence(
                memory_id="mem-running-march",
                quote="training for a half marathon in the spring",
                relevance="Shows interest in running",
                timestamp=now - timedelta(days=60),
            ),
            ObservationEvidence(
                memory_id="mem-running-feb",
                quote="went for a 10k run this morning",
                relevance="Active runner",
                timestamp=now - timedelta(days=100),
            ),
        ]

        trend = compute_trend(evidence, now=now)
        assert trend == Trend.STALE

    def test_stable_evidence_distribution(self):
        """Evidence spread evenly across time should return STABLE trend.

        Scenario: User has consistently mentioned working remotely over 4 months.
        Evidence is well-distributed, indicating a stable, ongoing preference.
        """
        now = datetime.now(timezone.utc)
        evidence = [
            # Recent (within 30 days)
            ObservationEvidence(
                memory_id="mem-remote-jan",
                quote="working from my home office today",
                relevance="Current remote work",
                timestamp=now - timedelta(days=5),
            ),
            ObservationEvidence(
                memory_id="mem-remote-dec",
                quote="the flexibility of remote work is great",
                relevance="Values remote work",
                timestamp=now - timedelta(days=15),
            ),
            # Middle period (30-90 days)
            ObservationEvidence(
                memory_id="mem-remote-nov",
                quote="set up a standing desk at home",
                relevance="Invested in home office",
                timestamp=now - timedelta(days=45),
            ),
            ObservationEvidence(
                memory_id="mem-remote-oct",
                quote="prefer async communication over meetings",
                relevance="Remote work style preference",
                timestamp=now - timedelta(days=60),
            ),
            # Older (90+ days)
            ObservationEvidence(
                memory_id="mem-remote-sep",
                quote="switched to fully remote last quarter",
                relevance="Original transition to remote",
                timestamp=now - timedelta(days=100),
            ),
            ObservationEvidence(
                memory_id="mem-remote-aug",
                quote="negotiated remote work in my new contract",
                relevance="Intentional choice for remote",
                timestamp=now - timedelta(days=120),
            ),
        ]

        trend = compute_trend(evidence, now=now)
        assert trend == Trend.STABLE

    def test_strengthening_trend(self):
        """Much more recent evidence than older should return STRENGTHENING trend.

        Scenario: User has been increasingly talking about learning Python recently
        after mentioning it once months ago. Interest appears to be growing.
        """
        now = datetime.now(timezone.utc)
        evidence = [
            # Lots of recent evidence - actively learning
            ObservationEvidence(
                memory_id="mem-python-project",
                quote="finished my first Python project - a web scraper",
                relevance="Completed Python project",
                timestamp=now - timedelta(days=2),
            ),
            ObservationEvidence(
                memory_id="mem-python-course",
                quote="halfway through the Python bootcamp",
                relevance="Active learning",
                timestamp=now - timedelta(days=5),
            ),
            ObservationEvidence(
                memory_id="mem-python-book",
                quote="reading Fluent Python, it's excellent",
                relevance="Deepening knowledge",
                timestamp=now - timedelta(days=10),
            ),
            ObservationEvidence(
                memory_id="mem-python-practice",
                quote="solved 50 LeetCode problems in Python",
                relevance="Practicing skills",
                timestamp=now - timedelta(days=15),
            ),
            ObservationEvidence(
                memory_id="mem-python-ide",
                quote="set up VS Code with all the Python extensions",
                relevance="Setting up environment",
                timestamp=now - timedelta(days=20),
            ),
            # Only one old mention - initial interest
            ObservationEvidence(
                memory_id="mem-python-start",
                quote="thinking about learning Python someday",
                relevance="Initial interest",
                timestamp=now - timedelta(days=100),
            ),
        ]

        trend = compute_trend(evidence, now=now)
        assert trend == Trend.STRENGTHENING

    def test_weakening_trend(self):
        """Much less recent evidence than older should return WEAKENING trend.

        Scenario: User was very active in a book club last year but mentions
        have tapered off. The observation about being a book club member
        may be becoming less relevant.
        """
        now = datetime.now(timezone.utc)
        evidence = [
            # Only one recent mention
            ObservationEvidence(
                memory_id="mem-book-recent",
                quote="haven't had time for book club lately",
                relevance="Reduced participation",
                timestamp=now - timedelta(days=10),
            ),
            # Lots of older evidence - was very active
            ObservationEvidence(
                memory_id="mem-book-aug",
                quote="hosting book club at my place next week",
                relevance="Active organizer",
                timestamp=now - timedelta(days=40),
            ),
            ObservationEvidence(
                memory_id="mem-book-july",
                quote="leading the discussion on 1984",
                relevance="Active participant",
                timestamp=now - timedelta(days=50),
            ),
            ObservationEvidence(
                memory_id="mem-book-june",
                quote="we picked The Midnight Library for June",
                relevance="Regular member",
                timestamp=now - timedelta(days=60),
            ),
            ObservationEvidence(
                memory_id="mem-book-may",
                quote="book club was amazing tonight",
                relevance="Enthusiastic member",
                timestamp=now - timedelta(days=100),
            ),
            ObservationEvidence(
                memory_id="mem-book-april",
                quote="joined a new book club in my neighborhood",
                relevance="Started participation",
                timestamp=now - timedelta(days=110),
            ),
            ObservationEvidence(
                memory_id="mem-book-march",
                quote="excited to finally join a book club",
                relevance="Initial enthusiasm",
                timestamp=now - timedelta(days=120),
            ),
        ]

        trend = compute_trend(evidence, now=now)
        assert trend == Trend.WEAKENING


class TestObservationModel:
    """Tests for the Observation model."""

    def test_observation_computed_trend(self):
        """Observation should have computed trend property based on evidence."""
        now = datetime.now(timezone.utc)
        obs = Observation(
            title="Morning meeting preference",
            content="Prefers morning meetings over afternoon ones",
            evidence=[
                ObservationEvidence(
                    memory_id="mem-morning-standup",
                    quote="I'm most productive in morning meetings",
                    relevance="Direct preference statement",
                    timestamp=now - timedelta(days=5),
                ),
            ],
            created_at=now,
        )

        assert obs.trend == Trend.NEW
        assert obs.evidence_count == 1

    def test_observation_evidence_span(self):
        """Observation should compute evidence span correctly.

        The span shows the date range of supporting evidence, helping
        understand how long this pattern has been observed.
        """
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=100)
        recent_time = now - timedelta(days=5)

        obs = Observation(
            title="Values work-life balance",
            content="Values work-life balance highly",
            evidence=[
                ObservationEvidence(
                    memory_id="mem-balance-old",
                    quote="turned down a promotion because of the hours",
                    relevance="Prioritized balance over advancement",
                    timestamp=old_time,
                ),
                ObservationEvidence(
                    memory_id="mem-balance-recent",
                    quote="always log off by 6pm no matter what",
                    relevance="Maintains boundaries",
                    timestamp=recent_time,
                ),
            ],
            created_at=now,
        )

        evidence_span = obs.evidence_span
        assert evidence_span["from"] == old_time.isoformat()
        assert evidence_span["to"] == recent_time.isoformat()

    def test_observation_empty_evidence_span(self):
        """Observation with no evidence should have null span."""
        obs = Observation(
            title="Test observation",
            content="Test observation without evidence",
            evidence=[],
        )

        evidence_span = obs.evidence_span
        assert evidence_span["from"] is None
        assert evidence_span["to"] is None


class TestVerifyEvidenceQuotes:
    """Tests for evidence quote verification.

    This ensures the LLM isn't hallucinating quotes - every quote
    must actually appear in the source memory.
    """

    def test_valid_quotes(self):
        """Should return True when quotes exist in their source memories."""
        obs = Observation(
            title="Enjoys hiking",
            content="Enjoys hiking on weekends",
            evidence=[
                ObservationEvidence(
                    memory_id="mem-hiking-trip",
                    quote="went hiking at Mount Tam",
                    relevance="Shows hiking activity",
                    timestamp=datetime.now(timezone.utc),
                ),
            ],
        )

        memories = {
            "mem-hiking-trip": "Had a great Saturday - went hiking at Mount Tam with friends and saw amazing views."
        }
        is_valid, errors = verify_evidence_quotes(obs, memories)

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_quote(self):
        """Should return False when quote doesn't exist in memory.

        This catches LLM hallucinations where it fabricates quotes.
        """
        obs = Observation(
            title="Loves spicy food",
            content="Loves spicy food",
            evidence=[
                ObservationEvidence(
                    memory_id="mem-dinner",
                    quote="I love extra hot salsa",
                    relevance="Shows spicy food preference",
                    timestamp=datetime.now(timezone.utc),
                ),
            ],
        )

        memories = {"mem-dinner": "Had tacos for dinner. The guacamole was really fresh."}
        is_valid, errors = verify_evidence_quotes(obs, memories)

        assert is_valid is False
        assert len(errors) == 1
        assert "Quote not found" in errors[0]

    def test_missing_memory(self):
        """Should return False when referenced memory doesn't exist.

        This catches cases where the LLM references a memory ID that
        was never actually retrieved.
        """
        obs = Observation(
            title="Has a dog named Max",
            content="Has a dog named Max",
            evidence=[
                ObservationEvidence(
                    memory_id="mem-pet-story",
                    quote="took Max to the vet",
                    relevance="Shows pet ownership",
                    timestamp=datetime.now(timezone.utc),
                ),
            ],
        )

        memories = {"mem-different-id": "Some unrelated memory content"}
        is_valid, errors = verify_evidence_quotes(obs, memories)

        assert is_valid is False
        assert len(errors) == 1
        assert "not found" in errors[0]


class TestCandidateObservation:
    """Tests for candidate observation model.

    Candidates are generated in the SEED phase and validated
    before becoming full observations.
    """

    def test_create_candidate(self):
        """Should create candidate with content and seed memories."""
        candidate = CandidateObservation(
            content="User prefers async communication over meetings",
            seed_memory_ids=["mem-slack-pref", "mem-meeting-decline"],
        )

        assert candidate.content == "User prefers async communication over meetings"
        assert len(candidate.seed_memory_ids) == 2
        assert "mem-slack-pref" in candidate.seed_memory_ids
