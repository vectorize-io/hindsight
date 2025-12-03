"""
Test query analyzer for temporal extraction.
"""
import pytest
from datetime import datetime
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer, QueryAnalysis


def test_query_analyzer_june_2024(query_analyzer):
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "june 2024"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint"
    assert analysis.temporal_constraint.start_date.year == 2024
    assert analysis.temporal_constraint.start_date.month == 6
    assert analysis.temporal_constraint.start_date.day == 1
    assert analysis.temporal_constraint.end_date.year == 2024
    assert analysis.temporal_constraint.end_date.month == 6
    assert analysis.temporal_constraint.end_date.day == 30


def test_query_analyzer_dogs_june_2023(query_analyzer):
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "dogs in June 2023"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint"
    assert analysis.temporal_constraint.start_date.year == 2023
    assert analysis.temporal_constraint.start_date.month == 6
    assert analysis.temporal_constraint.start_date.day == 1
    assert analysis.temporal_constraint.end_date.year == 2023
    assert analysis.temporal_constraint.end_date.month == 6
    assert analysis.temporal_constraint.end_date.day == 30


def test_query_analyzer_march_2023(query_analyzer):
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "March 2023"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint"
    assert analysis.temporal_constraint.start_date.year == 2023
    assert analysis.temporal_constraint.start_date.month == 3
    assert analysis.temporal_constraint.start_date.day == 1
    assert analysis.temporal_constraint.end_date.year == 2023
    assert analysis.temporal_constraint.end_date.month == 3
    assert analysis.temporal_constraint.end_date.day == 31


def test_query_analyzer_last_year(query_analyzer):
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "last year"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint"
    assert analysis.temporal_constraint.start_date.year == 2024
    assert analysis.temporal_constraint.start_date.month == 1
    assert analysis.temporal_constraint.start_date.day == 1
    assert analysis.temporal_constraint.end_date.year == 2024
    assert analysis.temporal_constraint.end_date.month == 12
    assert analysis.temporal_constraint.end_date.day == 31


def test_query_analyzer_no_temporal(query_analyzer):
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "what is the weather"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is None, "Should not extract temporal constraint"


def test_query_analyzer_activities_june_2024(query_analyzer):
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "melanie activities in june 2024"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint"
    assert analysis.temporal_constraint.start_date.year == 2024
    assert analysis.temporal_constraint.start_date.month == 6
    assert analysis.temporal_constraint.start_date.day == 1
    assert analysis.temporal_constraint.end_date.year == 2024
    assert analysis.temporal_constraint.end_date.month == 6
    assert analysis.temporal_constraint.end_date.day == 30


def test_query_analyzer_last_saturday(query_analyzer):
    """Test extraction of 'last Saturday' relative date."""
    # Reference date is Wednesday, January 15, 2025
    # Last Saturday would be January 11, 2025
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "I received a piece of jewelry last Saturday from whom?"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'last Saturday'"
    # Last Saturday from Wed Jan 15 is Sat Jan 11
    assert analysis.temporal_constraint.start_date.year == 2025
    assert analysis.temporal_constraint.start_date.month == 1
    assert analysis.temporal_constraint.start_date.day == 11
    assert analysis.temporal_constraint.end_date.year == 2025
    assert analysis.temporal_constraint.end_date.month == 1
    assert analysis.temporal_constraint.end_date.day == 11


def test_query_analyzer_yesterday(query_analyzer):
    """Test extraction of 'yesterday' relative date."""
    # Reference date is Wednesday, January 15, 2025
    # Yesterday would be January 14, 2025
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "what did I do yesterday?"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'yesterday'"
    assert analysis.temporal_constraint.start_date.year == 2025
    assert analysis.temporal_constraint.start_date.month == 1
    assert analysis.temporal_constraint.start_date.day == 14
    assert analysis.temporal_constraint.end_date.day == 14


def test_query_analyzer_last_week(query_analyzer):
    """Test extraction of 'last week' relative date."""
    # Reference date is Wednesday, January 15, 2025
    # Last week would be January 6-12, 2025 (Mon-Sun)
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "what meetings did I have last week?"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'last week'"
    assert analysis.temporal_constraint.start_date.year == 2025
    assert analysis.temporal_constraint.start_date.month == 1
    assert analysis.temporal_constraint.start_date.day == 6  # Monday
    assert analysis.temporal_constraint.end_date.day == 12  # Sunday


def test_query_analyzer_last_month(query_analyzer):
    """Test extraction of 'last month' relative date."""
    # Reference date is Wednesday, January 15, 2025
    # Last month would be December 2024
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "expenses from last month"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'last month'"
    assert analysis.temporal_constraint.start_date.year == 2024
    assert analysis.temporal_constraint.start_date.month == 12
    assert analysis.temporal_constraint.start_date.day == 1
    assert analysis.temporal_constraint.end_date.month == 12
    assert analysis.temporal_constraint.end_date.day == 31


def test_query_analyzer_last_friday(query_analyzer):
    """Test extraction of 'last Friday' relative date."""
    # Reference date is Wednesday, January 15, 2025
    # Last Friday would be January 10, 2025
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "who did I meet last Friday?"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'last Friday'"
    assert analysis.temporal_constraint.start_date.year == 2025
    assert analysis.temporal_constraint.start_date.month == 1
    assert analysis.temporal_constraint.start_date.day == 10
    assert analysis.temporal_constraint.end_date.day == 10


def test_query_analyzer_last_weekend(query_analyzer):
    """Test extraction of 'last weekend' relative date."""
    # Reference date is Wednesday, January 15, 2025
    # Last weekend would be January 11-12, 2025 (Sat-Sun)
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "what did I do last weekend?"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'last weekend'"
    assert analysis.temporal_constraint.start_date.year == 2025
    assert analysis.temporal_constraint.start_date.month == 1
    assert analysis.temporal_constraint.start_date.day == 11  # Saturday
    assert analysis.temporal_constraint.end_date.day == 12  # Sunday


def test_query_analyzer_couple_days_ago(query_analyzer):
    """Test extraction of 'a couple of days ago' colloquial expression."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "I mentioned cooking something for my friend a couple of days ago. What was it?"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'a couple of days ago'"
    # Range should be 1-3 days ago: Jan 12-14
    assert analysis.temporal_constraint.start_date.day == 12
    assert analysis.temporal_constraint.end_date.day == 14


def test_query_analyzer_few_days_ago(query_analyzer):
    """Test extraction of 'a few days ago' colloquial expression."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "What did I do a few days ago?"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'a few days ago'"
    # Range should be 2-5 days ago: Jan 10-13
    assert analysis.temporal_constraint.start_date.day == 10
    assert analysis.temporal_constraint.end_date.day == 13


def test_query_analyzer_couple_weeks_ago(query_analyzer):
    """Test extraction of 'a couple of weeks ago' colloquial expression."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    query = "a couple of weeks ago we discussed this"
    analysis = query_analyzer.analyze(query, reference_date)

    print(f"\nQuery: '{query}'")
    print(f"Reference date: {reference_date.strftime('%A, %Y-%m-%d')}")
    print(f"Analysis: {analysis}")

    assert analysis.temporal_constraint is not None, "Should extract temporal constraint for 'a couple of weeks ago'"
    # Range should be 1-3 weeks ago
    assert analysis.temporal_constraint.start_date.month == 12  # Dec 25 (3 weeks before Jan 15)
    assert analysis.temporal_constraint.end_date.month == 1  # Jan 8 (1 week before Jan 15)


