"""
Test query analyzer for temporal extraction.
"""

from datetime import datetime

import pytest


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


@pytest.mark.parametrize(
    ("query", "start", "end"),
    [
        ("今天做了什么", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("本日记录", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("今天清晨的记录", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("今天能做什么", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("今天下雨了吗", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("昨天做了什么", datetime(2025, 1, 14), datetime(2025, 1, 14)),
        ("昨天还说过什么", datetime(2025, 1, 14), datetime(2025, 1, 14)),
        ("昨天紀錄", datetime(2025, 1, 14), datetime(2025, 1, 14)),
        ("昨天傍晚发生了什么", datetime(2025, 1, 14), datetime(2025, 1, 14)),
        ("昨天說了什麼", datetime(2025, 1, 14), datetime(2025, 1, 14)),
        ("昨天帮我做了什么", datetime(2025, 1, 14), datetime(2025, 1, 14)),
        ("这周有哪些会议", datetime(2025, 1, 13), datetime(2025, 1, 19)),
        ("這週有哪些會議", datetime(2025, 1, 13), datetime(2025, 1, 19)),
        ("這週以內的記錄", datetime(2025, 1, 13), datetime(2025, 1, 19)),
        ("本周有哪些会议", datetime(2025, 1, 13), datetime(2025, 1, 19)),
        ("这个月的费用", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("這個月的費用", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("這個月期間的費用", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("这一个月的费用", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("本月的费用", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("本月中了奖", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("本月经费", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("本月資料", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("本月報告", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("本月工资", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("本月收入多少", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("月初的事", datetime(2025, 1, 1), datetime(2025, 1, 10)),
        ("上旬的记录", datetime(2025, 1, 1), datetime(2025, 1, 10)),
        ("中旬的记录", datetime(2025, 1, 11), datetime(2025, 1, 20)),
        ("月底的安排", datetime(2025, 1, 21), datetime(2025, 1, 31)),
        ("月尾的安排", datetime(2025, 1, 21), datetime(2025, 1, 31)),
        ("今年讨论过什么", datetime(2025, 1, 1), datetime(2025, 12, 31)),
        ("今年初的计划", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("当日安排", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("当天安排", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("當天記錄", datetime(2025, 1, 15), datetime(2025, 1, 15)),
        ("当年计划", datetime(2025, 1, 1), datetime(2025, 12, 31)),
        ("年初的计划", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("年底的计划", datetime(2025, 12, 1), datetime(2025, 12, 31)),
        ("年尾计划", datetime(2025, 12, 1), datetime(2025, 12, 31)),
        ("下周有哪些会议", datetime(2025, 1, 20), datetime(2025, 1, 26)),
        ("下周拜访客户", datetime(2025, 1, 20), datetime(2025, 1, 26)),
        ("下周再安排", datetime(2025, 1, 20), datetime(2025, 1, 26)),
        ("下一个星期有哪些会议", datetime(2025, 1, 20), datetime(2025, 1, 26)),
        ("上周一的会议", datetime(2025, 1, 6), datetime(2025, 1, 6)),
        ("上周星期一的会议", datetime(2025, 1, 6), datetime(2025, 1, 6)),
        ("上星期天去了哪里", datetime(2025, 1, 12), datetime(2025, 1, 12)),
        ("这周五聊了什么", datetime(2025, 1, 17), datetime(2025, 1, 17)),
        ("下周三的安排", datetime(2025, 1, 22), datetime(2025, 1, 22)),
        ("下周星期三的安排", datetime(2025, 1, 22), datetime(2025, 1, 22)),
        ("周一的会议", datetime(2025, 1, 13), datetime(2025, 1, 13)),
        ("星期天去哪", datetime(2025, 1, 19), datetime(2025, 1, 19)),
        ("礼拜五安排", datetime(2025, 1, 17), datetime(2025, 1, 17)),
        ("下周末去哪", datetime(2025, 1, 25), datetime(2025, 1, 26)),
        ("下一个周末去哪", datetime(2025, 1, 25), datetime(2025, 1, 26)),
        ("下下周有哪些会议", datetime(2025, 1, 27), datetime(2025, 2, 2)),
        ("下下周末去哪", datetime(2025, 2, 1), datetime(2025, 2, 2)),
        ("大下周有哪些会议", datetime(2025, 1, 27), datetime(2025, 2, 2)),
        ("下个月的费用", datetime(2025, 2, 1), datetime(2025, 2, 28)),
        ("下下个月的费用", datetime(2025, 3, 1), datetime(2025, 3, 31)),
        ("大下个月的费用", datetime(2025, 3, 1), datetime(2025, 3, 31)),
        ("下一个月的费用", datetime(2025, 2, 1), datetime(2025, 2, 28)),
        ("明年讨论什么", datetime(2026, 1, 1), datetime(2026, 12, 31)),
        ("下一个年度计划", datetime(2026, 1, 1), datetime(2026, 12, 31)),
        ("后年讨论什么", datetime(2027, 1, 1), datetime(2027, 12, 31)),
        ("大后年计划", datetime(2028, 1, 1), datetime(2028, 12, 31)),
        ("周末去哪", datetime(2025, 1, 18), datetime(2025, 1, 19)),
        ("这周末去哪", datetime(2025, 1, 18), datetime(2025, 1, 19)),
        ("本周末去哪", datetime(2025, 1, 18), datetime(2025, 1, 19)),
        ("上周有哪些会议", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上周代码改动", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上周又改了什么", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上周部署了什么", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上周转账记录", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上週開會說了什麼", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上週紀錄", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("當週記錄", datetime(2025, 1, 13), datetime(2025, 1, 19)),
        ("上一个星期有哪些会议", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("前一周有哪些会议", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("前一个星期有哪些会议", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上上周有哪些会议", datetime(2024, 12, 30), datetime(2025, 1, 5)),
        ("上上个星期有哪些会议", datetime(2024, 12, 30), datetime(2025, 1, 5)),
        ("大上周有哪些会议", datetime(2024, 12, 30), datetime(2025, 1, 5)),
        ("上週有哪些會議", datetime(2025, 1, 6), datetime(2025, 1, 12)),
        ("上个月的费用", datetime(2024, 12, 1), datetime(2024, 12, 31)),
        ("上个月3号的事", datetime(2024, 12, 3), datetime(2024, 12, 3)),
        ("本月5日的记录", datetime(2025, 1, 5), datetime(2025, 1, 5)),
        ("下个月10号安排", datetime(2025, 2, 10), datetime(2025, 2, 10)),
        ("當月計劃", datetime(2025, 1, 1), datetime(2025, 1, 31)),
        ("上月底的事", datetime(2024, 12, 21), datetime(2024, 12, 31)),
        ("这个月初的事", datetime(2025, 1, 1), datetime(2025, 1, 10)),
        ("上一个月的费用", datetime(2024, 12, 1), datetime(2024, 12, 31)),
        ("前一个月的费用", datetime(2024, 12, 1), datetime(2024, 12, 31)),
        ("上上个月的费用", datetime(2024, 11, 1), datetime(2024, 11, 30)),
        ("上個月的費用", datetime(2024, 12, 1), datetime(2024, 12, 31)),
        ("去年讨论过什么", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("今年曾经做过什么", datetime(2025, 1, 1), datetime(2025, 12, 31)),
        ("去年申请了什么", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("去年總結", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("去年报销记录", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("去年底的事", datetime(2024, 12, 1), datetime(2024, 12, 31)),
        ("去年年末的事", datetime(2024, 12, 1), datetime(2024, 12, 31)),
        ("本年度计划", datetime(2025, 1, 1), datetime(2025, 12, 31)),
        ("上一年度计划", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("前一年讨论过什么", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("前年讨论过什么", datetime(2023, 1, 1), datetime(2023, 12, 31)),
        ("大前年讨论过什么", datetime(2022, 1, 1), datetime(2022, 12, 31)),
        ("本季度计划", datetime(2025, 1, 1), datetime(2025, 3, 31)),
        ("今年第一季计划", datetime(2025, 1, 1), datetime(2025, 3, 31)),
        ("这一个季度计划", datetime(2025, 1, 1), datetime(2025, 3, 31)),
        ("上季度计划", datetime(2024, 10, 1), datetime(2024, 12, 31)),
        ("上一季计划", datetime(2024, 10, 1), datetime(2024, 12, 31)),
        ("上一个季度计划", datetime(2024, 10, 1), datetime(2024, 12, 31)),
        ("上一季度计划", datetime(2024, 10, 1), datetime(2024, 12, 31)),
        ("上上季度计划", datetime(2024, 7, 1), datetime(2024, 9, 30)),
        ("下季度计划", datetime(2025, 4, 1), datetime(2025, 6, 30)),
        ("下一季计划", datetime(2025, 4, 1), datetime(2025, 6, 30)),
        ("下一个季度计划", datetime(2025, 4, 1), datetime(2025, 6, 30)),
        ("下下季度计划", datetime(2025, 7, 1), datetime(2025, 9, 30)),
        ("第一季度计划", datetime(2025, 1, 1), datetime(2025, 3, 31)),
        ("去年第四季度计划", datetime(2024, 10, 1), datetime(2024, 12, 31)),
        ("上一年度第二季度计划", datetime(2024, 4, 1), datetime(2024, 6, 30)),
        ("下一年度第三季度计划", datetime(2026, 7, 1), datetime(2026, 9, 30)),
        ("2024年第二季度计划", datetime(2024, 4, 1), datetime(2024, 6, 30)),
        ("2024年第二季计划", datetime(2024, 4, 1), datetime(2024, 6, 30)),
        ("二零二四年第三季度计划", datetime(2024, 7, 1), datetime(2024, 9, 30)),
        ("2024年上半年计划", datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ("2024年下半年计划", datetime(2024, 7, 1), datetime(2024, 12, 31)),
        ("二零二四年上半年计划", datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ("去年上半年计划", datetime(2024, 1, 1), datetime(2024, 6, 30)),
        ("去年下半年计划", datetime(2024, 7, 1), datetime(2024, 12, 31)),
        ("今年下半年计划", datetime(2025, 7, 1), datetime(2025, 12, 31)),
        ("明年上半年计划", datetime(2026, 1, 1), datetime(2026, 6, 30)),
        ("上半年计划", datetime(2025, 1, 1), datetime(2025, 6, 30)),
        ("下半年计划", datetime(2025, 7, 1), datetime(2025, 12, 31)),
        ("今年六月中旬的活动", datetime(2025, 6, 11), datetime(2025, 6, 20)),
        ("2024年6月下旬的活动", datetime(2024, 6, 21), datetime(2024, 6, 30)),
        ("2024年6月底的活动", datetime(2024, 6, 21), datetime(2024, 6, 30)),
        ("六月初的活动", datetime(2024, 6, 1), datetime(2024, 6, 10)),
        ("6月底的活动", datetime(2024, 6, 21), datetime(2024, 6, 30)),
        ("前年六月的活动", datetime(2023, 6, 1), datetime(2023, 6, 30)),
        ("去年六月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("今年六月的活动", datetime(2025, 6, 1), datetime(2025, 6, 30)),
        ("明年六月的活动", datetime(2026, 6, 1), datetime(2026, 6, 30)),
        ("后年六月的活动", datetime(2027, 6, 1), datetime(2027, 6, 30)),
        ("上周末去了哪里", datetime(2025, 1, 11), datetime(2025, 1, 12)),
        ("上一个周末去了哪里", datetime(2025, 1, 11), datetime(2025, 1, 12)),
        ("上上周末去了哪里", datetime(2025, 1, 4), datetime(2025, 1, 5)),
        ("上星期末去了哪里", datetime(2025, 1, 11), datetime(2025, 1, 12)),
        ("上礼拜末去了哪里", datetime(2025, 1, 11), datetime(2025, 1, 12)),
        ("上週末去了哪裡", datetime(2025, 1, 11), datetime(2025, 1, 12)),
        ("上年讨论过什么", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("2024年6月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("2024年6月份的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("２０２４年６月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("2024年０６月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("2024年六月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("六月中了奖", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("二零二四年六月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("二Ｏ二四年六月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("二○二四年六月的活动", datetime(2024, 6, 1), datetime(2024, 6, 30)),
        ("二〇二四年十一月的活动", datetime(2024, 11, 1), datetime(2024, 11, 30)),
        ("2024年六月五日的活动", datetime(2024, 6, 5), datetime(2024, 6, 5)),
        ("2024年6月廿一日的活动", datetime(2024, 6, 21), datetime(2024, 6, 21)),
        ("二零二四年十二月卅一日的活动", datetime(2024, 12, 31), datetime(2024, 12, 31)),
        ("今年6月5日的活动", datetime(2025, 6, 5), datetime(2025, 6, 5)),
        ("六月五日的活动", datetime(2024, 6, 5), datetime(2024, 6, 5)),
        ("2024年6月5號的活动", datetime(2024, 6, 5), datetime(2024, 6, 5)),
        ("去年今天做了什么", datetime(2024, 1, 15), datetime(2024, 1, 15)),
        ("明年今日安排", datetime(2026, 1, 15), datetime(2026, 1, 15)),
        ("去年本日做了什么", datetime(2024, 1, 15), datetime(2024, 1, 15)),
        ("明年今晚安排", datetime(2026, 1, 15), datetime(2026, 1, 15)),
        ("去年昨晚吃了什么", datetime(2024, 1, 14), datetime(2024, 1, 14)),
    ],
)
def test_query_analyzer_chinese_periods(query_analyzer, query, start, end):
    """Test deterministic Chinese period extraction."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    analysis = query_analyzer.analyze(query, reference_date)

    assert analysis.temporal_constraint is not None
    assert analysis.temporal_constraint.start_date.date() == start.date()
    assert analysis.temporal_constraint.end_date.date() == end.date()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("两天前提到的菜是什么", datetime(2025, 1, 13)),
        ("明天要做什么", datetime(2025, 1, 16)),
        ("明天才开会", datetime(2025, 1, 16)),
        ("明天開會提醒我", datetime(2025, 1, 16)),
        ("明天半夜提醒我", datetime(2025, 1, 16)),
        ("明日要做什么", datetime(2025, 1, 16)),
        ("后天要做什么", datetime(2025, 1, 17)),
        ("大后天要做什么", datetime(2025, 1, 18)),
        ("大後天要做什么", datetime(2025, 1, 18)),
        ("大大后天要做什么", datetime(2025, 1, 19)),
        ("前天提到的菜是什么", datetime(2025, 1, 13)),
        ("大前天提到的菜是什么", datetime(2025, 1, 12)),
        ("大大前天提到的菜是什么", datetime(2025, 1, 11)),
        ("三天前提到的菜是什么", datetime(2025, 1, 12)),
        ("三日前的记录", datetime(2025, 1, 12)),
        ("十天前提到的菜是什么", datetime(2025, 1, 5)),
        ("十二天前提到的菜是什么", datetime(2025, 1, 3)),
        ("一百天前提到的菜是什么", datetime(2024, 10, 7)),
        ("两周前讨论了这个", datetime(2025, 1, 1)),
        ("一周前讨论了这个", datetime(2025, 1, 8)),
        ("一个星期前讨论了这个", datetime(2025, 1, 8)),
        ("兩週前討論了這個", datetime(2025, 1, 1)),
        ("两个月前的计划", datetime(2024, 11, 15)),
        ("俩月前的计划", datetime(2024, 11, 15)),
        ("倆月前的計畫", datetime(2024, 11, 15)),
        ("一个月前的计划", datetime(2024, 12, 15)),
        ("三个月前的计划", datetime(2024, 10, 15)),
        ("二十二个月前的计划", datetime(2023, 3, 15)),
        ("两年前的计划", datetime(2023, 1, 15)),
        ("三天后提醒我", datetime(2025, 1, 18)),
        ("三天之后提醒我", datetime(2025, 1, 18)),
        ("一个月以后提醒我", datetime(2025, 2, 15)),
        ("两年后提醒我", datetime(2027, 1, 15)),
        ("半个月后提醒我", datetime(2025, 1, 30)),
        ("一年半后提醒我", datetime(2026, 7, 15)),
        ("两年半以后提醒我", datetime(2027, 7, 15)),
        ("昨晚吃了什么", datetime(2025, 1, 14)),
        ("今晚安排", datetime(2025, 1, 15)),
        ("明早安排", datetime(2025, 1, 16)),
        ("半个月前的计划", datetime(2024, 12, 31)),
        ("一个半月前的计划", datetime(2024, 11, 30)),
        ("半年前的计划", datetime(2024, 7, 15)),
        ("一年半前的计划", datetime(2023, 7, 15)),
        ("两年半前的计划", datetime(2022, 7, 15)),
        ("兩個月前的計畫", datetime(2024, 11, 15)),
    ],
)
def test_query_analyzer_chinese_exact_relative_periods(query_analyzer, query, expected):
    """Test Chinese exact relative time expressions are not treated as fuzzy couple ranges."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    analysis = query_analyzer.analyze(query, reference_date)

    assert analysis.temporal_constraint is not None
    assert analysis.temporal_constraint.start_date.date() == expected.date()
    assert analysis.temporal_constraint.end_date.date() == expected.date()


@pytest.mark.parametrize(
    ("query", "start", "end"),
    [
        ("前两天提到的菜是什么", datetime(2025, 1, 12), datetime(2025, 1, 14)),
        ("三两天前提到的菜是什么", datetime(2025, 1, 12), datetime(2025, 1, 14)),
        ("几天前我做了什么", datetime(2025, 1, 10), datetime(2025, 1, 13)),
        ("前幾天我做了什麼", datetime(2025, 1, 10), datetime(2025, 1, 13)),
        ("一两周前讨论了这个", datetime(2024, 12, 25), datetime(2025, 1, 8)),
        ("两三周前讨论了这个", datetime(2024, 12, 25), datetime(2025, 1, 8)),
        ("几周前讨论了这个", datetime(2024, 12, 11), datetime(2025, 1, 1)),
        ("几个星期前讨论了这个", datetime(2024, 12, 11), datetime(2025, 1, 1)),
        ("几个礼拜前讨论了这个", datetime(2024, 12, 11), datetime(2025, 1, 1)),
        ("一两个月前的计划", datetime(2024, 10, 17), datetime(2024, 12, 16)),
        ("两三个月前的计划", datetime(2024, 10, 17), datetime(2024, 12, 16)),
        ("几个月前的计划", datetime(2024, 8, 18), datetime(2024, 11, 16)),
        ("一两年前的计划", datetime(2022, 1, 15), datetime(2024, 1, 15)),
        ("三四天前的记录", datetime(2025, 1, 11), datetime(2025, 1, 12)),
        ("四五周前讨论了什么", datetime(2024, 12, 11), datetime(2024, 12, 18)),
        ("數天前的記錄", datetime(2025, 1, 10), datetime(2025, 1, 13)),
    ],
)
def test_query_analyzer_chinese_fuzzy_periods(query_analyzer, query, start, end):
    """Test Chinese fuzzy relative period extraction mirrors English ranges."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    analysis = query_analyzer.analyze(query, reference_date)

    assert analysis.temporal_constraint is not None
    assert analysis.temporal_constraint.start_date.date() == start.date()
    assert analysis.temporal_constraint.end_date.date() == end.date()


@pytest.mark.parametrize(
    ("query", "start", "end"),
    [
        ("过去一周的记录", datetime(2025, 1, 8), datetime(2025, 1, 15)),
        ("過去一週的記錄", datetime(2025, 1, 8), datetime(2025, 1, 15)),
        ("過去一週以內的記錄", datetime(2025, 1, 8), datetime(2025, 1, 15)),
        ("今年以来的记录", datetime(2025, 1, 1), datetime(2025, 1, 15)),
        ("本周以来的记录", datetime(2025, 1, 13), datetime(2025, 1, 15)),
        ("去年至今的记录", datetime(2024, 1, 1), datetime(2025, 1, 15)),
        ("2024年以来的记录", datetime(2024, 1, 1), datetime(2025, 1, 15)),
        ("2024年6月5日以来的记录", datetime(2024, 6, 5), datetime(2025, 1, 15)),
        ("本周一以来的进展", datetime(2025, 1, 13), datetime(2025, 1, 15)),
        ("昨晚以来的记录", datetime(2025, 1, 14), datetime(2025, 1, 15)),
        ("上周末以来的记录", datetime(2025, 1, 11), datetime(2025, 1, 15)),
        ("去年今天以来的记录", datetime(2024, 1, 15), datetime(2025, 1, 15)),
        ("本季度以来的记录", datetime(2025, 1, 1), datetime(2025, 1, 15)),
        ("去年第四季度以来的记录", datetime(2024, 10, 1), datetime(2025, 1, 15)),
        ("去年底以来的记录", datetime(2024, 12, 1), datetime(2025, 1, 15)),
        ("2024年6月下旬以来的活动", datetime(2024, 6, 21), datetime(2025, 1, 15)),
        ("三天前以来的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("三天前到現在的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("三天前开始的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("三天前開始的記錄", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("2024年6月開始的記錄", datetime(2024, 6, 1), datetime(2025, 1, 15)),
        ("近三日的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("去年迄今的记录", datetime(2024, 1, 1), datetime(2025, 1, 15)),
        ("这两天的记录", datetime(2025, 1, 13), datetime(2025, 1, 15)),
        ("这几天的记录", datetime(2025, 1, 10), datetime(2025, 1, 15)),
        ("最近几天的记录", datetime(2025, 1, 10), datetime(2025, 1, 15)),
        ("最近半个月记录", datetime(2024, 12, 31), datetime(2025, 1, 15)),
        ("最近几个月的记录", datetime(2024, 8, 15), datetime(2025, 1, 15)),
        ("最近两三天的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("过去一两个月的记录", datetime(2024, 10, 15), datetime(2025, 1, 15)),
        ("这两三天的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("过去几个星期的记录", datetime(2024, 12, 11), datetime(2025, 1, 15)),
        ("近半年记录", datetime(2024, 7, 15), datetime(2025, 1, 15)),
        ("三天内的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("一週內的記錄", datetime(2025, 1, 8), datetime(2025, 1, 15)),
        ("半个月内的记录", datetime(2024, 12, 31), datetime(2025, 1, 15)),
        ("过去24小时记录", datetime(2025, 1, 14), datetime(2025, 1, 15)),
        ("過去24小時的記錄", datetime(2025, 1, 14), datetime(2025, 1, 15)),
        ("過去24鐘頭的記錄", datetime(2025, 1, 14), datetime(2025, 1, 15)),
        ("未来24小时计划", datetime(2025, 1, 15), datetime(2025, 1, 16)),
        ("前三天的记录", datetime(2025, 1, 12), datetime(2025, 1, 15)),
        ("前5天的记录", datetime(2025, 1, 10), datetime(2025, 1, 15)),
        ("前两周的记录", datetime(2025, 1, 1), datetime(2025, 1, 15)),
        ("前两个月的记录", datetime(2024, 11, 15), datetime(2025, 1, 15)),
        ("最近一个月的记录", datetime(2024, 12, 15), datetime(2025, 1, 15)),
        ("近三个月的记录", datetime(2024, 10, 15), datetime(2025, 1, 15)),
        ("过去一年做了什么", datetime(2024, 1, 15), datetime(2025, 1, 15)),
        ("未来一周的计划", datetime(2025, 1, 15), datetime(2025, 1, 22)),
        ("未來三天計劃", datetime(2025, 1, 15), datetime(2025, 1, 18)),
        ("未来几天计划", datetime(2025, 1, 15), datetime(2025, 1, 20)),
        ("未来几个月计划", datetime(2025, 1, 15), datetime(2025, 6, 15)),
        ("未来两三天计划", datetime(2025, 1, 15), datetime(2025, 1, 18)),
        ("未来半年计划", datetime(2025, 1, 15), datetime(2025, 7, 15)),
        ("接下来一个月的计划", datetime(2025, 1, 15), datetime(2025, 2, 15)),
        ("接下来一两周计划", datetime(2025, 1, 15), datetime(2025, 2, 5)),
        ("接下来几周的计划", datetime(2025, 1, 15), datetime(2025, 2, 19)),
        ("未来一年做什么", datetime(2025, 1, 15), datetime(2026, 1, 15)),
        ("两三天后提醒我", datetime(2025, 1, 17), datetime(2025, 1, 18)),
        ("三四天后提醒我", datetime(2025, 1, 18), datetime(2025, 1, 19)),
        ("几天后提醒我", datetime(2025, 1, 17), datetime(2025, 1, 20)),
        ("明后天安排", datetime(2025, 1, 16), datetime(2025, 1, 17)),
        ("明后两天安排", datetime(2025, 1, 16), datetime(2025, 1, 17)),
        ("今明两天的记录", datetime(2025, 1, 15), datetime(2025, 1, 16)),
        ("昨今两天的记录", datetime(2025, 1, 14), datetime(2025, 1, 15)),
        ("本周六和周日的安排", datetime(2025, 1, 18), datetime(2025, 1, 19)),
        ("上周六、周日做了什么", datetime(2025, 1, 11), datetime(2025, 1, 12)),
        ("周六日安排", datetime(2025, 1, 18), datetime(2025, 1, 19)),
        ("昨天到今天的记录", datetime(2025, 1, 14), datetime(2025, 1, 15)),
        ("本周一到周三的会议", datetime(2025, 1, 13), datetime(2025, 1, 15)),
        ("上周一到周三的会议", datetime(2025, 1, 6), datetime(2025, 1, 8)),
        ("上周五到周日的会议", datetime(2025, 1, 10), datetime(2025, 1, 12)),
        ("下周五到周一的安排", datetime(2025, 1, 24), datetime(2025, 1, 27)),
        ("上周五到这周一的会议", datetime(2025, 1, 10), datetime(2025, 1, 13)),
        ("上周周五到本周周一的会议", datetime(2025, 1, 10), datetime(2025, 1, 13)),
        ("本周五到下周一的会议", datetime(2025, 1, 17), datetime(2025, 1, 20)),
        ("周五到周一的安排", datetime(2025, 1, 17), datetime(2025, 1, 20)),
        ("2024年6月至8月的活动", datetime(2024, 6, 1), datetime(2024, 8, 31)),
        ("2024年6月5日到6月8日的活动", datetime(2024, 6, 5), datetime(2024, 6, 8)),
        ("2024年6月5日至8日的活动", datetime(2024, 6, 5), datetime(2024, 6, 8)),
        ("2024年6月5至8日的活动", datetime(2024, 6, 5), datetime(2024, 6, 8)),
        ("2024年6月5-8日的活动", datetime(2024, 6, 5), datetime(2024, 6, 8)),
        ("6月5日到6月8日的活动", datetime(2024, 6, 5), datetime(2024, 6, 8)),
        ("6月5到8号的活动", datetime(2024, 6, 5), datetime(2024, 6, 8)),
        ("去年到今年的记录", datetime(2024, 1, 1), datetime(2025, 12, 31)),
    ],
)
def test_query_analyzer_chinese_rolling_windows(query_analyzer, query, start, end):
    """Test Chinese rolling-window temporal expressions."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    analysis = query_analyzer.analyze(query, reference_date)

    assert analysis.temporal_constraint is not None
    assert analysis.temporal_constraint.start_date.date() == start.date()
    assert analysis.temporal_constraint.end_date.date() == end.date()


@pytest.mark.parametrize("query", ["三两天前提到的菜是什么"])
def test_query_analyzer_chinese_exact_relative_boundaries(query_analyzer, query):
    """Test malformed Chinese numerals are not truncated into exact relative rules."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    analysis = query_analyzer.analyze(query, reference_date)

    assert analysis.temporal_constraint is not None
    assert analysis.temporal_constraint.start_date.date() == datetime(2025, 1, 12).date()
    assert analysis.temporal_constraint.end_date.date() == datetime(2025, 1, 14).date()


@pytest.mark.parametrize(
    "query",
    [
        "上周杰伦的歌",
        "下周星驰电影",
        "今年糕点",
        "上个月亮很圆",
        "下个月亮很圆",
        "本月饼很好吃",
        "周末端项目",
        "明日方舟攻略",
        "明日之后攻略",
        "今日头条新闻",
        "庆余年第一季剧情",
        "后天免疫因素",
        "会议之后三天发生了什么",
        "每周末做什么",
        "每个周末做什么",
        "每年上半年计划",
        "大大大后天要做什么",
        "大大大前天提到的菜是什么",
        "2024年6月前的记录",
        "2024年6月份前的记录",
        "2024年6月5日之前的记录",
        "2024年前的记录",
        "2026年后的计划",
        "今年之前的记录",
        "上周之前的记录",
        "这个月以后的计划",
        "明天起的计划",
        "下周起的计划",
        "三天后开始的计划",
        "三天以前的记录",
        "三天之前的记录",
        "三年以前的记录",
        "每周一开会",
        "每周星期一开会",
        "每个周一开会",
        "每周一到周三开会",
        "每个星期一至星期五开会",
        "隔周一到周五排班",
    ],
)
def test_query_analyzer_chinese_compound_word_false_positives(query_analyzer, query):
    """Test Chinese compound-word prefixes do not fall through to dateparser false positives."""
    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    analysis = query_analyzer.analyze(query, reference_date)

    assert analysis.temporal_constraint is None


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


def test_query_analyzer_dateparser_crash_returns_no_constraint(query_analyzer, monkeypatch, caplog):
    """
    dateparser has been observed to crash with internal errors (e.g.,
    IndexError from locale.translate_search) on certain query inputs.
    A parser bug should not propagate up the search/consolidation pipeline —
    the analyzer should treat any failure as "no temporal constraint found".
    """
    import logging

    reference_date = datetime(2025, 1, 15, 12, 0, 0)

    # Make sure the lazy loader has run so we can monkey-patch the cached call.
    query_analyzer.load()

    def boom(*args, **kwargs):
        raise IndexError("list index out of range")

    monkeypatch.setattr(query_analyzer, "_search_dates", boom)

    # Use a query that doesn't match any of the period regex patterns so the
    # code path actually reaches the dateparser call.
    query = "tell me what happened recently with the project"

    with caplog.at_level(logging.WARNING):
        analysis = query_analyzer.analyze(query, reference_date)

    assert analysis.temporal_constraint is None, (
        "dateparser failures should be treated as no temporal constraint, not propagated"
    )
    assert any("dateparser" in rec.message for rec in caplog.records), "Should log a warning when dateparser fails"
