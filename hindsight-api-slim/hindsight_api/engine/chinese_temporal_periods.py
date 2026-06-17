"""Chinese period extraction helpers for DateparserQueryAnalyzer.

Chinese period expressions need a dedicated rule set because dateparser
frequently returns None, single-day windows for whole periods, or substring
false positives for Chinese queries.
"""

import calendar
import re
import unicodedata
from datetime import datetime, timedelta

from hindsight_api.engine.temporal_periods import NO_TEMPORAL_CONSTRAINT, DateRange, NoTemporalConstraintSentinel

# Normalize only Chinese temporal vocabulary before regex matching; keep the original query for dateparser fallback.
_CHINESE_TEMPORAL_TRANSLATION = str.maketrans(
    "週禮個兩幾後這過來現倆內間號當數開時鐘鍾頭",
    "周礼个两几后这过来现俩内间号当数开时钟钟头",
)
_CHINESE_NUMERAL_PREFIX_CHARS = "一二三四五六七八九十百千万零〇○两俩半前年0-9"
_CHINESE_NUMERAL_CHARS = "零〇○一二两俩三四五六七八九十百千万"
_CHINESE_OPTIONAL_PERIOD_MARKER = r"(?:一个|个|一)?"
_CHINESE_TEMPORAL_FOLLOWER_CHARS = frozenset(
    " \t\r\n"
    ".,!?;:()[]{}<>\"'"
    "，。！？；：（）【】《》“”‘’、"
    "的得地了过着吗呢吧呀啊嘛么和及与或至到起内里中时后前份"
    "有要去做干说聊谈讨论查找看问见给提记想开买吃喝玩用学写发订安排"
    "帮测试部署收入申请下雨代码改动工资回总复统整比哪怎什谁几多少会活事项费录信消新天计划我你他她它咱"
    "才再能还還已又曾"
    "討論計劃劃會議費記錄訊開說談寫發買訂問見給學錄"
)
_CHINESE_TEMPORAL_FOLLOWER_PREFIXES = (
    "是否",
    "是不是",
    "已经",
    "已經",
    "曾经",
    "曾經",
    "的时候",
    "期间",
    "以内",
    "以来",
    "之前",
    "左右",
    "上午",
    "下午",
    "早上",
    "晚上",
    "中午",
    "凌晨",
    "转账",
    "付款",
    "经费",
    "报销",
    "拜访",
    "支出",
    "阅读",
    "清晨",
    "傍晚",
    "黄昏",
    "夜里",
    "半夜",
    "午夜",
    "读",
    "紀錄",
    "資料",
    "報告",
    "報表",
    "日誌",
    "日記",
    "總結",
)


def _is_cjk_character(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def is_embedded_cjk_dateparser_match(query: str, matched_text: str) -> bool:
    """Return true when dateparser matched a Chinese date token inside a larger word."""
    if not any(_is_cjk_character(char) for char in matched_text):
        return False

    matches = list(re.finditer(re.escape(matched_text), query))
    if not matches:
        return False

    def is_embedded(match: re.Match[str]) -> bool:
        has_cjk_prefix = match.start() > 0 and _is_cjk_character(query[match.start() - 1])
        has_cjk_suffix = match.end() < len(query) and _is_cjk_character(query[match.end()])
        return has_cjk_prefix or has_cjk_suffix

    return all(is_embedded(match) for match in matches)


def extract_chinese_period(query: str, reference_date: datetime) -> DateRange | NoTemporalConstraintSentinel | None:
    """
    Extract Chinese period-based temporal expressions.

    These need special handling as they represent date ranges, not single dates.
    """
    query = unicodedata.normalize("NFKC", query).translate(_CHINESE_TEMPORAL_TRANSLATION)
    has_cjk_text = any(_is_cjk_character(char) for char in query)

    def constraint(start: datetime, end: datetime) -> DateRange:
        return (
            start.replace(hour=0, minute=0, second=0, microsecond=0),
            end.replace(hour=23, minute=59, second=59, microsecond=999999),
        )

    def subtract_months(months: int) -> datetime:
        month_index = reference_date.month - months - 1
        year = reference_date.year + month_index // 12
        month = month_index % 12 + 1
        day = min(reference_date.day, calendar.monthrange(year, month)[1])
        return reference_date.replace(year=year, month=month, day=day)

    def month_end(year: int, month: int) -> datetime:
        return datetime(year, month, calendar.monthrange(year, month)[1])

    def add_months(base_date: datetime, months: int) -> datetime:
        month_index = base_date.month + months - 1
        year = base_date.year + month_index // 12
        month = month_index % 12 + 1
        day = min(base_date.day, calendar.monthrange(year, month)[1])
        return base_date.replace(year=year, month=month, day=day)

    def add_years(base_date: datetime, years: int) -> datetime:
        year = base_date.year + years
        day = min(base_date.day, calendar.monthrange(year, base_date.month)[1])
        return base_date.replace(year=year, day=day)

    def has_chinese_temporal_context(match: re.Match[str]) -> bool:
        if match.end() >= len(query):
            return True
        suffix = query[match.end() :]
        if any(suffix.startswith(prefix) for prefix in _CHINESE_TEMPORAL_FOLLOWER_PREFIXES):
            return True
        return suffix[0] in _CHINESE_TEMPORAL_FOLLOWER_CHARS

    def chinese_search(pattern: str) -> re.Match[str] | None:
        if not has_cjk_text:
            return None
        for match in re.finditer(pattern, query):
            if has_chinese_temporal_context(match):
                return match
        return None

    def chinese_digit_value(char: str) -> int | None:
        if char in "零〇○":
            return 0
        if char in ("两", "俩"):
            return 2
        index = "一二三四五六七八九".find(char)
        return index + 1 if index >= 0 else None

    def parse_chinese_number(text: str) -> int | None:
        if text.isdigit():
            return int(text)
        if text == "廿":
            return 20
        if text.startswith("廿") and len(text) == 2:
            digit = chinese_digit_value(text[1])
            return 20 + digit if digit is not None else None
        if text == "卅":
            return 30
        if text.startswith("卅") and len(text) == 2:
            digit = chinese_digit_value(text[1])
            return 30 + digit if digit is not None else None
        if not any(unit in text for unit in "十百千万"):
            if len(text) != 1:
                return None
            return chinese_digit_value(text)

        total = 0
        section = 0
        number = 0
        for char in text:
            digit = chinese_digit_value(char)
            if digit is not None:
                number = digit
                continue
            if char == "万":
                section = (section + number) * 10000
                total += section
                section = 0
                number = 0
                continue
            unit = {"十": 10, "百": 100, "千": 1000}.get(char)
            if unit is None:
                return None
            section += (number or 1) * unit
            number = 0
        return total + section + number

    def parse_chinese_digit_year(year_text: str) -> int | None:
        digits = ""
        for char in year_text:
            if char in "零〇○oO":
                digits += "0"
                continue
            index = "一二三四五六七八九".find(char)
            if index < 0:
                return None
            digits += str(index + 1)
        return int(digits) if len(digits) == 4 else None

    def parse_chinese_month(month_text: str) -> int | None:
        if month_text.isdigit():
            month = int(month_text)
            return month if 1 <= month <= 12 else None
        if month_text == "十":
            return 10
        if month_text == "十一":
            return 11
        if month_text == "十二":
            return 12
        index = "一二三四五六七八九".find(month_text)
        return index + 1 if index >= 0 else None

    def parse_chinese_day(day_text: str) -> int | None:
        day = parse_chinese_number(day_text)
        return day if day is not None and 1 <= day <= 31 else None

    def parse_chinese_quarter(quarter_text: str) -> int | None:
        quarter_text = quarter_text.removeprefix("第")
        if quarter_text.isdigit():
            quarter = int(quarter_text)
            return quarter if 1 <= quarter <= 4 else None
        index = "一二三四".find(quarter_text)
        return index + 1 if index >= 0 else None

    def parse_chinese_weekday(weekday_text: str) -> int | None:
        if weekday_text in ("日", "天", "7"):
            return 6
        if weekday_text in "123456":
            return int(weekday_text) - 1
        index = "一二三四五六".find(weekday_text)
        return index if index >= 0 else None

    def quarter_period(year: int, quarter: int) -> DateRange:
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        return constraint(datetime(year, start_month, 1), month_end(year, end_month))

    def shifted_quarter_period(offset: int) -> DateRange:
        current_quarter = (reference_date.month - 1) // 3 + 1
        quarter_index = current_quarter + offset - 1
        year = reference_date.year + quarter_index // 4
        quarter = quarter_index % 4 + 1
        return quarter_period(year, quarter)

    def chinese_month_period(match: re.Match[str]) -> DateRange | None:
        year_text = match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        month_num = parse_chinese_month(match.group(2))
        if year is None or month_num is None:
            return None
        return constraint(datetime(year, month_num, 1), month_end(year, month_num))

    def relative_year_month_period(year: int, month_text: str) -> DateRange | None:
        month_num = parse_chinese_month(month_text)
        if month_num is None:
            return None
        return constraint(datetime(year, month_num, 1), month_end(year, month_num))

    def month_phase_period(year: int, month: int, phase: str) -> DateRange:
        phase = phase.removeprefix("月")
        if phase in ("初", "上旬"):
            return constraint(datetime(year, month, 1), datetime(year, month, 10))
        if phase in ("中", "中旬"):
            return constraint(datetime(year, month, 11), datetime(year, month, 20))
        return constraint(datetime(year, month, 21), month_end(year, month))

    def year_edge_period(year: int, phase: str) -> DateRange:
        phase = phase.removeprefix("年")
        if phase == "初":
            return constraint(datetime(year, 1, 1), datetime(year, 1, 31))
        return constraint(datetime(year, 12, 1), datetime(year, 12, 31))

    relative_year_pattern = (
        r"下一个年度|下一年度|下年度|下一年|大后年|明年|后年|今年|本年|"
        r"上一年度|上一个年度|上年度|上一年|去年|上年|前一年|大前年|前年"
    )
    chinese_year_pattern = r"\d{4}|[零〇○o一二三四五六七八九]{4}"
    chinese_month_pattern = r"十二|十一|十|[一二三四五六七八九]|1[0-2]|0?[1-9]"
    chinese_day_pattern = r"[0-2]?[0-9]|3[01]|卅一|卅|廿[一二三四五六七八九]?|三十一|三十|二十[一二三四五六七八九]?|十[一二三四五六七八九]?|[一二两俩三四五六七八九]"
    chinese_month_phase_pattern = r"上旬|中旬|下旬|月初|月中(?![了过])|月末|月底|月尾|初|中(?![了过])|末|底|尾"
    chinese_boundary_suffix_pattern = r"\s*(?:以?前|之前|以?后|之后)"
    chinese_month_boundary_suffix_pattern = r"份?\s*(?:以?前|之前|以?后|之后)"
    chinese_since_suffix_pattern = r"(?:以来|至今|到现在|到目前|迄今|截至目前|截止目前|截至现在|截止现在|起|开始)"
    chinese_relative_past_suffix_pattern = r"(?:以?前|之前)"
    chinese_relative_future_suffix_pattern = r"(?:以?后|之后)"
    chinese_range_separator = r"\s*(?:到|至|[-~～—])\s*"

    def relative_year_number(year_text: str) -> int:
        if year_text in ("今年", "本年"):
            return reference_date.year
        if year_text in ("明年", "下年度", "下一年度", "下一个年度", "下一年"):
            return reference_date.year + 1
        if year_text == "后年":
            return reference_date.year + 2
        if year_text == "大后年":
            return reference_date.year + 3
        if year_text in ("去年", "上年", "上一年度", "上一个年度", "上年度", "上一年", "前一年"):
            return reference_date.year - 1
        if year_text == "大前年":
            return reference_date.year - 3
        return reference_date.year - 2

    def fixed_day_offset(day_text: str) -> int:
        return {
            "大大后天": 4,
            "大后天": 3,
            "后天": 2,
            "明天": 1,
            "明日": 1,
            "今天": 0,
            "今日": 0,
            "本日": 0,
            "当日": 0,
            "当天": 0,
            "昨天": -1,
            "昨日": -1,
            "前天": -2,
            "大前天": -3,
            "大大前天": -4,
        }[day_text]

    def daypart_day_offset(daypart_text: str) -> int:
        if daypart_text in ("昨晚", "昨夜"):
            return -1
        if daypart_text in ("前晚", "前夜"):
            return -2
        if daypart_text in ("明早", "明晚", "明夜"):
            return 1
        return 0

    def relative_period_offset(period: str | None) -> int:
        if period in ("上上", "大上"):
            return -2
        if period == "上":
            return -1
        if period in ("下下", "大下"):
            return 2
        if period == "下":
            return 1
        return 0

    def relative_week_start(period: str | None) -> datetime:
        return (
            reference_date - timedelta(days=reference_date.weekday()) + timedelta(weeks=relative_period_offset(period))
        )

    def relative_weekend_period(period: str | None) -> DateRange:
        start = relative_week_start(period)
        sat = start + timedelta(days=5)
        return constraint(sat, sat + timedelta(days=1))

    def relative_month_start(period: str | None) -> datetime:
        return add_months(reference_date.replace(day=1), relative_period_offset(period))

    def exact_day_constraint(year: int, month_text: str, day_text: str) -> DateRange | None:
        d = exact_day_datetime(year, month_text, day_text)
        return None if d is None else constraint(d, d)

    def exact_day_datetime(year: int, month_text: str, day_text: str) -> datetime | None:
        month = parse_chinese_month(month_text)
        day = parse_chinese_day(day_text)
        if month is None or day is None:
            return None
        if day > calendar.monthrange(year, month)[1]:
            return None
        return datetime(year, month, day)

    def bare_month_day_constraint(month_text: str, day_text: str) -> DateRange | None:
        d = bare_month_day_datetime(month_text, day_text)
        return None if d is None else constraint(d, d)

    def bare_month_day_datetime(month_text: str, day_text: str) -> datetime | None:
        month = parse_chinese_month(month_text)
        day = parse_chinese_day(day_text)
        if month is None or day is None:
            return None
        year = reference_date.year
        if day > calendar.monthrange(year, month)[1]:
            return None
        d = datetime(year, month, day)
        if d.date() > reference_date.date():
            year -= 1
            if day > calendar.monthrange(year, month)[1]:
                return None
            d = datetime(year, month, day)
        return d

    def relative_month_day_constraint(period: str, day_text: str) -> DateRange | None:
        d = relative_month_day_datetime(period, day_text)
        return None if d is None else constraint(d, d)

    def relative_month_day_datetime(period: str, day_text: str) -> datetime | None:
        day = parse_chinese_day(day_text)
        if day is None:
            return None
        start = relative_month_start(period)
        if day > calendar.monthrange(start.year, start.month)[1]:
            return None
        return datetime(start.year, start.month, day)

    def weekday_datetime(period: str | None, weekday_text: str) -> datetime | None:
        weekday = parse_chinese_weekday(weekday_text)
        if weekday is None:
            return None
        return relative_week_start(period) + timedelta(days=weekday)

    def bare_month_phase_period(month_text: str, phase: str) -> DateRange | None:
        month = parse_chinese_month(month_text)
        if month is None:
            return None
        result = month_phase_period(reference_date.year, month, phase)
        if result[0].date() <= reference_date.date():
            return result
        return month_phase_period(reference_date.year - 1, month, phase)

    def bare_month_period(month_text: str) -> DateRange | None:
        month = parse_chinese_month(month_text)
        if month is None:
            return None
        year = reference_date.year
        start = datetime(year, month, 1)
        if start.date() > reference_date.date():
            year -= 1
            start = datetime(year, month, 1)
        return constraint(start, month_end(year, month))

    def since_constraint(start: datetime) -> DateRange | NoTemporalConstraintSentinel:
        if start.date() > reference_date.date():
            return NO_TEMPORAL_CONSTRAINT
        return constraint(start, reference_date)

    def since_from_period(
        period: DateRange | None,
    ) -> DateRange | NoTemporalConstraintSentinel | None:
        if period is None:
            return None
        return since_constraint(period[0])

    def since_from_day(day: datetime | None) -> DateRange | NoTemporalConstraintSentinel | None:
        if day is None:
            return None
        return since_constraint(day)

    def relative_offset_datetime(amount: int, unit: str, direction: int) -> datetime:
        if unit in ("天", "日"):
            return reference_date + timedelta(days=direction * amount)
        if unit in ("周", "星期", "礼拜"):
            return reference_date + timedelta(weeks=direction * amount)
        if unit == "月":
            return add_months(reference_date, direction * amount)
        return add_years(reference_date, direction * amount)

    def point_constraint_at_offset(amount: int, unit: str, direction: int) -> DateRange:
        d = relative_offset_datetime(amount, unit, direction)
        return constraint(d, d)

    def window_to_reference(amount: int, unit: str) -> DateRange:
        return constraint(relative_offset_datetime(amount, unit, -1), reference_date)

    def window_from_reference(amount: int, unit: str) -> DateRange:
        return constraint(reference_date, relative_offset_datetime(amount, unit, 1))

    # Chinese rule guide
    #
    # dateparser's Chinese coverage is uneven for period expressions: it often returns None, a single-day
    # constraint for a whole period, or a substring false positive. Keep explicit Chinese rules here so callers
    # get stable range semantics instead of falling through to dateparser.
    #
    # The matching order below is intentional:
    # - Multi-token ranges and "since" forms come before bare periods, so "去年六月以来" is not swallowed as
    #   "去年" and "下周末" is not swallowed as "下周".
    # - Open future starts such as "明天起", "下周起", and "三天后开始" return the sentinel for no temporal
    #   constraint. The API model only represents closed ranges, and inventing an end date would be misleading.
    # - Fixed day words such as "前天", "大后天", and "明天" are exact single days.
    # - Exact count offsets such as "两天前", "三个月前", and "半年后" are exact points at that offset.
    # - Fuzzy colloquial offsets such as "前两天", "几天前", "一两周前", and "两三个月后" are ranges.
    # - Rolling windows such as "过去三天", "最近几周", "这几个月", and "未来半年" are ranges anchored at
    #   reference_date.
    # - Calendar periods such as "本周", "下个月", "去年", "2024年六月", "今年上半年", "第一季度",
    #   "月初", and "年末" expand to the corresponding calendar range.
    # - Weekends are two-day calendar ranges. Bare "周末" means the nearest upcoming/current weekend for the
    #   reference week; prefixed forms such as "上周末", "这周末", and "下下周末" use the requested week.
    #
    # Chinese has no reliable whitespace word boundary, so explicit Chinese rules go through chinese_search():
    # a match is accepted at query end, or when the following text looks like temporal context. Individual
    # regexes still carry prefix/suffix guards for known compounds. The dateparser fallback has a separate
    # embedded-CJK filter; the English fallback path only uses a small false-positive token list.

    if chinese_search(r"大{3,}(前|后)天"):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(r"明后两?天"):
        return constraint(reference_date + timedelta(days=1), reference_date + timedelta(days=2))

    if chinese_search(r"今明两?天"):
        return constraint(reference_date, reference_date + timedelta(days=1))

    if chinese_search(r"昨今两?天"):
        return constraint(reference_date - timedelta(days=1), reference_date)

    day_range_match = chinese_search(
        rf"(大大后天|大后天|后天|明天|明日|今天|今日|本日|昨天|昨日|大大前天|大前天|前天)"
        rf"{chinese_range_separator}"
        r"(大大后天|大后天|后天|明天|明日|今天|今日|本日|昨天|昨日|大大前天|大前天|前天)"
    )
    if day_range_match:
        first = reference_date + timedelta(days=fixed_day_offset(day_range_match.group(1)))
        second = reference_date + timedelta(days=fixed_day_offset(day_range_match.group(2)))
        return constraint(min(first, second), max(first, second))

    if chinese_search(r"(每个?|各|隔)(周|星期|礼拜)(?:(?:周|星期|礼拜))?([一二三四五六日天1-7])"):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"(每个?|各|隔)(周|星期|礼拜)(?:(?:周|星期|礼拜))?([一二三四五六日天1-7])"
        rf"{chinese_range_separator}"
        r"(?:(周|星期|礼拜)(?:(?:周|星期|礼拜))?)?([一二三四五六日天1-7])"
    ):
        return NO_TEMPORAL_CONSTRAINT

    weekday_range_match = chinese_search(
        rf"(?<![上下大小每个各隔])(?:(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER})?"
        rf"(周|星期|礼拜)(?:(?:周|星期|礼拜))?([一二三四五六日天1-7])"
        rf"{chinese_range_separator}"
        rf"(?:(?:(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER})?"
        r"(周|星期|礼拜)(?:(?:周|星期|礼拜))?)?([一二三四五六日天1-7])"
    )
    if weekday_range_match:
        first = weekday_datetime(weekday_range_match.group(1), weekday_range_match.group(3))
        if first is None:
            return None
        second_period = weekday_range_match.group(4)
        if second_period is not None:
            second = weekday_datetime(second_period, weekday_range_match.group(6))
        else:
            end_weekday = parse_chinese_weekday(weekday_range_match.group(6))
            if end_weekday is None:
                return None
            second = relative_week_start(weekday_range_match.group(1)) + timedelta(days=end_weekday)
        if second is None:
            return None
        if second < first:
            if second_period is None:
                second += timedelta(weeks=1)
            else:
                return constraint(second, first)
        return constraint(first, second)

    absolute_month_range_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月份?"
        rf"{chinese_range_separator}"
        rf"({chinese_month_pattern})\s*月份?"
    )
    if absolute_month_range_match:
        year_text = absolute_month_range_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        start_month = parse_chinese_month(absolute_month_range_match.group(2))
        end_month = parse_chinese_month(absolute_month_range_match.group(3))
        if year is None or start_month is None or end_month is None:
            return None
        end_year = year + (1 if end_month < start_month else 0)
        return constraint(datetime(year, start_month, 1), month_end(end_year, end_month))

    relative_year_range_match = chinese_search(
        rf"({relative_year_pattern}){chinese_range_separator}({relative_year_pattern})"
    )
    if relative_year_range_match:
        first_year = relative_year_number(relative_year_range_match.group(1))
        second_year = relative_year_number(relative_year_range_match.group(2))
        return constraint(datetime(min(first_year, second_year), 1, 1), datetime(max(first_year, second_year), 12, 31))

    relative_week_range_match = chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)"
        rf"{chinese_range_separator}"
        rf"(?<![上下大小])(上上|大上|上|这|本|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)"
    )
    if relative_week_range_match:
        first = relative_week_start(relative_week_range_match.group(1))
        second = relative_week_start(relative_week_range_match.group(3))
        return constraint(min(first, second), max(first, second) + timedelta(days=6))

    relative_month_range_match = chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}月"
        rf"{chinese_range_separator}"
        rf"(?<![上下大小])(上上|大上|上|这|本|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}月"
    )
    if relative_month_range_match:
        first = relative_month_start(relative_month_range_match.group(1))
        second = relative_month_start(relative_month_range_match.group(2))
        start = min(first, second)
        end = max(first, second)
        return constraint(start, month_end(end.year, end.month))

    absolute_date_range_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月\s*({chinese_day_pattern})(?:日|号)?"
        rf"{chinese_range_separator}"
        rf"(?:(?:({chinese_year_pattern})\s*年\s*)?({chinese_month_pattern})\s*月\s*)?"
        rf"({chinese_day_pattern})(?:日|号)"
    )
    if absolute_date_range_match:
        start_year_text = absolute_date_range_match.group(1)
        start_year = int(start_year_text) if start_year_text.isdigit() else parse_chinese_digit_year(start_year_text)
        if start_year is None:
            return None
        start = exact_day_datetime(
            start_year,
            absolute_date_range_match.group(2),
            absolute_date_range_match.group(3),
        )
        end_year_text = absolute_date_range_match.group(4)
        end_year = int(end_year_text) if end_year_text and end_year_text.isdigit() else None
        if end_year_text and end_year is None:
            end_year = parse_chinese_digit_year(end_year_text)
        end_month_text = absolute_date_range_match.group(5) or absolute_date_range_match.group(2)
        if end_year is None:
            end_year = start_year
            start_month = parse_chinese_month(absolute_date_range_match.group(2))
            end_month = parse_chinese_month(end_month_text)
            if start_month is not None and end_month is not None and end_month < start_month:
                end_year += 1
        end = exact_day_datetime(end_year, end_month_text, absolute_date_range_match.group(6))
        if start is None or end is None:
            return None
        return constraint(min(start, end), max(start, end))

    bare_date_range_match = chinese_search(
        rf"(?<!年)({chinese_month_pattern})\s*月\s*({chinese_day_pattern})(?:日|号)?"
        rf"{chinese_range_separator}"
        rf"(?:(?:({chinese_month_pattern})\s*月\s*)?({chinese_day_pattern})(?:日|号))"
    )
    if bare_date_range_match:
        start = bare_month_day_datetime(bare_date_range_match.group(1), bare_date_range_match.group(2))
        if start is None:
            return None
        end_month_text = bare_date_range_match.group(3) or bare_date_range_match.group(1)
        end_month = parse_chinese_month(end_month_text)
        end_day = parse_chinese_day(bare_date_range_match.group(4))
        if end_month is None or end_day is None:
            return None
        end_year = start.year
        start_month = parse_chinese_month(bare_date_range_match.group(1))
        if start_month is not None and end_month < start_month:
            end_year += 1
        if end_day > calendar.monthrange(end_year, end_month)[1]:
            return None
        end = datetime(end_year, end_month, end_day)
        return constraint(min(start, end), max(start, end))

    absolute_date_since_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月\s*({chinese_day_pattern})"
        rf"(?:日|号){chinese_since_suffix_pattern}"
    )
    if absolute_date_since_match:
        year_text = absolute_date_since_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        if year is None:
            return None
        return since_from_day(
            exact_day_datetime(year, absolute_date_since_match.group(2), absolute_date_since_match.group(3))
        )

    relative_year_date_since_match = chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月\s*({chinese_day_pattern})"
        rf"(?:日|号){chinese_since_suffix_pattern}"
    )
    if relative_year_date_since_match:
        return since_from_day(
            exact_day_datetime(
                relative_year_number(relative_year_date_since_match.group(1)),
                relative_year_date_since_match.group(2),
                relative_year_date_since_match.group(3),
            )
        )

    relative_month_day_since_match = chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下)"
        rf"{_CHINESE_OPTIONAL_PERIOD_MARKER}月\s*({chinese_day_pattern})(?:日|号)"
        rf"{chinese_since_suffix_pattern}"
    )
    if relative_month_day_since_match:
        return since_from_day(
            relative_month_day_datetime(
                relative_month_day_since_match.group(1), relative_month_day_since_match.group(2)
            )
        )

    bare_month_day_since_match = chinese_search(
        rf"(?<!年)({chinese_month_pattern})\s*月\s*({chinese_day_pattern})(?:日|号)"
        rf"{chinese_since_suffix_pattern}"
    )
    if bare_month_day_since_match:
        return since_from_day(
            bare_month_day_datetime(bare_month_day_since_match.group(1), bare_month_day_since_match.group(2))
        )

    weekday_since_match = chinese_search(
        rf"(?<![上下大小每个各隔])(?:(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER})?"
        rf"(周|星期|礼拜)(?:(?:周|星期|礼拜))?([一二三四五六日天1-7])"
        rf"{chinese_since_suffix_pattern}"
    )
    if weekday_since_match:
        return since_from_day(weekday_datetime(weekday_since_match.group(1), weekday_since_match.group(3)))

    if chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月\s*({chinese_day_pattern})"
        rf"(?:日|号){chinese_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月\s*({chinese_day_pattern})"
        rf"(?:日|号){chinese_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下)"
        rf"{_CHINESE_OPTIONAL_PERIOD_MARKER}月\s*({chinese_day_pattern})(?:日|号)"
        rf"{chinese_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"(?<!年)({chinese_month_pattern})\s*月\s*({chinese_day_pattern})(?:日|号)"
        rf"{chinese_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"(?<![上下大小每个各隔])(?:(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER})?"
        rf"(周|星期|礼拜)(?:(?:周|星期|礼拜))?([一二三四五六日天1-7])"
        rf"{chinese_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    absolute_date_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月\s*({chinese_day_pattern})(日|号)"
    )
    if absolute_date_match:
        year_text = absolute_date_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        if year is None:
            return None
        return exact_day_constraint(year, absolute_date_match.group(2), absolute_date_match.group(3))

    relative_year_date_match = chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月\s*({chinese_day_pattern})(日|号)"
    )
    if relative_year_date_match:
        return exact_day_constraint(
            relative_year_number(relative_year_date_match.group(1)),
            relative_year_date_match.group(2),
            relative_year_date_match.group(3),
        )

    relative_month_day_match = chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下)"
        rf"{_CHINESE_OPTIONAL_PERIOD_MARKER}月\s*({chinese_day_pattern})(日|号)"
    )
    if relative_month_day_match:
        return relative_month_day_constraint(relative_month_day_match.group(1), relative_month_day_match.group(2))

    bare_month_day_match = chinese_search(rf"(?<!年)({chinese_month_pattern})\s*月\s*({chinese_day_pattern})(日|号)")
    if bare_month_day_match:
        return bare_month_day_constraint(bare_month_day_match.group(1), bare_month_day_match.group(2))

    relative_year_fixed_day_since_match = chinese_search(
        rf"({relative_year_pattern})\s*"
        rf"(大大后天|大后天|后天|明天|明日|今天|今日|本日|当日|当天|昨天|昨日|大大前天|大前天|前天)"
        rf"{chinese_since_suffix_pattern}"
    )
    if relative_year_fixed_day_since_match:
        year = relative_year_number(relative_year_fixed_day_since_match.group(1))
        base = add_years(reference_date, year - reference_date.year)
        d = base + timedelta(days=fixed_day_offset(relative_year_fixed_day_since_match.group(2)))
        return since_constraint(d)

    fixed_day_since_match = chinese_search(
        rf"(大大后天|大后天|后天|明天|明日|今天|今日|本日|当日|当天|昨天|昨日|大大前天|大前天|前天)"
        rf"{chinese_since_suffix_pattern}"
    )
    if fixed_day_since_match:
        return since_constraint(reference_date + timedelta(days=fixed_day_offset(fixed_day_since_match.group(1))))

    exact_relative_since_match = chinese_search(
        rf"([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(天|日|周|星期|礼拜|月|年)"
        rf"{chinese_relative_past_suffix_pattern}{chinese_since_suffix_pattern}"
    )
    if exact_relative_since_match:
        amount = parse_chinese_number(exact_relative_since_match.group(1))
        unit = exact_relative_since_match.group(2)
        if amount is not None:
            return since_constraint(relative_offset_datetime(amount, unit, -1))

    weekend_since_match = chinese_search(
        rf"(?<![上下大小每个各隔])"
        rf"(?:(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER})?"
        rf"(周|星期|礼拜)末{chinese_since_suffix_pattern}"
    )
    if weekend_since_match:
        return since_from_period(relative_weekend_period(weekend_since_match.group(1)))

    week_since_match = chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜){chinese_since_suffix_pattern}"
    )
    if week_since_match:
        return since_constraint(relative_week_start(week_since_match.group(1)))

    month_since_match = chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}月{chinese_since_suffix_pattern}"
    )
    if month_since_match:
        return since_constraint(relative_month_start(month_since_match.group(1)))

    absolute_year_month_since_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月{chinese_since_suffix_pattern}"
    )
    if absolute_year_month_since_match:
        year_text = absolute_year_month_since_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        month = parse_chinese_month(absolute_year_month_since_match.group(2))
        if year is None or month is None:
            return None
        return since_constraint(datetime(year, month, 1))

    relative_year_month_since_match = chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月{chinese_since_suffix_pattern}"
    )
    if relative_year_month_since_match:
        year = relative_year_number(relative_year_month_since_match.group(1))
        month = parse_chinese_month(relative_year_month_since_match.group(2))
        if month is None:
            return None
        return since_constraint(datetime(year, month, 1))

    absolute_year_since_match = chinese_search(rf"({chinese_year_pattern})\s*年{chinese_since_suffix_pattern}")
    if absolute_year_since_match:
        year_text = absolute_year_since_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        if year is None:
            return None
        return since_constraint(datetime(year, 1, 1))

    relative_year_since_match = chinese_search(rf"({relative_year_pattern}){chinese_since_suffix_pattern}")
    if relative_year_since_match:
        return since_constraint(datetime(relative_year_number(relative_year_since_match.group(1)), 1, 1))

    if chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月{chinese_month_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(rf"({chinese_year_pattern})\s*年{chinese_boundary_suffix_pattern}"):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月{chinese_month_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(rf"({relative_year_pattern}){chinese_boundary_suffix_pattern}"):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜){chinese_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}月{chinese_month_boundary_suffix_pattern}"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(天|日|周|星期|礼拜|月|年)"
        rf"{chinese_relative_future_suffix_pattern}(起|开始)"
    ):
        return NO_TEMPORAL_CONSTRAINT

    if chinese_search(
        rf"(?<![{_CHINESE_NUMERAL_PREFIX_CHARS}])([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)"
        r"个?(天|日|周|星期|礼拜|月|年)(?:以前|之前)"
    ):
        return NO_TEMPORAL_CONSTRAINT

    relative_year_daypart_since_match = chinese_search(
        rf"({relative_year_pattern})\s*(昨晚|昨夜|前晚|前夜|今晚|今早|今晨|明早|明晚|明夜)"
        rf"{chinese_since_suffix_pattern}"
    )
    if relative_year_daypart_since_match:
        year = relative_year_number(relative_year_daypart_since_match.group(1))
        base = add_years(reference_date, year - reference_date.year)
        d = base + timedelta(days=daypart_day_offset(relative_year_daypart_since_match.group(2)))
        return since_constraint(d)

    daypart_since_match = chinese_search(
        rf"(昨晚|昨夜|前晚|前夜|今晚|今早|今晨|明早|明晚|明夜){chinese_since_suffix_pattern}"
    )
    if daypart_since_match:
        d = reference_date + timedelta(days=daypart_day_offset(daypart_since_match.group(1)))
        return since_constraint(d)

    relative_year_daypart_match = chinese_search(
        rf"({relative_year_pattern})\s*(昨晚|昨夜|前晚|前夜|今晚|今早|今晨|明早|明晚|明夜)"
    )
    if relative_year_daypart_match:
        year = relative_year_number(relative_year_daypart_match.group(1))
        base = add_years(reference_date, year - reference_date.year)
        d = base + timedelta(days=daypart_day_offset(relative_year_daypart_match.group(2)))
        return constraint(d, d)

    # Day-part abbreviations still resolve only to date granularity.
    if chinese_search(r"昨晚|昨夜"):
        d = reference_date + timedelta(days=daypart_day_offset("昨晚"))
        return constraint(d, d)

    if chinese_search(r"前晚|前夜"):
        d = reference_date + timedelta(days=daypart_day_offset("前晚"))
        return constraint(d, d)

    if chinese_search(r"今晚|今早|今晨"):
        return constraint(reference_date, reference_date)

    if chinese_search(r"明早|明晚|明夜"):
        d = reference_date + timedelta(days=daypart_day_offset("明早"))
        return constraint(d, d)

    relative_year_fixed_day_match = chinese_search(
        rf"({relative_year_pattern})\s*"
        r"(大大后天|大后天|后天|明天|明日|今天|今日|本日|当日|当天|昨天|昨日|大大前天|大前天|前天)"
    )
    if relative_year_fixed_day_match:
        year = relative_year_number(relative_year_fixed_day_match.group(1))
        base = add_years(reference_date, year - reference_date.year)
        d = base + timedelta(days=fixed_day_offset(relative_year_fixed_day_match.group(2)))
        return constraint(d, d)

    if chinese_search(r"昨天|昨日"):
        d = reference_date - timedelta(days=1)
        return constraint(d, d)

    if chinese_search(r"今天|今日|本日|当日|当天"):
        return constraint(reference_date, reference_date)

    if chinese_search(r"(?<!大)大大后天"):
        d = reference_date + timedelta(days=4)
        return constraint(d, d)

    if chinese_search(r"(?<!大)大后天"):
        d = reference_date + timedelta(days=3)
        return constraint(d, d)

    if chinese_search(r"(?<![大小明])后天"):
        d = reference_date + timedelta(days=2)
        return constraint(d, d)

    if chinese_search(r"明天|明日"):
        d = reference_date + timedelta(days=1)
        return constraint(d, d)

    if chinese_search(r"(?<!大)大大前天"):
        d = reference_date - timedelta(days=4)
        return constraint(d, d)

    if chinese_search(r"(?<!大)大前天"):
        d = reference_date - timedelta(days=3)
        return constraint(d, d)

    if chinese_search(r"(?<![大小])前天"):
        d = reference_date - timedelta(days=2)
        return constraint(d, d)

    # Chinese exact-count relative times are precise. Keep them separate from fuzzier colloquialisms like
    # "前两天" and "两三天前"; malformed numerals such as "三两" deliberately fall through to fuzzy rules.
    exact_days_match = chinese_search(
        rf"(?<![{_CHINESE_NUMERAL_PREFIX_CHARS}])([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)(天|日){chinese_relative_past_suffix_pattern}"
    )
    if exact_days_match:
        days = parse_chinese_number(exact_days_match.group(1))
        if days is not None:
            return point_constraint_at_offset(days, exact_days_match.group(2), -1)

    exact_weeks_match = chinese_search(
        rf"(?<![{_CHINESE_NUMERAL_PREFIX_CHARS}])([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(周|星期|礼拜){chinese_relative_past_suffix_pattern}"
    )
    if exact_weeks_match:
        weeks = parse_chinese_number(exact_weeks_match.group(1))
        if weeks is not None:
            return point_constraint_at_offset(weeks, exact_weeks_match.group(2), -1)

    exact_months_match = chinese_search(
        rf"(?<![{_CHINESE_NUMERAL_PREFIX_CHARS}])([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?月{chinese_relative_past_suffix_pattern}"
    )
    if exact_months_match:
        months = parse_chinese_number(exact_months_match.group(1))
        if months is not None:
            return point_constraint_at_offset(months, "月", -1)

    exact_years_match = chinese_search(
        rf"(?<![{_CHINESE_NUMERAL_PREFIX_CHARS}])([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)年{chinese_relative_past_suffix_pattern}"
    )
    if exact_years_match:
        years = parse_chinese_number(exact_years_match.group(1))
        if years is not None:
            return point_constraint_at_offset(years, "年", -1)

    if chinese_search(r"一年半前"):
        d = subtract_months(18)
        return constraint(d, d)

    if chinese_search(r"([一二两三四五六七八九十]+)年半前"):
        match = chinese_search(r"([一二两三四五六七八九十]+)年半前")
        if match is not None:
            years = parse_chinese_number(match.group(1))
            if years is not None:
                d = subtract_months(years * 12 + 6)
                return constraint(d, d)

    if chinese_search(r"([0-9]+|[一二两三四五六七八九十]+)个?半月前"):
        match = chinese_search(r"([0-9]+|[一二两三四五六七八九十]+)个?半月前")
        if match is not None:
            months = parse_chinese_number(match.group(1))
            if months is not None:
                d = subtract_months(months) - timedelta(days=15)
                return constraint(d, d)

    if chinese_search(r"半个?月前"):
        d = reference_date - timedelta(days=15)
        return constraint(d, d)

    if chinese_search(r"半年前"):
        d = subtract_months(6)
        return constraint(d, d)

    future_year_half_match = chinese_search(
        rf"([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)年半{chinese_relative_future_suffix_pattern}"
    )
    if future_year_half_match:
        years = parse_chinese_number(future_year_half_match.group(1))
        if years is not None:
            d = add_months(reference_date, years * 12 + 6)
            return constraint(d, d)

    future_half_month_match = chinese_search(
        rf"([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?半月{chinese_relative_future_suffix_pattern}"
    )
    if future_half_month_match:
        months = parse_chinese_number(future_half_month_match.group(1))
        if months is not None:
            d = add_months(reference_date, months) + timedelta(days=15)
            return constraint(d, d)

    if chinese_search(rf"半个?月{chinese_relative_future_suffix_pattern}"):
        d = reference_date + timedelta(days=15)
        return constraint(d, d)

    if chinese_search(rf"半年{chinese_relative_future_suffix_pattern}"):
        d = add_months(reference_date, 6)
        return constraint(d, d)

    adjacent_fuzzy_future_match = chinese_search(
        r"(?<![一二三四五六七八九十百千万零\d后])"
        r"(一两|[两二]三|三两|三四|四五|五六|六七|七八|八九|九十)"
        rf"个?(天|日|周|星期|礼拜|月|年){chinese_relative_future_suffix_pattern}"
    )
    if adjacent_fuzzy_future_match:
        amount_text = adjacent_fuzzy_future_match.group(1)
        if amount_text == "一两":
            start_amount = 1
            end_amount = 3
        elif amount_text == "三两":
            start_amount = 1
            end_amount = 3
        else:
            start_amount = parse_chinese_number(amount_text[0])
            end_amount = parse_chinese_number(amount_text[-1])
        unit = adjacent_fuzzy_future_match.group(2)
        if start_amount is not None and end_amount is not None:
            return constraint(
                relative_offset_datetime(start_amount, unit, 1),
                relative_offset_datetime(end_amount, unit, 1),
            )

    few_future_match = chinese_search(rf"[几数]个?(天|日|周|星期|礼拜|月|年){chinese_relative_future_suffix_pattern}")
    if few_future_match:
        unit = few_future_match.group(1)
        return constraint(relative_offset_datetime(2, unit, 1), relative_offset_datetime(5, unit, 1))

    exact_future_match = chinese_search(
        rf"(?<![{_CHINESE_NUMERAL_PREFIX_CHARS}])([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(天|日|周|星期|礼拜|月|年){chinese_relative_future_suffix_pattern}"
    )
    if exact_future_match:
        amount = parse_chinese_number(exact_future_match.group(1))
        unit = exact_future_match.group(2)
        if amount is not None:
            return point_constraint_at_offset(amount, unit, 1)

    adjacent_fuzzy_past_match = chinese_search(
        r"(?<![一二三四五六七八九十百千万零\d前])"
        r"([三四五六七八九])([四五六七八九十])个?(天|日|周|星期|礼拜|月|年)前"
    )
    if adjacent_fuzzy_past_match:
        first_amount = parse_chinese_number(adjacent_fuzzy_past_match.group(1))
        second_amount = parse_chinese_number(adjacent_fuzzy_past_match.group(2))
        unit = adjacent_fuzzy_past_match.group(3)
        if first_amount is not None and second_amount is not None and second_amount == first_amount + 1:
            return constraint(
                relative_offset_datetime(second_amount, unit, -1),
                relative_offset_datetime(first_amount, unit, -1),
            )

    # Chinese fuzzy colloquialisms are imprecise ranges. dateparser usually returns a point date here.
    if chinese_search(r"前[两二](天|日)|一两(天|日)前|[两二]三(天|日)前|三两(天|日)前"):
        # "a couple of days" = approximately 2 days, give range of 1-3 days
        return constraint(reference_date - timedelta(days=3), reference_date - timedelta(days=1))

    if chinese_search(r"前[几数](天|日)|[几数](天|日)前"):
        # "a few days" = approximately 3-4 days, give range of 2-5 days
        return constraint(reference_date - timedelta(days=5), reference_date - timedelta(days=2))

    if chinese_search(r"一两个?(周|星期|礼拜)前|[两二]三(周|星期|礼拜)前|三两(周|星期|礼拜)前"):
        # "a couple of weeks" = approximately 2 weeks, give range of 1-3 weeks
        return constraint(reference_date - timedelta(weeks=3), reference_date - timedelta(weeks=1))

    if chinese_search(r"前[几数]个?(周|星期|礼拜)|[几数]个?(周|星期|礼拜)前"):
        # "a few weeks" = approximately 3-4 weeks, give range of 2-5 weeks
        return constraint(reference_date - timedelta(weeks=5), reference_date - timedelta(weeks=2))

    if chinese_search(r"一两个?月前|前一两个?月|[两二]三个月前|三两个月前"):
        # "a couple of months" = approximately 2 months, give range of 1-3 months
        return constraint(reference_date - timedelta(days=90), reference_date - timedelta(days=30))

    if chinese_search(r"前[几数]个?月|[几数]个?月前"):
        # "a few months" = approximately 3-4 months, give range of 2-5 months
        return constraint(reference_date - timedelta(days=150), reference_date - timedelta(days=60))

    if chinese_search(r"一两年前|[两二]三年前|三两年前"):
        return constraint(add_years(reference_date, -3), add_years(reference_date, -1))

    rolling_this_adjacent_match = chinese_search(
        r"这(一两|[两二]三|三两|三四|四五|五六|六七|七八|八九|九十)个?(天|日|周|星期|礼拜|月|年)"
    )
    if rolling_this_adjacent_match:
        amount_text = rolling_this_adjacent_match.group(1)
        end_amount = 3 if amount_text in ("一两", "三两") else parse_chinese_number(amount_text[-1])
        unit = rolling_this_adjacent_match.group(2)
        if end_amount is not None:
            return constraint(relative_offset_datetime(end_amount, unit, -1), reference_date)

    rolling_this_count_match = chinese_search(rf"这([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(天|日|周|星期|礼拜|月|年)")
    if rolling_this_count_match:
        amount = parse_chinese_number(rolling_this_count_match.group(1))
        unit = rolling_this_count_match.group(2)
        if amount is not None and amount > 1:
            return window_to_reference(amount, unit)

    rolling_this_few_match = chinese_search(r"这几个?(天|日|周|星期|礼拜|月|年)")
    if rolling_this_few_match:
        unit = rolling_this_few_match.group(1)
        return window_to_reference(5, unit)

    front_count_days_match = chinese_search(rf"前([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(天|日)")
    if front_count_days_match:
        amount_text = front_count_days_match.group(1)
        amount = parse_chinese_number(amount_text)
        if amount is not None and (amount > 2 or amount_text.isdigit()):
            return window_to_reference(amount, front_count_days_match.group(2))

    front_count_period_match = chinese_search(rf"前([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(周|星期|礼拜|月|年)")
    if front_count_period_match:
        amount = parse_chinese_number(front_count_period_match.group(1))
        unit = front_count_period_match.group(2)
        if amount is not None and amount > 1:
            return window_to_reference(amount, unit)

    rolling_past_adjacent_match = chinese_search(
        r"(过去|近|最近)(一两|[两二]三|三两|三四|四五|五六|六七|七八|八九|九十)"
        r"个?(天|日|周|星期|礼拜|月|年)"
    )
    if rolling_past_adjacent_match:
        amount_text = rolling_past_adjacent_match.group(2)
        end_amount = 3 if amount_text in ("一两", "三两") else parse_chinese_number(amount_text[-1])
        unit = rolling_past_adjacent_match.group(3)
        if end_amount is not None:
            return constraint(relative_offset_datetime(end_amount, unit, -1), reference_date)

    rolling_past_few_match = chinese_search(r"(过去|近|最近)几个?(天|日|周|星期|礼拜|月|年)")
    if rolling_past_few_match:
        unit = rolling_past_few_match.group(2)
        return window_to_reference(5, unit)

    rolling_past_match = chinese_search(
        rf"(过去|近|最近)([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(天|日|周|星期|礼拜|月|年)"
    )
    if rolling_past_match:
        amount = parse_chinese_number(rolling_past_match.group(2))
        unit = rolling_past_match.group(3)
        if amount is not None:
            return window_to_reference(amount, unit)

    rolling_past_half_match = chinese_search(r"(过去|近|最近)半个?(月|年)")
    if rolling_past_half_match:
        unit = rolling_past_half_match.group(2)
        if unit == "月":
            return constraint(reference_date - timedelta(days=15), reference_date)
        return constraint(subtract_months(6), reference_date)

    within_half_match = chinese_search(r"半个?(月|年)(?:以内|之内|内)")
    if within_half_match:
        unit = within_half_match.group(1)
        if unit == "月":
            return constraint(reference_date - timedelta(days=15), reference_date)
        return constraint(subtract_months(6), reference_date)

    within_count_match = chinese_search(
        rf"([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)(个?)(天|日|周|星期|礼拜|月|年)(?:以内|之内|内)"
    )
    if within_count_match:
        amount_text = within_count_match.group(1)
        measure = within_count_match.group(2)
        unit = within_count_match.group(3)
        amount = parse_chinese_number(amount_text)
        if amount is not None:
            if unit == "月" and measure != "个":
                pass
            elif unit == "年" and amount_text.isdigit() and len(amount_text) == 4:
                pass
            else:
                return window_to_reference(amount, unit)

    rolling_past_hour_match = chinese_search(rf"(过去|近|最近)([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(小时|钟头)")
    if rolling_past_hour_match:
        hours = parse_chinese_number(rolling_past_hour_match.group(2))
        if hours is not None:
            return constraint(reference_date - timedelta(hours=hours), reference_date)

    rolling_future_hour_match = chinese_search(rf"(未来|接下来|往后)([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(小时|钟头)")
    if rolling_future_hour_match:
        hours = parse_chinese_number(rolling_future_hour_match.group(2))
        if hours is not None:
            return constraint(reference_date, reference_date + timedelta(hours=hours))

    rolling_future_adjacent_match = chinese_search(
        r"(未来|接下来|往后)"
        r"(一两|[两二]三|三四|四五|五六|六七|七八|八九|九十)个?(天|日|周|星期|礼拜|月|年)"
    )
    if rolling_future_adjacent_match:
        amount_text = rolling_future_adjacent_match.group(2)
        if amount_text == "一两":
            end_amount = 3
        else:
            end_amount = parse_chinese_number(amount_text[-1])
        unit = rolling_future_adjacent_match.group(3)
        if end_amount is not None:
            return window_from_reference(end_amount, unit)

    rolling_future_few_match = chinese_search(r"(未来|接下来|往后)几个?(天|日|周|星期|礼拜|月|年)")
    if rolling_future_few_match:
        unit = rolling_future_few_match.group(2)
        return window_from_reference(5, unit)

    rolling_future_match = chinese_search(
        rf"(未来|接下来|往后)([0-9]+|[{_CHINESE_NUMERAL_CHARS}]+)个?(天|日|周|星期|礼拜|月|年)"
    )
    if rolling_future_match:
        amount = parse_chinese_number(rolling_future_match.group(2))
        unit = rolling_future_match.group(3)
        if amount is not None:
            return window_from_reference(amount, unit)

    rolling_future_half_match = chinese_search(r"(未来|接下来|往后)半个?(月|年)")
    if rolling_future_half_match:
        unit = rolling_future_half_match.group(2)
        if unit == "月":
            return constraint(reference_date, reference_date + timedelta(days=15))
        return constraint(reference_date, add_months(reference_date, 6))

    absolute_year_quarter_since_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*(第?[一二三四1-4])季(?:度)?{chinese_since_suffix_pattern}"
    )
    if absolute_year_quarter_since_match:
        year_text = absolute_year_quarter_since_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        quarter = parse_chinese_quarter(absolute_year_quarter_since_match.group(2))
        if year is None or quarter is None:
            return None
        return since_from_period(quarter_period(year, quarter))

    relative_year_quarter_since_match = chinese_search(
        rf"({relative_year_pattern})\s*(第?[一二三四1-4])季(?:度)?{chinese_since_suffix_pattern}"
    )
    if relative_year_quarter_since_match:
        quarter = parse_chinese_quarter(relative_year_quarter_since_match.group(2))
        if quarter is None:
            return None
        return since_from_period(
            quarter_period(relative_year_number(relative_year_quarter_since_match.group(1)), quarter)
        )

    if chinese_search(rf"(这|本|当){_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?{chinese_since_suffix_pattern}"):
        return since_from_period(shifted_quarter_period(0))

    if chinese_search(
        rf"(?<![上大])(上上|大上){_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?{chinese_since_suffix_pattern}"
    ):
        return since_from_period(shifted_quarter_period(-2))

    if chinese_search(
        rf"(?<![下大])(下下|大下){_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?{chinese_since_suffix_pattern}"
    ):
        return since_from_period(shifted_quarter_period(2))

    if chinese_search(
        rf"(?<![上大])上{_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?{chinese_since_suffix_pattern}|前一季(?:度)?{chinese_since_suffix_pattern}"
    ):
        return since_from_period(shifted_quarter_period(-1))

    if chinese_search(rf"(?<![下大])下{_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?{chinese_since_suffix_pattern}"):
        return since_from_period(shifted_quarter_period(1))

    bare_quarter_since_match = chinese_search(rf"第?([一二三四1-4])季度{chinese_since_suffix_pattern}")
    if bare_quarter_since_match:
        quarter = parse_chinese_quarter(bare_quarter_since_match.group(1))
        if quarter is None:
            return None
        return since_from_period(quarter_period(reference_date.year, quarter))

    absolute_year_half_since_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*(上|下)半年{chinese_since_suffix_pattern}"
    )
    if absolute_year_half_since_match:
        year_text = absolute_year_half_since_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        if year is None:
            return None
        start_month = 1 if absolute_year_half_since_match.group(2) == "上" else 7
        return since_constraint(datetime(year, start_month, 1))

    relative_year_half_since_match = chinese_search(
        rf"({relative_year_pattern})\s*(上|下)半年{chinese_since_suffix_pattern}"
    )
    if relative_year_half_since_match:
        year = relative_year_number(relative_year_half_since_match.group(1))
        start_month = 1 if relative_year_half_since_match.group(2) == "上" else 7
        return since_constraint(datetime(year, start_month, 1))

    bare_half_since_match = chinese_search(rf"(?<![年月])(上|下)半年{chinese_since_suffix_pattern}")
    if bare_half_since_match:
        start_month = 1 if bare_half_since_match.group(1) == "上" else 7
        return since_constraint(datetime(reference_date.year, start_month, 1))

    absolute_year_phase_since_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*(年初|年末|年底|年尾|初|末|底|尾){chinese_since_suffix_pattern}"
    )
    if absolute_year_phase_since_match:
        year_text = absolute_year_phase_since_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        if year is None:
            return None
        return since_from_period(year_edge_period(year, absolute_year_phase_since_match.group(2)))

    relative_year_phase_since_match = chinese_search(
        rf"({relative_year_pattern})\s*(年初|年末|年底|年尾|初|末|底|尾){chinese_since_suffix_pattern}"
    )
    if relative_year_phase_since_match:
        year = relative_year_number(relative_year_phase_since_match.group(1))
        return since_from_period(year_edge_period(year, relative_year_phase_since_match.group(2)))

    bare_year_phase_since_match = chinese_search(
        rf"(?<![本今去前后上下])(年初|年末|年底|年尾){chinese_since_suffix_pattern}"
    )
    if bare_year_phase_since_match:
        return since_from_period(year_edge_period(reference_date.year, bare_year_phase_since_match.group(1)))

    absolute_year_month_phase_since_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月份?\s*"
        rf"({chinese_month_phase_pattern}){chinese_since_suffix_pattern}"
    )
    if absolute_year_month_phase_since_match:
        year_text = absolute_year_month_phase_since_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        month = parse_chinese_month(absolute_year_month_phase_since_match.group(2))
        if year is None or month is None:
            return None
        return since_from_period(month_phase_period(year, month, absolute_year_month_phase_since_match.group(3)))

    relative_year_month_phase_since_match = chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月份?\s*"
        rf"({chinese_month_phase_pattern}){chinese_since_suffix_pattern}"
    )
    if relative_year_month_phase_since_match:
        year = relative_year_number(relative_year_month_phase_since_match.group(1))
        month = parse_chinese_month(relative_year_month_phase_since_match.group(2))
        if month is None:
            return None
        return since_from_period(month_phase_period(year, month, relative_year_month_phase_since_match.group(3)))

    current_month_phase_since_match = chinese_search(
        rf"(这|本|当){_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*"
        rf"({chinese_month_phase_pattern}){chinese_since_suffix_pattern}"
    )
    if current_month_phase_since_match:
        return since_from_period(
            month_phase_period(reference_date.year, reference_date.month, current_month_phase_since_match.group(2))
        )

    next_month_phase_since_match = chinese_search(
        rf"(?<![下大])下{_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*"
        rf"({chinese_month_phase_pattern}){chinese_since_suffix_pattern}"
    )
    if next_month_phase_since_match:
        start = add_months(reference_date.replace(day=1), 1)
        return since_from_period(month_phase_period(start.year, start.month, next_month_phase_since_match.group(1)))

    second_next_month_phase_since_match = chinese_search(
        rf"(?<![下大])(下下|大下){_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*"
        rf"({chinese_month_phase_pattern}){chinese_since_suffix_pattern}"
    )
    if second_next_month_phase_since_match:
        start = add_months(reference_date.replace(day=1), 2)
        return since_from_period(
            month_phase_period(start.year, start.month, second_next_month_phase_since_match.group(2))
        )

    previous_month_phase_since_match = chinese_search(
        rf"(?<![上大])上{_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*"
        rf"({chinese_month_phase_pattern}){chinese_since_suffix_pattern}"
    )
    if previous_month_phase_since_match:
        first = reference_date.replace(day=1)
        start = (first - timedelta(days=1)).replace(day=1)
        return since_from_period(month_phase_period(start.year, start.month, previous_month_phase_since_match.group(1)))

    second_previous_month_phase_since_match = chinese_search(
        rf"(?<![上大])(上上|大上){_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*"
        rf"({chinese_month_phase_pattern}){chinese_since_suffix_pattern}"
    )
    if second_previous_month_phase_since_match:
        start = subtract_months(2).replace(day=1)
        return since_from_period(
            month_phase_period(start.year, start.month, second_previous_month_phase_since_match.group(2))
        )

    bare_specific_month_phase_since_match = chinese_search(
        rf"(?<!年)({chinese_month_pattern})\s*月份?\s*({chinese_month_phase_pattern})"
        rf"{chinese_since_suffix_pattern}"
    )
    if bare_specific_month_phase_since_match:
        return since_from_period(
            bare_month_phase_period(
                bare_specific_month_phase_since_match.group(1),
                bare_specific_month_phase_since_match.group(2),
            )
        )

    bare_month_phase_since_match = chinese_search(
        rf"(?<![年月])(上旬|中旬|下旬|月初|月中(?![了过])|月末|月底|月尾){chinese_since_suffix_pattern}"
    )
    if bare_month_phase_since_match:
        return since_from_period(
            month_phase_period(reference_date.year, reference_date.month, bare_month_phase_since_match.group(1))
        )

    absolute_year_quarter_match = chinese_search(rf"({chinese_year_pattern})\s*年\s*(第?[一二三四1-4])季(?:度)?")
    if absolute_year_quarter_match:
        year_text = absolute_year_quarter_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        quarter = parse_chinese_quarter(absolute_year_quarter_match.group(2))
        if year is None or quarter is None:
            return None
        return quarter_period(year, quarter)

    relative_year_quarter_match = chinese_search(rf"({relative_year_pattern})\s*(第?[一二三四1-4])季(?:度)?")
    if relative_year_quarter_match:
        quarter = parse_chinese_quarter(relative_year_quarter_match.group(2))
        if quarter is None:
            return None
        return quarter_period(relative_year_number(relative_year_quarter_match.group(1)), quarter)

    if chinese_search(rf"(这|本|当){_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?"):
        return shifted_quarter_period(0)

    if chinese_search(rf"(?<![上大])(上上|大上){_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?"):
        return shifted_quarter_period(-2)

    if chinese_search(rf"(?<![下大])(下下|大下){_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?"):
        return shifted_quarter_period(2)

    if chinese_search(rf"(?<![上大])上{_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?|前一季(?:度)?"):
        return shifted_quarter_period(-1)

    if chinese_search(rf"(?<![下大])下{_CHINESE_OPTIONAL_PERIOD_MARKER}季(?:度)?"):
        return shifted_quarter_period(1)

    bare_quarter_match = chinese_search(r"第?([一二三四1-4])季度")
    if bare_quarter_match:
        quarter = parse_chinese_quarter(bare_quarter_match.group(1))
        if quarter is None:
            return None
        return quarter_period(reference_date.year, quarter)

    absolute_year_half_match = chinese_search(rf"({chinese_year_pattern})\s*年\s*(上|下)半年")
    if absolute_year_half_match:
        year_text = absolute_year_half_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        if year is None:
            return None
        if absolute_year_half_match.group(2) == "上":
            return constraint(datetime(year, 1, 1), datetime(year, 6, 30))
        return constraint(datetime(year, 7, 1), datetime(year, 12, 31))

    relative_year_half_match = chinese_search(rf"({relative_year_pattern})\s*(上|下)半年")
    if relative_year_half_match:
        year = relative_year_number(relative_year_half_match.group(1))
        if relative_year_half_match.group(2) == "上":
            return constraint(datetime(year, 1, 1), datetime(year, 6, 30))
        return constraint(datetime(year, 7, 1), datetime(year, 12, 31))

    absolute_year_phase_match = chinese_search(rf"({chinese_year_pattern})\s*年\s*(年初|年末|年底|年尾|初|末|底|尾)")
    if absolute_year_phase_match:
        year_text = absolute_year_phase_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        if year is None:
            return None
        return year_edge_period(year, absolute_year_phase_match.group(2))

    relative_year_phase_match = chinese_search(rf"({relative_year_pattern})\s*(年初|年末|年底|年尾|初|末|底|尾)")
    if relative_year_phase_match:
        year = relative_year_number(relative_year_phase_match.group(1))
        return year_edge_period(year, relative_year_phase_match.group(2))

    absolute_year_month_phase_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月份?\s*({chinese_month_phase_pattern})"
    )
    if absolute_year_month_phase_match:
        year_text = absolute_year_month_phase_match.group(1)
        year = int(year_text) if year_text.isdigit() else parse_chinese_digit_year(year_text)
        month = parse_chinese_month(absolute_year_month_phase_match.group(2))
        if year is None or month is None:
            return None
        return month_phase_period(year, month, absolute_year_month_phase_match.group(3))

    relative_year_month_phase_match = chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月份?\s*({chinese_month_phase_pattern})"
    )
    if relative_year_month_phase_match:
        year = relative_year_number(relative_year_month_phase_match.group(1))
        month = parse_chinese_month(relative_year_month_phase_match.group(2))
        if month is None:
            return None
        return month_phase_period(year, month, relative_year_month_phase_match.group(3))

    current_month_phase_match = chinese_search(
        rf"(这|本|当){_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*({chinese_month_phase_pattern})"
    )
    if current_month_phase_match:
        return month_phase_period(reference_date.year, reference_date.month, current_month_phase_match.group(2))

    next_month_phase_match = chinese_search(
        rf"(?<![下大])下{_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*({chinese_month_phase_pattern})"
    )
    if next_month_phase_match:
        start = add_months(reference_date.replace(day=1), 1)
        return month_phase_period(start.year, start.month, next_month_phase_match.group(1))

    second_next_month_phase_match = chinese_search(
        rf"(?<![下大])(下下|大下){_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*({chinese_month_phase_pattern})"
    )
    if second_next_month_phase_match:
        start = add_months(reference_date.replace(day=1), 2)
        return month_phase_period(start.year, start.month, second_next_month_phase_match.group(2))

    previous_month_phase_match = chinese_search(
        rf"(?<![上大])上{_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*({chinese_month_phase_pattern})"
    )
    if previous_month_phase_match:
        first = reference_date.replace(day=1)
        start = (first - timedelta(days=1)).replace(day=1)
        return month_phase_period(start.year, start.month, previous_month_phase_match.group(1))

    second_previous_month_phase_match = chinese_search(
        rf"(?<![上大])(上上|大上){_CHINESE_OPTIONAL_PERIOD_MARKER}月份?\s*({chinese_month_phase_pattern})"
    )
    if second_previous_month_phase_match:
        start = subtract_months(2).replace(day=1)
        return month_phase_period(start.year, start.month, second_previous_month_phase_match.group(2))

    bare_specific_month_phase_match = chinese_search(
        rf"(?<!年)({chinese_month_pattern})\s*月份?\s*({chinese_month_phase_pattern})"
    )
    if bare_specific_month_phase_match:
        return bare_month_phase_period(
            bare_specific_month_phase_match.group(1),
            bare_specific_month_phase_match.group(2),
        )

    bare_month_phase_match = chinese_search(r"(?<![年月])(上旬|中旬|下旬|月初|月中(?![了过])|月末|月底|月尾)")
    if bare_month_phase_match:
        return month_phase_period(reference_date.year, reference_date.month, bare_month_phase_match.group(1))

    bare_year_phase_match = chinese_search(
        r"(?<![本今去前后上下])年初|(?<![本今去前后上下])年末|(?<![本今去前后上下])年底|(?<![本今去前后上下])年尾"
    )
    if bare_year_phase_match:
        return year_edge_period(reference_date.year, bare_year_phase_match.group(0))

    # More specific relative year+month and weekend forms must precede broad year/week rules.
    relative_year_month_match = chinese_search(
        rf"({relative_year_pattern})\s*({chinese_month_pattern})\s*月(?!{chinese_month_boundary_suffix_pattern})"
    )
    if relative_year_month_match:
        year = relative_year_number(relative_year_month_match.group(1))
        return relative_year_month_period(year, relative_year_month_match.group(2))

    weekend_pair_match = chinese_search(
        rf"(?<![上下大小每个各隔])"
        rf"(?:(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER})?"
        rf"(周|星期|礼拜)(?:(?:周|星期|礼拜))?[六6]"
        r"(?:\s*(?:和|及|与|、)\s*(?:(?:周|星期|礼拜)(?:(?:周|星期|礼拜))?)?[日天7]|[日天7])"
    )
    if weekend_pair_match:
        return relative_weekend_period(weekend_pair_match.group(1))

    weekday_match = chinese_search(
        rf"(?<![上下大小])(上上|大上|上|这|本|当|下下|大下|下){_CHINESE_OPTIONAL_PERIOD_MARKER}"
        rf"(周|星期|礼拜)(?:(?:周|星期|礼拜))?([一二三四五六日天1-7])"
    )
    if weekday_match:
        d = weekday_datetime(weekday_match.group(1), weekday_match.group(3))
        if d is None:
            return None
        return constraint(d, d)

    bare_weekday_match = chinese_search(
        r"(?<![上下大小每个各隔])(周|星期|礼拜)(?:(?:周|星期|礼拜))?([一二三四五六日天1-7])"
    )
    if bare_weekday_match:
        weekday = parse_chinese_weekday(bare_weekday_match.group(2))
        if weekday is None:
            return None
        start = reference_date - timedelta(days=reference_date.weekday())
        d = start + timedelta(days=weekday)
        return constraint(d, d)

    if chinese_search(rf"(这|本|当){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)末"):
        sat = reference_date - timedelta(days=reference_date.weekday()) + timedelta(days=5)
        return constraint(sat, sat + timedelta(days=1))

    if chinese_search(rf"(下下|大下){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)末"):
        sat = reference_date - timedelta(days=reference_date.weekday()) + timedelta(days=19)
        return constraint(sat, sat + timedelta(days=1))

    if chinese_search(rf"下{_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)末"):
        sat = reference_date - timedelta(days=reference_date.weekday()) + timedelta(days=12)
        return constraint(sat, sat + timedelta(days=1))

    if chinese_search(rf"(上上|大上){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)末"):
        start = reference_date - timedelta(days=reference_date.weekday() + 14)
        sat = start + timedelta(days=5)
        return constraint(sat, sat + timedelta(days=1))

    if chinese_search(rf"上{_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)末"):
        days_since_sat = (reference_date.weekday() + 2) % 7
        if days_since_sat == 0:
            days_since_sat = 7
        sat = reference_date - timedelta(days=days_since_sat)
        return constraint(sat, sat + timedelta(days=1))

    if chinese_search(r"(?<!每)(?<!个)(?<!各)(?<!隔)周末"):
        sat = reference_date - timedelta(days=reference_date.weekday()) + timedelta(days=5)
        return constraint(sat, sat + timedelta(days=1))

    # Current and future period patterns. dateparser treats these Chinese period terms inconsistently, often as
    # no match or as a single day, so keep them in the explicit range extractor.
    if chinese_search(
        rf"(这|本|当){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)(?!末)(?!{chinese_boundary_suffix_pattern})"
    ):
        start = reference_date - timedelta(days=reference_date.weekday())
        return constraint(start, start + timedelta(days=6))

    if chinese_search(rf"(这|本|当){_CHINESE_OPTIONAL_PERIOD_MARKER}月(?!{chinese_month_boundary_suffix_pattern})"):
        start = reference_date.replace(day=1)
        return constraint(start, month_end(reference_date.year, reference_date.month))

    if chinese_search(rf"(本年度|今年|本年|当年)(?!{chinese_boundary_suffix_pattern})"):
        return constraint(datetime(reference_date.year, 1, 1), datetime(reference_date.year, 12, 31))

    if chinese_search(
        rf"(?<![下大])(下下|大下){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)(?!末)(?!{chinese_boundary_suffix_pattern})"
    ):
        start = reference_date - timedelta(days=reference_date.weekday()) + timedelta(days=14)
        return constraint(start, start + timedelta(days=6))

    if chinese_search(
        rf"(?<![下大])下{_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)(?!末)(?!{chinese_boundary_suffix_pattern})"
    ):
        start = reference_date - timedelta(days=reference_date.weekday()) + timedelta(days=7)
        return constraint(start, start + timedelta(days=6))

    if chinese_search(
        rf"(?<![下大])(下下|大下){_CHINESE_OPTIONAL_PERIOD_MARKER}月(?!{chinese_month_boundary_suffix_pattern})"
    ):
        start = add_months(reference_date.replace(day=1), 2)
        return constraint(start, month_end(start.year, start.month))

    if chinese_search(rf"(?<![下大])下{_CHINESE_OPTIONAL_PERIOD_MARKER}月(?!{chinese_month_boundary_suffix_pattern})"):
        start = add_months(reference_date.replace(day=1), 1)
        return constraint(start, month_end(start.year, start.month))

    if chinese_search(rf"(下一个年度|下一年度|下年度|下一年|明年)(?!{chinese_boundary_suffix_pattern})"):
        year = reference_date.year + 1
        return constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    if chinese_search(rf"大后年(?!{chinese_boundary_suffix_pattern})"):
        year = reference_date.year + 3
        return constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    if chinese_search(rf"(?<!大)后年(?!{chinese_boundary_suffix_pattern})"):
        year = reference_date.year + 2
        return constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    if chinese_search(r"(?<!年)上半年"):
        return constraint(datetime(reference_date.year, 1, 1), datetime(reference_date.year, 6, 30))

    if chinese_search(r"(?<!年)下半年"):
        return constraint(datetime(reference_date.year, 7, 1), datetime(reference_date.year, 12, 31))

    if chinese_search(
        rf"(?<![上大])(上上|大上){_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)(?!{chinese_boundary_suffix_pattern})"
    ):
        start = reference_date - timedelta(days=reference_date.weekday() + 14)
        return constraint(start, start + timedelta(days=6))

    if chinese_search(
        rf"(?<![上大])(上上|大上){_CHINESE_OPTIONAL_PERIOD_MARKER}月(?!{chinese_month_boundary_suffix_pattern})"
    ):
        start = subtract_months(2).replace(day=1)
        return constraint(start, month_end(start.year, start.month))

    if chinese_search(rf"前一个?(周|星期|礼拜)(?!{chinese_boundary_suffix_pattern})"):
        start = reference_date - timedelta(days=reference_date.weekday() + 7)
        return constraint(start, start + timedelta(days=6))

    if chinese_search(rf"前一个?月(?!{chinese_month_boundary_suffix_pattern})"):
        first = reference_date.replace(day=1)
        end = first - timedelta(days=1)
        start = end.replace(day=1)
        return constraint(start, end)

    if chinese_search(rf"前一年(?!{chinese_boundary_suffix_pattern})"):
        year = reference_date.year - 1
        return constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    if chinese_search(rf"大前年(?!{chinese_boundary_suffix_pattern})"):
        year = reference_date.year - 3
        return constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    if chinese_search(rf"(?<!大)前年(?!{chinese_boundary_suffix_pattern})"):
        year = reference_date.year - 2
        return constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    if chinese_search(
        rf"(?<![上大])上{_CHINESE_OPTIONAL_PERIOD_MARKER}(周|星期|礼拜)(?!末)(?!{chinese_boundary_suffix_pattern})"
    ):
        start = reference_date - timedelta(days=reference_date.weekday() + 7)
        return constraint(start, start + timedelta(days=6))

    if chinese_search(rf"(?<![上大])上{_CHINESE_OPTIONAL_PERIOD_MARKER}月(?!{chinese_month_boundary_suffix_pattern})"):
        first = reference_date.replace(day=1)
        end = first - timedelta(days=1)
        start = end.replace(day=1)
        return constraint(start, end)

    if chinese_search(rf"(上一年度|上一个年度|上年度|去年|上一年|上年)(?!{chinese_boundary_suffix_pattern})"):
        year = reference_date.year - 1
        return constraint(datetime(year, 1, 1), datetime(year, 12, 31))

    chinese_month_match = chinese_search(
        rf"({chinese_year_pattern})\s*年\s*({chinese_month_pattern})\s*月(?!{chinese_month_boundary_suffix_pattern})"
    )
    if chinese_month_match:
        return chinese_month_period(chinese_month_match)

    bare_chinese_month_match = chinese_search(
        rf"(?<![年月])({chinese_month_pattern})\s*月份?(?!{chinese_month_boundary_suffix_pattern})"
    )
    if bare_chinese_month_match:
        return bare_month_period(bare_chinese_month_match.group(1))

    return None
