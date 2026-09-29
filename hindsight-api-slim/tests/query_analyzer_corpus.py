"""Query corpus for the temporal-extraction characterization suite.

This module is the single source of truth for the inputs used by
``test_temporal_extraction.py`` (behaviour must not change) and by the
standalone latency harness in ``query_analyzer_bench.py``.

The corpus deliberately over-samples the shapes that recall actually sees:
ordinary non-temporal questions (the common case, and the one that used to be
the *slowest*), plus every branch of the hand-written period rules, plus the
inputs that historically broke the parser (issues #2768, #3217).

Each entry is (query, category). The reference date is fixed by the tests so
relative expressions resolve deterministically.
"""

from datetime import datetime

# Fixed reference date for every golden case. Chosen as a Wednesday so that
# "last monday"/"last friday" resolve on both sides of the week boundary.
REFERENCE_DATE = datetime(2026, 8, 12, 14, 30, 45, 123456)

# --------------------------------------------------------------------------
# 1. Non-temporal queries — the common recall case.
# --------------------------------------------------------------------------
NON_TEMPORAL_EN = [
    "what did I decide about the database schema",
    "user preferences for code style",
    "how does the reranker work",
    "postgres connection pooling settings in the api server and what we changed",
    "bug",
    "",
    "   ",
    "a",
    "the",
    "what is the current architecture of the retain pipeline",
    "summarize my notes on vector indexes",
    "who is responsible for the billing service",
    "explain the difference between world facts and experience facts",
    "list all the mental models about my coding preferences",
    "why did we choose pgvector over qdrant",
    "the user prefers functional programming patterns",
    "error handling conventions in the rust cli",
    "what are the disposition traits",
    "how do I configure the embeddings provider",
    "tell me about entity resolution",
    "recall everything about authentication",
    "notes",
    "?",
    "!!!",
    "...",
    "n/a",
    "TODO",
    "SELECT * FROM memory_units",
    "def analyze(self, query: str) -> QueryAnalysis:",
    "https://example.com/docs/page?id=42",
    "user@example.com",
    "/Users/nico/dev/hindsight/api",
    "🧠 memory system",
    "the quick brown fox jumps over the lazy dog",
]

# Words that are timezone abbreviations as substrings ('ect' in "connection",
# 'hat' in "what") or standalone. These drove the tz-scan hot path.
TZ_ABBREV_TRAPS = [
    "connection pooling",
    "what happened to the expected output",
    "the cat sat on the mat",
    "east west north south",
    "met the team",
    "wet season",
    "subject matter expert",
    "architecture decision record",
    "project status",
    "select the best candidate",
    "detect the language",
    "expect a response",
    "perfect score",
    "rejected the proposal",
    "collection of documents",
    "inspection report",
]

# Short common words that dateparser's search resolved as weekdays (#2768).
FALSE_POSITIVE_TRAPS = [
    "we",
    "me",
    "did",
    "do",
    "we did it",
    "did we do that",
    "do we have a plan",
    "me and the team",
    "we discussed the api",
    "did the migration run",
    "so what do we do now",
    "can we do this",
    "mar",
    "may",
    "sun",
    "sat",
    "wed",
    "the may report",
    "may I ask a question",
    "she may have been right",
]

# --------------------------------------------------------------------------
# 2. Period fast-path — one per branch of _extract_non_chinese_period.
# --------------------------------------------------------------------------
PERIOD_EN = [
    "what happened yesterday",
    "yesterday",
    "notes from today",
    "today",
    "a couple of days ago",
    "couple days ago",
    "a few days ago",
    "few day ago",
    "a couple of weeks ago",
    "couple weeks ago",
    "a few weeks ago",
    "few week ago",
    "a couple of months ago",
    "couple months ago",
    "a few months ago",
    "few month ago",
    "last week",
    "what did I do last week",
    "last month",
    "the incident last month",
    "last year",
    "revenue last year",
    "last weekend",
    "what did I do last weekend",
]

PERIOD_ES = [
    "ayer",
    "qué hice ayer",
    "hoy",
    "la semana pasada",
    "qué hice la semana pasada",
    "el mes pasado",
    "el año pasado",
    "el fin de semana pasado",
]

PERIOD_IT = [
    "ieri",
    "cosa ho fatto ieri",
    "oggi",
    "la settimana scorsa",
    "il mese scorso",
    "l'anno scorso",
    "lo scorso fine settimana",
]

PERIOD_FR = [
    "hier",
    "qu'ai-je fait hier",
    "aujourd'hui",
    "aujourd hui",
    "la semaine dernière",
    "la semaine derniere",
    "le mois dernier",
    "l'année dernière",
    "le week-end dernier",
    "le weekend dernier",
]

PERIOD_DE = [
    "gestern",
    "was ist gestern passiert",
    "heute",
    "letzte woche",
    "letzten monat",
    "letzte monat",
    "letztes jahr",
    "letzte jahr",
    "letztes wochenende",
]

PERIOD_RU = [
    "вчера",
    "что было вчера",
    "позавчера",
    "сегодня",
    "пару дней назад",
    "пары дней назад",
    "несколько дней назад",
    "пару недель назад",
    "несколько недель назад",
    "пару месяцев назад",
    "несколько месяцев назад",
    "на прошлой неделе",
    "прошлая неделя",
    "в прошлом месяце",
    "прошлый месяц",
    "в прошлом году",
    "прошлый год",
    "на прошлых выходных",
    "прошлые выходные",
]

# Month + year across every language variant in the month_patterns table.
MONTH_YEAR = []
for _month_variants in [
    ("january", "enero", "gennaio", "janvier", "januar", "января"),
    ("february", "febrero", "febbraio", "février", "februar", "февраля"),
    ("march", "marzo", "mars", "märz", "марта"),
    ("april", "abril", "aprile", "avril", "апреля"),
    ("may", "mayo", "maggio", "mai", "мая"),
    ("june", "junio", "giugno", "juin", "juni", "июня"),
    ("july", "julio", "luglio", "juillet", "juli", "июля"),
    ("august", "agosto", "août", "aout", "августа"),
    ("september", "septiembre", "settembre", "septembre", "сентября"),
    ("october", "octubre", "ottobre", "octobre", "oktober", "октября"),
    ("november", "noviembre", "novembre", "ноября"),
    ("december", "diciembre", "dicembre", "décembre", "dezember", "декабря"),
]:
    for _name in _month_variants:
        MONTH_YEAR.append(f"{_name} 2024")
        MONTH_YEAR.append(f"notes from {_name} 2025")
# The "day precedes month" carve-out that must fall through to dateparser.
MONTH_YEAR += [
    "13 июля 2026",
    "13 July 2026",
    "3 marzo 2025",
    "10 de julio de 2026",
    "1 january 2020",
    "31 december 1999",
]

# --------------------------------------------------------------------------
# 3. Chinese period rules (chinese_temporal_periods.py is the largest rule set).
# --------------------------------------------------------------------------
PERIOD_ZH = [
    "昨天",
    "昨天做了什么",
    "今天",
    "前天",
    "上周",
    "上周做了什么",
    "上个月",
    "上月",
    "去年",
    "上周末",
    "本周",
    "这周",
    "本月",
    "这个月",
    "今年",
    "明天",
    "下周",
    "下个月",
    "明年",
    "三天前",
    "五天前",
    "两天前",
    "几天前",
    "三周前",
    "两周前",
    "几周前",
    "三个月前",
    "两个月前",
    "几个月前",
    "三年前",
    "两年前",
    "几年前",
    "2024年3月",
    "2024年3月3日",
    "2024年",
    "3月3日",
    "一月",
    "十二月",
    "上半年",
    "下半年",
    "第一季度",
    "第二季度",
    "十万年前",  # issue #3217: year underflow
    "一百万年前",
    "十亿年前",
    "最近三天",
    "最近一周",
    "最近一个月",
    "过去三天",
    "过去一周",
    "这周末",
    "上个星期",
    "上星期",
    "星期一",
    "周一",
    "礼拜一",
]

# CJK text where dateparser used to match an embedded substring.
CJK_EMBEDDED = [
    "上海的天气",
    "北京会议记录",
    "我在东京工作",
    "数据库设计文档",
    "这个功能怎么用",
    "日本語のテキスト",
    "한국어 텍스트",
    "上海と東京",
]

# --------------------------------------------------------------------------
# 4. Explicit dates that fall through to dateparser.
# --------------------------------------------------------------------------
EXPLICIT_DATES = [
    "2026-06-10",
    "what did I do on 2026-06-10",
    "meeting notes from March 3rd",
    "the incident on 10 July 2026",
    "notes from 3 March 2025",
    "2024/01/15",
    "01/15/2024",
    "15.01.2024",
    "Jan 5, 2021",
    "5 Jan 2021",
    "December 25th 2020",
    "the release on 2023-11-17",
    "between 2020-01-01 and 2020-12-31",
    "riunione del 3 marzo 2025",
    "reunión del 10 de julio de 2026",
    "Besprechung am 3. März 2025",
    "réunion du 3 mars 2025",
    "встреча 3 марта 2025",
    "2025年3月3日の会議",
    "o que aconteceu há 5 dias",
    "что было 5 дней назад",
    "hace 5 días",
    "il y a 5 jours",
    "vor 5 Tagen",
    "5 giorni fa",
    "at 3pm",
    "3:30pm",
    "noon",
    "midnight",
    "this morning",
    "tonight",
    "tomorrow",
    "next week",
    "in 3 days",
    "two days later",
    "1mon ago",
    "june 0000",  # issue #3217: unrepresentable year
    "january 0000",
    "december 0001",
    "year 9999",
    "31 february 2020",  # invalid date
    "2020-13-45",
    "99999999999",
    "0000-00-00",
]

# --------------------------------------------------------------------------
# 5. Long inputs — consolidation recalls pass stored fact text as the query.
# --------------------------------------------------------------------------
_LONG_BASE = (
    "The user configured the retain pipeline to use a chunk size of 2000 tokens "
    "and enabled delta refresh for mental models. They prefer functional patterns "
    "and asked that the reranker stay on the local cross-encoder rather than TEI. "
)
LONG_TEXTS = [
    _LONG_BASE,
    _LONG_BASE * 3,
    _LONG_BASE * 8,
    _LONG_BASE + "This was decided last week during the architecture review.",
    _LONG_BASE * 4 + "The migration ran on 2026-06-10 without incident.",
    "x" * 500,
    "1234567890" * 40,  # long digit run — the 1.4.1 ReDoS shape
    " ".join(["word"] * 300),
]

# --------------------------------------------------------------------------
# 6. Adversarial / structural edge cases.
# --------------------------------------------------------------------------
EDGE_CASES = [
    "\n",
    "\t\t",
    "\x00",
    "café",
    "naïve",
    "ÅÄÖ",
    "ﬁ ligature",
    "ＦＵＬＬＷＩＤＴＨ",  # NFKC normalization path
    "２０２４年３月",  # fullwidth digits
    "a" * 1000,
    "-" * 100,
    "2026-06-10 " * 20,
    "yesterday " * 30,
    "上周 " * 30,
    "mixed 上周 and last week",
    "yesterday and today and tomorrow",
    "last week or last month",
    "not yesterday",
    "the word yesterday appears in this sentence about nothing",
    "yesterdays",
    "yesterday's meeting",
    "LAST WEEK",
    "LaSt WeEk",
    "   last week   ",
]


def _labelled(items: list[str], category: str) -> list[tuple[str, str]]:
    return [(q, category) for q in items]


def build_corpus() -> list[tuple[str, str]]:
    """Return the full (query, category) corpus, de-duplicated, order-stable."""
    groups = [
        (NON_TEMPORAL_EN, "non_temporal_en"),
        (TZ_ABBREV_TRAPS, "tz_abbrev_trap"),
        (FALSE_POSITIVE_TRAPS, "false_positive_trap"),
        (PERIOD_EN, "period_en"),
        (PERIOD_ES, "period_es"),
        (PERIOD_IT, "period_it"),
        (PERIOD_FR, "period_fr"),
        (PERIOD_DE, "period_de"),
        (PERIOD_RU, "period_ru"),
        (MONTH_YEAR, "month_year"),
        (PERIOD_ZH, "period_zh"),
        (CJK_EMBEDDED, "cjk_embedded"),
        (EXPLICIT_DATES, "explicit_date"),
        (LONG_TEXTS, "long_text"),
        (EDGE_CASES, "edge_case"),
    ]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for items, category in groups:
        for q, cat in _labelled(items, category):
            if q in seen:
                continue
            seen.add(q)
            out.append((q, cat))
    return out


# Queries used by the latency gate. Weighted toward the real recall mix:
# mostly non-temporal, some period expressions, a few explicit dates.
#
# Split deliberately by input *shape*, because the two behave very differently.
# Detection cost is O(text length x 205 locales), so a user-typed query and a
# 2 KB document excerpt are not the same workload and averaging them hides which
# one is slow.
def build_query_workload() -> list[str]:
    """Query-shaped input: what a user (or an agent) actually types into recall."""
    return (
        NON_TEMPORAL_EN * 3 + TZ_ABBREV_TRAPS * 2 + PERIOD_EN + PERIOD_ZH + PERIOD_RU + EXPLICIT_DATES + MONTH_YEAR[:20]
    )


def build_document_workload() -> list[str]:
    """Document-shaped input: consolidation recalls pass stored fact text."""
    return LONG_TEXTS


def build_perf_workload() -> list[str]:
    return build_query_workload() + build_document_workload()
