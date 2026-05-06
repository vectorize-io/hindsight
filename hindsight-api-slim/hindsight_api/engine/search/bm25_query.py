"""BM25 query text preparation helpers.

Supports Chinese tokenization via jieba (lazy-loaded) and provides
native PostgreSQL tsquery builders matching indexed tokenization.
"""

import logging
import re

_IPV4_TOKEN_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_CHINESE_LANGUAGES = {"zh", "zh-cn", "zh_cn", "chinese", "cn"}
_CHINESE_QUERY_STOPWORDS = {"的", "和", "与", "及", "或", "了", "呢", "吗", "吧", "啊"}

_jieba_instance = None


def _get_jieba():
    """Lazy-load jieba for Chinese query segmentation when installed."""
    global _jieba_instance
    if _jieba_instance is not None:
        return _jieba_instance

    try:
        import jieba

        jieba.setLogLevel(logging.WARNING)
        _jieba_instance = jieba
        return jieba
    except ImportError:
        _jieba_instance = False
        return None


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for token in tokens:
        normalized = token.strip()
        if normalized and _TOKEN_CHAR_RE.search(normalized) and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _tokenize_chinese_query(text: str) -> list[str]:
    jieba = _get_jieba()
    if jieba:
        try:
            tokens = []
            for token in jieba.cut(text):
                if _CHINESE_CHAR_RE.search(token):
                    tokens.append(token)
                else:
                    tokens.extend(re.sub(r"[^\w\s]", " ", token).split())
            return _dedupe_tokens(tokens)
        except Exception:
            pass

    return _dedupe_tokens(_CHINESE_CHAR_RE.findall(text))


def _format_chinese_token_query(token: str) -> str:
    token_chars = _dedupe_tokens(_CHINESE_CHAR_RE.findall(token))
    if len(token_chars) > 1:
        return f"({token} | {' & '.join(token_chars)})"
    return token


def tokenize_query(query_text: str, *, language: str = "zh") -> list[str]:
    """Normalize query text into BM25 tokens.

    Supports Chinese tokenization via jieba (lazy-loaded) and falls back
    to ASCII word splitting for non-Chinese text.
    """
    normalized = query_text.lower()
    preserved_tokens = _IPV4_TOKEN_RE.findall(normalized)
    normalized = _IPV4_TOKEN_RE.sub(" ", normalized)

    if language.lower() in _CHINESE_LANGUAGES and _CHINESE_CHAR_RE.search(normalized):
        return _dedupe_tokens([*preserved_tokens, *_tokenize_chinese_query(normalized)])

    return _dedupe_tokens([*preserved_tokens, *re.sub(r"[^\w\s]", " ", normalized).split()])


def build_native_tsquery(query_text: str, *, language: str = "zh") -> str | None:
    """Build native PostgreSQL tsquery text matching indexed tokenization.

    For Chinese queries, produces a tsquery that matches both jieba word
    tokens and individual characters, giving the best recall for Chinese text.
    """
    tokens = tokenize_query(query_text, language=language)
    if not tokens:
        return None

    normalized = query_text.lower()
    if language.lower() not in _CHINESE_LANGUAGES or not _CHINESE_CHAR_RE.search(normalized):
        return " | ".join(tokens)

    ascii_tokens = [token for token in tokens if not _CHINESE_CHAR_RE.search(token)]
    chinese_word_tokens = [
        token for token in tokens if _CHINESE_CHAR_RE.search(token) and token not in _CHINESE_QUERY_STOPWORDS
    ]
    chinese_chars = _dedupe_tokens(
        char for char in _CHINESE_CHAR_RE.findall(normalized) if char not in _CHINESE_QUERY_STOPWORDS
    )

    chinese_alternatives = []
    if chinese_word_tokens:
        chinese_alternatives.append(" & ".join(chinese_word_tokens))
    if chinese_chars:
        chinese_alternatives.append(" & ".join(chinese_chars))

    query_parts = [*ascii_tokens]
    if chinese_alternatives:
        query_parts.append(f"({' | '.join(chinese_alternatives)})")

    return " & ".join(query_parts) if query_parts else None


def build_native_tsquery_fallback(query_text: str, *, language: str = "zh") -> str | None:
    """Build a looser native PostgreSQL tsquery used only when strict BM25 is empty.

    For Chinese queries, uses pairwise AND combinations for better recall
    when the strict tsquery returns no results.
    """
    tokens = tokenize_query(query_text, language=language)
    if not tokens:
        return None

    normalized = query_text.lower()
    if language.lower() not in _CHINESE_LANGUAGES or not _CHINESE_CHAR_RE.search(normalized):
        return " | ".join(tokens)

    ascii_tokens = [token for token in tokens if not _CHINESE_CHAR_RE.search(token)]
    chinese_word_tokens = [
        token for token in tokens if _CHINESE_CHAR_RE.search(token) and token not in _CHINESE_QUERY_STOPWORDS
    ]
    if not chinese_word_tokens:
        return " & ".join(ascii_tokens) if ascii_tokens else None
    if len(chinese_word_tokens) == 1:
        chinese_query = _format_chinese_token_query(chinese_word_tokens[0])
    else:
        token_queries = [_format_chinese_token_query(token) for token in chinese_word_tokens]
        pairs = [
            f"({left} & {right})"
            for index, left in enumerate(token_queries)
            for right in token_queries[index + 1 :]
        ]
        chinese_query = f"({' | '.join(pairs)})"

    query_parts = [*ascii_tokens, chinese_query]
    return " & ".join(query_parts) if query_parts else None
