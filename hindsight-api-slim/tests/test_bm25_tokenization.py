from hindsight_api.engine.search.bm25_query import build_native_tsquery, build_native_tsquery_fallback, tokenize_query


def test_tokenize_query_preserves_ipv4_address_for_native_text_search():
    assert tokenize_query("59.41.7.83") == ["59.41.7.83"]


def test_tokenize_query_preserves_ipv4_address_in_mixed_query():
    assert tokenize_query("zhouwei@59.41.7.83 SSH") == ["59.41.7.83", "zhouwei", "ssh"]


def test_tokenize_query_still_strips_general_punctuation():
    assert tokenize_query("Gitea SSH连接信息: ssh -p 65017") == [
        "gitea",
        "ssh",
        "连接",
        "信息",
        "p",
        "65017",
    ]


def test_build_native_tsquery_keeps_non_chinese_or_semantics():
    assert build_native_tsquery("Gitea SSH") == "gitea | ssh"


def test_build_native_tsquery_matches_chinese_word_or_character_indexes():
    assert build_native_tsquery("周奕婷") == "(周奕婷 | 周 & 奕 & 婷)"


def test_build_native_tsquery_remains_strict_for_chinese_natural_language_queries():
    assert build_native_tsquery("周奕婷的学校和家庭信息") == (
        "(周奕婷 & 学校 & 家庭 & 信息 | 周 & 奕 & 婷 & 学 & 校 & 家 & 庭 & 信 & 息)"
    )


def test_build_native_tsquery_fallback_requires_at_least_two_chinese_terms():
    assert build_native_tsquery_fallback("周奕婷的学校和家庭信息") == (
        "(((周奕婷 | 周 & 奕 & 婷) & (学校 | 学 & 校)) | "
        "((周奕婷 | 周 & 奕 & 婷) & (家庭 | 家 & 庭)) | "
        "((周奕婷 | 周 & 奕 & 婷) & (信息 | 信 & 息)) | "
        "((学校 | 学 & 校) & (家庭 | 家 & 庭)) | "
        "((学校 | 学 & 校) & (信息 | 信 & 息)) | "
        "((家庭 | 家 & 庭) & (信息 | 信 & 息)))"
    )


def test_build_native_tsquery_fallback_uses_pairwise_terms_without_specific_anchor():
    assert build_native_tsquery_fallback("学校和家庭信息") == (
        "(((学校 | 学 & 校) & (家庭 | 家 & 庭)) | "
        "((学校 | 学 & 校) & (信息 | 信 & 息)) | "
        "((家庭 | 家 & 庭) & (信息 | 信 & 息)))"
    )


def test_build_native_tsquery_requires_ascii_terms_for_mixed_language_queries():
    assert build_native_tsquery("Gitea 端口") == "gitea & (端口 | 端 & 口)"


def test_build_native_tsquery_preserves_ipv4_address():
    assert build_native_tsquery("zhouwei 59.41.7.83") == "59.41.7.83 | zhouwei"


def test_build_native_tsquery_can_use_english_legacy_mode():
    assert build_native_tsquery("周奕婷", language="english") == "周奕婷"
