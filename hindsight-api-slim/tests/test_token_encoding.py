"""The shared tokenizer: what every token budget in the engine is counted with.

These are the properties the rest of the engine relies on and that a tokenizer
swap could silently break — the counts themselves, the special-token tolerance
that issue #1883 was about, and the fact that the encoding is selectable.
"""

import pytest

from hindsight_api.config import DEFAULT_TOKENIZER_ENCODING, ENV_TOKENIZER_ENCODING, clear_config_cache
from hindsight_api.engine.token_encoding import (
    BUNDLED_ENCODINGS,
    _load_encoding,
    count_tokens,
    truncate_many_to_tokens,
    truncate_to_tokens,
)

SPECIAL_TOKEN_TEXT = "the model emits <|endoftext|> and <|fim_prefix|> markers"


@pytest.fixture
def encoding_env(monkeypatch):
    """Select an encoding for one test, and undo both caches afterwards.

    ``_load_encoding`` is lru_cached and the config is globally cached, so a
    test that sets the variable without clearing both would either read a stale
    encoding itself or leak one into the next test.
    """

    def _set(name: str | None):
        if name is None:
            monkeypatch.delenv(ENV_TOKENIZER_ENCODING, raising=False)
        else:
            monkeypatch.setenv(ENV_TOKENIZER_ENCODING, name)
        clear_config_cache()
        _load_encoding.cache_clear()
        return _load_encoding()

    yield _set

    monkeypatch.delenv(ENV_TOKENIZER_ENCODING, raising=False)
    clear_config_cache()
    _load_encoding.cache_clear()


def test_defaults_to_o200k_base(encoding_env):
    assert DEFAULT_TOKENIZER_ENCODING == "o200k_base"
    assert encoding_env(None).name == "o200k_base"


def test_encoding_is_selectable(encoding_env):
    assert encoding_env("cl100k_base").name == "cl100k_base"


@pytest.mark.parametrize("name", BUNDLED_ENCODINGS)
def test_every_advertised_encoding_actually_loads(name, encoding_env):
    """``BUNDLED_ENCODINGS`` is hand-maintained, and it is what a misconfigured
    deployment is told to choose from — so it must not name a vocabulary the
    installed tokenizer does not ship. It has been wrong before: it still listed
    ``llama3`` and ``qwen3`` after the move off quicktok, which bundles neither.
    """
    assert encoding_env(name).name == name


def test_unknown_encoding_names_the_valid_ones(encoding_env):
    # A typo in the env var must not surface as a bare extension-module error in
    # the middle of a recall.
    with pytest.raises(ValueError, match="cl100k_base"):
        encoding_env("not_a_real_encoding")


def test_o200k_counts_non_latin_text_more_cheaply(encoding_env):
    """The reason o200k_base is the default.

    Both vocabularies are close on English, but cl100k_base spends far more
    tokens on non-Latin scripts than a current model would charge.
    """
    text = "🧠 memory ✅ done — naïve café 東京 مرحبا"

    o200k = encoding_env("o200k_base").count(text)
    cl100k = encoding_env("cl100k_base").count(text)

    assert o200k < cl100k


# --- special-token literals (issue #1883) ------------------------------------
# Counting and encoding must treat these as ordinary text. The tokenizer's own
# default is to *raise*, which reached users as an HTTP 500 on retain/recall.


def test_count_tokens_tolerates_special_token_literals():
    assert count_tokens(SPECIAL_TOKEN_TEXT) > 0


def test_truncation_tolerates_special_token_literals():
    """The other half of #1883: truncation must treat the literal as ordinary text
    too, not reject it. A budget this generous cuts nothing, so it must come back
    byte-for-byte."""
    assert truncate_to_tokens(SPECIAL_TOKEN_TEXT, 10_000).text == SPECIAL_TOKEN_TEXT


def test_counting_agrees_with_what_truncation_reports():
    """``count_tokens`` and ``truncate``'s ``original_tokens`` must be one number.

    They come from two different toktok entry points, and the engine mixes them:
    ``_truncate_inputs`` decides it truncated by comparing ``original_tokens``
    against a cap that other code arrived at with ``count_tokens``. A divergence
    would show up as an off-by-N in a budget, not as an error.
    """
    for text in (
        "",
        "   \n\t  ",
        "the user asked about deployment latency",
        SPECIAL_TOKEN_TEXT,
        "🧠 naïve café 東京 مرحبا",
        "def f(x: int) -> str:\n    return f'{x!r}'  # ok\n",
    ):
        assert count_tokens(text) == truncate_to_tokens(text, 1_000_000).original_tokens, repr(text)


# --- truncation ---------------------------------------------------------------


def test_truncate_leaves_short_text_untouched():
    result = truncate_to_tokens("alpha beta gamma", 100)
    assert result.text == "alpha beta gamma"
    assert result.original_tokens == count_tokens("alpha beta gamma")


def test_truncate_reports_the_original_size():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    result = truncate_to_tokens(text, 3)
    assert count_tokens(result.text) <= 3
    assert result.original_tokens == count_tokens(text)
    assert text.startswith(result.text)


def test_truncate_survives_special_token_literals():
    result = truncate_to_tokens(SPECIAL_TOKEN_TEXT * 50, 10)
    assert count_tokens(result.text) <= 10


def test_truncate_never_cuts_a_character_in_half(encoding_env):
    """A token boundary is not a character boundary.

    Byte-level BPE splits one character across several tokens — under o200k_base
    ``🧠`` is three — so cutting the id list at ``n`` can land mid-character.
    ``decode(encode("hello 🧠")[:2])``, which is what this function used to do,
    returned ``"hello \ufffd"``. The partial character must be dropped instead, at
    every cut point, for every vocabulary.
    """
    for name in BUNDLED_ENCODINGS:
        encoding_env(name)
        for text in ("hello 🧠", "東京タワー", "مرحبا بالعالم", "🧠🧠🧠"):
            for budget in range(0, count_tokens(text) + 1):
                truncated = truncate_to_tokens(text, budget).text
                assert "\ufffd" not in truncated, f"{name} {text!r}[:{budget}] -> {truncated!r}"
                # Dropping a partial character can only shorten the result, never
                # push it over the ceiling the caller asked for.
                assert count_tokens(truncated) <= budget
                assert text.startswith(truncated)


def test_truncate_reports_the_original_size_even_when_it_fits():
    """Callers branch on ``original_tokens`` to decide whether to warn, so it must
    be the whole input's count on the untruncated path too."""
    result = truncate_to_tokens("alpha beta gamma", 100)
    assert result.original_tokens == count_tokens("alpha beta gamma")


def test_truncate_many_matches_truncating_one_at_a_time():
    """The batched form is what reranking and embedding truncation both use, so it
    must not diverge from the single-text one it stands in for."""
    texts = ["hello 🧠", "short", "alpha beta gamma delta epsilon " * 20, "", SPECIAL_TOKEN_TEXT]

    batched = truncate_many_to_tokens(texts, 8)

    assert batched == [truncate_to_tokens(t, 8) for t in texts]


def test_truncate_many_of_nothing():
    assert truncate_many_to_tokens([], 10) == []


def test_a_negative_budget_empties_rather_than_raising():
    """A misconfigured cap used to slice a list with a negative index and silently
    drop tokens off the *end*; the native call takes an unsigned count and would
    raise. Neither is acceptable mid-request, so it clamps to empty."""
    result = truncate_to_tokens("alpha beta gamma", -5)
    assert result.text == ""
    assert result.original_tokens == count_tokens("alpha beta gamma")
