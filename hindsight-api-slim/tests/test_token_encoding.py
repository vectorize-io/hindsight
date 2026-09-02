"""The shared tokenizer: what every token budget in the engine is counted with.

These are the properties the rest of the engine relies on and that a tokenizer
swap could silently break — the counts themselves, the special-token tolerance
that issue #1883 was about, and the fact that the encoding is selectable.
"""

import pytest

from hindsight_api.config import DEFAULT_TOKENIZER_ENCODING, ENV_TOKENIZER_ENCODING, clear_config_cache
from hindsight_api.engine.token_encoding import (
    BUNDLED_ENCODINGS,
    count_tokens,
    _load_encoding,
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
    """Truncation is the only operation that needs ids, so it is the only one that
    could still hit the tokenizer's ``encode()`` check. It must not."""
    assert truncate_to_tokens(SPECIAL_TOKEN_TEXT, 10_000).text == SPECIAL_TOKEN_TEXT


def test_counting_agrees_with_the_ids_truncation_uses():
    """``count()`` is the fast path for the same number ``len(encode_ordinary())``
    gives.

    Every budget in the engine counts one way and truncates the other, so a
    divergence here would show up as an off-by-N in the wrong direction. It is a
    real risk: the tokenizer has *two* id-producing spellings, and only
    ``encode_ordinary`` agrees with ``count`` on special-token literals.
    """
    for text in (
        "",
        "   \n\t  ",
        "the user asked about deployment latency",
        SPECIAL_TOKEN_TEXT,
        "🧠 naïve café 東京 مرحبا",
        "def f(x: int) -> str:\n    return f'{x!r}'  # ok\n",
    ):
        enc = _load_encoding()
        assert count_tokens(text) == len(enc.encode_ordinary(text)), repr(text)


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
