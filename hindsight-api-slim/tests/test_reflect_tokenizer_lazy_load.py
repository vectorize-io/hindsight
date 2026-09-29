"""The tokenizer encoding must load lazily — on the first count, not on import.

These tests need ``_load_encoding`` to actually call ``toktok._encoding`` while it
is patched, which means starting from an empty cache. They arrange that by clearing
the cache, *not* by dropping ``engine.token_encoding`` from ``sys.modules``: a
reimport under the patch builds a second module object whose ``lru_cache`` then
holds the MagicMock and stays there after the patch is undone. See
``test_patched_tokenizer_does_not_leak_into_later_token_calls``.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.engine import token_encoding
from hindsight_api.engine.token_encoding import _load_encoding, truncate_to_tokens


def _drop_reflect_modules() -> None:
    for name in list(sys.modules):
        if name == "hindsight_api.engine.reflect" or name.startswith("hindsight_api.engine.reflect."):
            sys.modules.pop(name)


@pytest.fixture(autouse=True)
def _empty_encoding_cache():
    """Give each test an empty cache, and leave one behind.

    Clearing beforehand is what makes the patch observable at all. Clearing
    afterwards is what keeps a patched tokenizer from outliving its ``with`` block:
    ``_load_encoding`` is ``lru_cache``d process-wide, so a MagicMock cached here
    would otherwise serve every later test in the same worker.
    """
    _load_encoding.cache_clear()
    yield
    _load_encoding.cache_clear()


def test_reflect_import_does_not_load_tokenizer_encoding():
    _drop_reflect_modules()

    with patch("toktok._encoding") as load_encoding:
        reflect = importlib.import_module("hindsight_api.engine.reflect")

    load_encoding.assert_not_called()
    assert reflect.run_reflect_agent is not None


def test_reflect_token_counting_loads_tokenizer_encoding_when_used():
    _drop_reflect_modules()
    fake_encoding = MagicMock()
    # Counting goes through Tokenizer.count(), which takes no kwargs.
    fake_encoding.count.side_effect = lambda text: len(text.split())
    # Truncation still needs ids; truncate_to_tokens encodes with encode_ordinary.
    fake_encoding.encode_ordinary.side_effect = lambda text: text.split()

    with patch("toktok._encoding", return_value=fake_encoding) as load_encoding:
        agent = importlib.import_module("hindsight_api.engine.reflect.agent")
        prompts = importlib.import_module("hindsight_api.engine.reflect.prompts")

        count = agent._count_messages_tokens([{"role": "user", "content": "one two"}])
        final_prompt = prompts.build_final_prompt(
            query="What happened?",
            context_history=[{"tool": "recall", "output": {"answer": "three four"}}],
            bank_profile={"name": "test"},
            max_context_tokens=1000,
        )

    from hindsight_api.config import _get_raw_config

    assert count == 2
    assert "three four" in final_prompt
    # Whatever this deployment configured — asserting the literal default would
    # break for anyone with HINDSIGHT_API_TOKENIZER_ENCODING set in their .env.
    load_encoding.assert_called_once_with(_get_raw_config().tokenizer_encoding)


def test_patched_tokenizer_does_not_leak_into_later_token_calls():
    """A patched tokenizer must not outlive its ``with`` block.

    Regression test for a cross-test leak that reddened test-api shards at random.
    This module used to drop ``engine.token_encoding`` from ``sys.modules`` so that
    the reimport under ``patch("toktok._encoding")`` would observe the patch. The
    reimported module stayed in ``sys.modules`` with the MagicMock in its
    ``lru_cache``, so the next test in the same worker to truncate anything failed
    with ``KeyError: unknown encoding <MagicMock name='mock.name'>`` out of
    ``toktok.truncate`` — and anything that merely *counted* got
    ``len(text.split())`` back with no error at all, which is worse.

    The victim was whichever test the scheduler happened to run next, so it looked
    like a different flaky test each run and passed in isolation every time.
    """
    fake_encoding = MagicMock()
    fake_encoding.count.side_effect = lambda text: len(text.split())

    _drop_reflect_modules()
    with patch("toktok._encoding", return_value=fake_encoding):
        agent = importlib.import_module("hindsight_api.engine.reflect.agent")
        assert agent._count_messages_tokens([{"role": "user", "content": "one two"}]) == 2

    # No second module object: everything that did `from ..token_encoding import ...`
    # still points at the one whose cache the fixture clears.
    assert sys.modules["hindsight_api.engine.token_encoding"] is token_encoding

    # And the real tokenizer is back. This is the call that used to raise.
    _load_encoding.cache_clear()
    result = truncate_to_tokens("hello world", 1)
    assert isinstance(result.original_tokens, int)
    assert result.original_tokens > 0
