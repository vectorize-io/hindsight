"""Tests that recall_async surfaces a non-opaque error message when the
underlying retrieval pipeline raises an exception with empty __str__.

Regression for issue #1384: ``raise Exception(f"Failed to search memories: ...{e}")``
collapsed to ``Failed to search memories: `` for any exception whose __str__()
returns blank, dropping the original class name and traceback chain.
"""

from unittest.mock import patch

import pytest

from hindsight_api import MemoryEngine, RequestContext

RC = RequestContext(tenant_id="default")


class _SilentError(Exception):
    """Mimics asyncpg.exceptions.ConnectionDoesNotExistError() and similar
    exceptions whose __str__() returns blank when raised with no args."""


async def _raise_silent(*_args, **_kwargs):
    raise _SilentError()


async def test_recall_async_error_preserves_original(memory_no_llm_verify: MemoryEngine):
    engine = memory_no_llm_verify
    bank_id = "test-error-propagation"
    await engine.get_bank_profile(bank_id, request_context=RC)

    try:
        with patch(
            "hindsight_api.engine.memory_engine.embedding_utils.generate_embeddings_batch",
            side_effect=_raise_silent,
        ):
            with pytest.raises(RuntimeError) as excinfo:
                await engine.recall_async(
                    bank_id=bank_id,
                    query="anything",
                    request_context=RC,
                )

        # The wrapping message must include the original exception class name —
        # the symptom in #1384 was an empty trailer like "Failed to search memories: ".
        message = str(excinfo.value)
        assert "Failed to search memories" in message
        assert "_SilentError" in message, (
            f"wrapper message dropped the original exception class: {message!r}"
        )

        # `from e` chain must be preserved so worker logs / debuggers can walk
        # back to the real cause.
        assert isinstance(excinfo.value.__cause__, _SilentError)
    finally:
        await engine.delete_bank(bank_id, request_context=RC)
