"""Regression tests for vectorize-io/hindsight#5644.

An HTTP 401/403 from the LLM is a property of the configured credential, not of
the moment: every replay sends the same key and earns the same rejection. The
providers already refuse to retry it, but ``extract_facts_from_text`` flattened
it into a generic ``RuntimeError`` and the worker replayed the whole retain
(every chunk) through its retry schedule.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from hindsight_api.engine.llm_interface import ProviderAuthError
from hindsight_api.engine.memory_engine import _is_non_retryable_task_error
from tests.test_content_policy_refusal_permanent import _run_extraction


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://proxy.example/v1/chat/completions")
    response = httpx.Response(
        status, request=request, json={"error": {"type": "auth_error", "message": "invalid proxy key"}}
    )
    cls = {401: openai.AuthenticationError, 403: openai.PermissionDeniedError}.get(status, openai.APIStatusError)
    return cls(f"Error code: {status}", response=response, body=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_rejected_chunk_raises_permanent_error(status):
    with pytest.raises(ProviderAuthError) as excinfo:
        await _run_extraction({0: _status_error(status), 1: _status_error(status)})

    assert "chunk 0" in str(excinfo.value)
    assert _is_non_retryable_task_error(excinfo.value) is True


@pytest.mark.asyncio
async def test_server_error_chunk_stays_retryable():
    with pytest.raises(RuntimeError) as excinfo:
        await _run_extraction({0: _status_error(503)})

    assert not isinstance(excinfo.value, ProviderAuthError)
    assert _is_non_retryable_task_error(excinfo.value) is False


@pytest.mark.parametrize("status", [401, 403])
def test_raw_auth_error_is_not_retried_by_the_worker(status):
    """Tasks that let the provider error through (consolidation, refresh) stop too."""
    assert _is_non_retryable_task_error(_status_error(status)) is True
    assert _is_non_retryable_task_error(_status_error(503)) is False
