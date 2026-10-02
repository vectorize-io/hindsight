"""An unusable Retry-After must not turn a bounded retry into an infinite wait."""

import asyncio
import random

import pytest

from hindsight_client.hindsight_client import _retry_after_seconds, _retry_on_capacity
from hindsight_client_api.exceptions import ApiException


@pytest.mark.parametrize("header", ["Infinity", "inf", "1e309", "NaN"])
async def test_nonfinite_retry_after_uses_finite_backoff(header: str) -> None:
    error = ApiException(status=503)
    error.headers = {"Retry-After": header}
    attempts = 0

    async def send() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error
        return "recovered"

    assert await asyncio.wait_for(_retry_on_capacity(send, 2, random.Random(1)), timeout=1) == "recovered"
    assert attempts == 2
    assert _retry_after_seconds(error) is None
