"""Retry policy for idempotent calls when the server is at capacity.

The server's admission control answers 503 with ``Retry-After`` when a lane is
full. Retrying is only safe if it honours that hint and spreads the retries out:
a burst that all obey ``Retry-After: 1`` exactly comes back in lockstep and
rebuilds the spike that caused the rejection.
"""

import random

import pytest

from hindsight_client.hindsight_client import (
    _retry_after_seconds,
    _retry_on_capacity,
)
from hindsight_client_api.exceptions import ApiException


def _at_capacity(status: int = 503, retry_after: str | None = "2") -> ApiException:
    e = ApiException(status=status)
    e.headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return e


class TestRetryAfterParsing:
    def test_reads_delta_seconds(self):
        assert _retry_after_seconds(_at_capacity(retry_after="7")) == 7.0

    def test_missing_header_returns_none(self):
        assert _retry_after_seconds(_at_capacity(retry_after=None)) is None

    def test_http_date_form_falls_back(self):
        """Date form is legal but rarer; the caller's backoff is a better answer."""
        assert _retry_after_seconds(_at_capacity(retry_after="Wed, 21 Oct 2026 07:28:00 GMT")) is None

    def test_negative_is_clamped(self):
        assert _retry_after_seconds(_at_capacity(retry_after="-5")) == 0.0


class TestRetryBehaviour:
    async def test_succeeds_without_retrying(self):
        calls = []

        async def call():
            calls.append(1)
            return "ok"

        assert await _retry_on_capacity(call, 3, random.Random(0)) == "ok"
        assert len(calls) == 1

    async def test_retries_503_then_succeeds(self):
        calls = []

        async def call():
            calls.append(1)
            if len(calls) < 3:
                raise _at_capacity(503, "0")
            return "ok"

        assert await _retry_on_capacity(call, 3, random.Random(0)) == "ok"
        assert len(calls) == 3

    async def test_retries_429_too(self):
        calls = []

        async def call():
            calls.append(1)
            if len(calls) < 2:
                raise _at_capacity(429, "0")
            return "ok"

        assert await _retry_on_capacity(call, 3, random.Random(0)) == "ok"

    async def test_gives_up_after_max_attempts_and_reraises(self):
        calls = []

        async def call():
            calls.append(1)
            raise _at_capacity(503, "0")

        with pytest.raises(ApiException) as excinfo:
            await _retry_on_capacity(call, 3, random.Random(0))
        assert excinfo.value.status == 503
        assert len(calls) == 3

    async def test_does_not_retry_other_errors(self):
        """A 400 is the caller's problem; repeating it just wastes a round trip."""
        calls = []

        async def call():
            calls.append(1)
            raise ApiException(status=400)

        with pytest.raises(ApiException):
            await _retry_on_capacity(call, 3, random.Random(0))
        assert len(calls) == 1

    async def test_max_attempts_of_one_disables_retrying(self):
        calls = []

        async def call():
            calls.append(1)
            raise _at_capacity(503, "0")

        with pytest.raises(ApiException):
            await _retry_on_capacity(call, 1, random.Random(0))
        assert len(calls) == 1

    async def test_wait_is_jittered_within_retry_after(self, monkeypatch):
        """Two clients given the same Retry-After must not wake together."""
        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr("hindsight_client.hindsight_client.asyncio.sleep", fake_sleep)

        async def call():
            raise _at_capacity(503, "4")

        for seed in (1, 2, 3):
            with pytest.raises(ApiException):
                await _retry_on_capacity(call, 2, random.Random(seed))

        assert all(0 <= s <= 4 for s in slept), slept
        assert len(set(slept)) > 1, "identical waits: jitter is not being applied"
