"""Tests for shared TEI Retry-After and jitter handling."""

from datetime import datetime, timezone
from email.utils import format_datetime
from unittest.mock import patch

import httpx

from hindsight_api.engine.tei_retry import MAX_RETRY_DELAY_SECONDS, tei_retry_delay


def test_retry_after_takes_precedence_over_fallback() -> None:
    response = httpx.Response(429, headers={"Retry-After": "3"})

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0):
        assert tei_retry_delay(response, 0.5, request_timeout=30.0) == 3.0


def test_http_date_retry_after_is_supported() -> None:
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    response = httpx.Response(
        429,
        headers={"Retry-After": format_datetime(now.replace(second=3), usegmt=True)},
    )

    with (
        patch("hindsight_api.engine.tei_retry.datetime") as mock_datetime,
        patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0),
    ):
        mock_datetime.now.return_value = now
        assert tei_retry_delay(response, 0.5, request_timeout=30.0) == 3.0


def test_fallback_jitter_spans_half_the_delay() -> None:
    """The jitter window must be wide enough to break a synchronised burst apart."""
    response = httpx.Response(503)

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.25) as uniform:
        assert tei_retry_delay(response, 2.0, request_timeout=30.0) == 2.25

    uniform.assert_called_once_with(0.0, 1.0)


def test_non_finite_and_malformed_retry_after_use_fallback() -> None:
    for value in ("Infinity", "NaN", "not-a-delay"):
        response = httpx.Response(429, headers={"Retry-After": value})
        with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0):
            assert tei_retry_delay(response, 0.5, request_timeout=30.0) == 0.5


def test_oversized_retry_after_is_capped_below_the_request_timeout() -> None:
    """A large Retry-After must not stall the reranker for a whole request timeout."""
    response = httpx.Response(429, headers={"Retry-After": "1000000"})

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0):
        delay = tei_retry_delay(response, 0.5, request_timeout=30.0)

    assert delay == MAX_RETRY_DELAY_SECONDS
    assert delay < 30.0


def test_short_request_timeout_lowers_the_cap() -> None:
    response = httpx.Response(429, headers={"Retry-After": "1000000"})

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0):
        assert tei_retry_delay(response, 0.5, request_timeout=1.0) == 1.0


def test_capped_delay_uses_downward_jitter() -> None:
    response = httpx.Response(429, headers={"Retry-After": "1000000"})

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.75) as uniform:
        assert tei_retry_delay(response, 0.5, request_timeout=30.0) == MAX_RETRY_DELAY_SECONDS - 0.75

    uniform.assert_called_once_with(0.0, MAX_RETRY_DELAY_SECONDS * 0.5)


def test_zero_delay_short_circuits_without_jitter() -> None:
    response = httpx.Response(429)

    with patch("hindsight_api.engine.tei_retry.random.uniform") as uniform:
        assert tei_retry_delay(response, 0.0, request_timeout=30.0) == 0.0

    uniform.assert_not_called()
