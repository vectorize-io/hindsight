"""Tests for shared TEI Retry-After and jitter handling."""

from datetime import datetime, timezone
from email.utils import format_datetime
from unittest.mock import patch

import httpx

from hindsight_api.engine.tei_retry import tei_retry_delay


def test_retry_after_takes_precedence_over_fallback() -> None:
    response = httpx.Response(429, headers={"Retry-After": "3"})

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0):
        assert tei_retry_delay(response, 0.5, max_delay=30.0) == 3.0


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
        assert tei_retry_delay(response, 0.5, max_delay=30.0) == 3.0


def test_fallback_jitter_is_bounded() -> None:
    response = httpx.Response(503)

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.25) as uniform:
        assert tei_retry_delay(response, 5.0, max_delay=30.0) == 5.25

    uniform.assert_called_once_with(0.0, 0.5)


def test_non_finite_and_malformed_retry_after_use_fallback() -> None:
    for value in ("Infinity", "NaN", "not-a-delay"):
        response = httpx.Response(429, headers={"Retry-After": value})
        with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0):
            assert tei_retry_delay(response, 0.5, max_delay=30.0) == 0.5


def test_oversized_retry_after_is_capped_by_client_timeout() -> None:
    response = httpx.Response(429, headers={"Retry-After": "1000000"})

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.0):
        assert tei_retry_delay(response, 0.5, max_delay=30.0) == 30.0


def test_capped_delay_uses_downward_jitter() -> None:
    response = httpx.Response(429, headers={"Retry-After": "1000000"})

    with patch("hindsight_api.engine.tei_retry.random.uniform", return_value=0.75) as uniform:
        assert tei_retry_delay(response, 0.5, max_delay=30.0) == 29.25

    uniform.assert_called_once_with(0.0, 1.0)
