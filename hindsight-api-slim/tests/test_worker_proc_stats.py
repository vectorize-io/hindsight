"""Tests for the worker's process-memory stats line.

The distinction these pin down is not cosmetic. ``[WORKER_STATS] ... rss_mb=``
used to report ``ru_maxrss``, the high-water mark, which never decreases — so a
single transient allocation pinned the field at its peak for the life of the
process and every later reading looked like memory that was still held. That is
exactly how a transient reranker burst was misread as retained heap while
diagnosing issue #3355.
"""

import sys
from unittest.mock import MagicMock, patch

from hindsight_api.worker.poller import WorkerPoller, _current_rss_bytes

# _format_proc_stats never touches self, so it can be called unbound against a
# stand-in rather than building a DB-backed poller.
_format_proc_stats = WorkerPoller._format_proc_stats


def _fake_rusage(peak_native: int) -> MagicMock:
    """getrusage() result whose ru_maxrss is in the running platform's own units."""
    return MagicMock(ru_maxrss=peak_native)


def _native_peak_units(peak_bytes: int) -> int:
    """Express a byte count the way ru_maxrss would on this platform."""
    return peak_bytes if sys.platform == "darwin" else peak_bytes // 1024


class TestCurrentRssBytes:
    def test_returns_positive_size_or_none(self):
        rss = _current_rss_bytes()
        if sys.platform == "linux":
            assert rss is not None and rss > 0
        else:
            # No cheap /proc equivalent elsewhere; None is the contract.
            assert rss is None or rss > 0

    def test_returns_none_when_proc_is_unreadable(self):
        with patch("builtins.open", side_effect=OSError("no /proc")):
            assert _current_rss_bytes() is None

    def test_returns_none_on_unparseable_statm(self):
        handle = MagicMock()
        handle.read.return_value = "garbage"
        handle.__enter__ = lambda s: handle
        handle.__exit__ = lambda *a: False
        with patch("builtins.open", return_value=handle):
            assert _current_rss_bytes() is None


class TestFormatProcStats:
    def test_reports_current_and_peak_separately(self):
        with patch("resource.getrusage", return_value=_fake_rusage(_native_peak_units(8 * 1024**3))):
            with patch("hindsight_api.worker.poller._current_rss_bytes", return_value=1024**3):
                out = _format_proc_stats(object())

        assert out == "rss_mb=1024 peak_rss_mb=8192"

    def test_current_rss_is_not_the_peak(self):
        """The regression itself: a process that burst to 10GB and released it
        must report ~1GB current, not 10GB."""
        with patch("resource.getrusage", return_value=_fake_rusage(_native_peak_units(10 * 1024**3))):
            with patch("hindsight_api.worker.poller._current_rss_bytes", return_value=1024**3):
                out = _format_proc_stats(object())

        assert "rss_mb=1024 " in out
        assert "peak_rss_mb=10240" in out

    def test_omits_current_when_platform_cannot_supply_it(self):
        """Without a current reading we report the peak under its own name rather
        than passing it off as ``rss_mb``."""
        with patch("resource.getrusage", return_value=_fake_rusage(_native_peak_units(2 * 1024**3))):
            with patch("hindsight_api.worker.poller._current_rss_bytes", return_value=None):
                out = _format_proc_stats(object())

        assert out == "peak_rss_mb=2048"
        # No bare `rss_mb=` field — only the peak, under its own name.
        assert "rss_mb=" not in out.replace("peak_rss_mb=", "")

    def test_unavailable_when_introspection_fails(self):
        with patch("resource.getrusage", side_effect=RuntimeError("no rusage")):
            assert _format_proc_stats(object()) == "unavailable"
