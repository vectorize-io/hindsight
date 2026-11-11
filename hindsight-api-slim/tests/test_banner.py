"""Tests for the startup banner.

The banner is the first thing a user sees, and the tagline in it is the
project's public one-liner — it should not drift from the description on the
repository by accident. The gradient wraps every character in an ANSI escape,
so these strip the escapes and assert on the text that actually reads.
"""

import importlib
import io
import re
import sys

import pytest

from hindsight_api.banner import (
    GRADIENT_END,
    GRADIENT_START,
    gradient_text,
    print_banner,
)
from hindsight_api.utils import configure_stdio

TAGLINE = "Hindsight: Agent Memory That Learns"

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class TestGradientText:
    def test_plain_text_survives_the_gradient(self):
        assert strip_ansi(gradient_text(TAGLINE)) == TAGLINE

    def test_spans_the_full_gradient(self):
        rendered = gradient_text("abc")
        r, g, b = GRADIENT_START
        assert f"\033[38;2;{r};{g};{b}m" in rendered
        r, g, b = GRADIENT_END
        assert f"\033[38;2;{r};{g};{b}m" in rendered

    def test_spaces_are_not_colored(self):
        # Spaces are emitted bare, so a run of them costs no escape sequences.
        assert "  " in gradient_text("a  b")

    def test_empty_string_does_not_divide_by_zero(self):
        assert strip_ansi(gradient_text("")) == ""


class TestPrintBanner:
    def test_prints_the_tagline(self, capsys):
        print_banner()
        assert TAGLINE in strip_ansi(capsys.readouterr().out)

    def test_prints_the_logo_above_the_tagline(self, capsys):
        print_banner()
        out = strip_ansi(capsys.readouterr().out)
        # The logo is drawn with half-block characters.
        assert "▄" in out
        assert out.index("▄") < out.index(TAGLINE)


class TestRedirectedStdout:
    """A stdout whose encoding has no block characters must not stop the server (#6458).

    Before Python 3.15, ``hindsight-api > log`` on Windows writes in the ANSI code page.
    """

    @staticmethod
    def _stream(encoding: str) -> io.TextIOWrapper:
        return io.TextIOWrapper(io.BytesIO(), encoding=encoding)

    def test_banner_prints_after_configure_stdio(self, monkeypatch):
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        out = self._stream("cp1252")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", self._stream("cp949"))

        configure_stdio()
        print_banner()

        out.flush()
        text = out.buffer.getvalue().decode("utf-8")
        assert "▄" in text
        assert TAGLINE in strip_ansi(text)

    def test_operator_encoding_is_kept_and_never_raises(self, monkeypatch):
        monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
        out = self._stream("cp1252")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", self._stream("cp1252"))

        configure_stdio()
        print_banner()

        out.flush()
        text = out.buffer.getvalue().decode("cp1252")
        assert out.encoding == "cp1252"
        assert "\\u2584" in text  # escaped, not dropped and not raised
        assert TAGLINE in strip_ansi(text)


class _ConfiguredStdio(Exception):
    pass


@pytest.mark.parametrize("module", ["hindsight_api.main", "hindsight_api.worker.main", "hindsight_api.admin.cli"])
def test_every_entry_point_configures_stdio_first(monkeypatch, module):
    """Each console script must fix its streams before it prints anything (#6458)."""
    mod = importlib.import_module(module)

    def stop() -> None:
        raise _ConfiguredStdio

    monkeypatch.setattr(mod, "configure_stdio", stop)
    with pytest.raises(_ConfiguredStdio):
        mod.main()
