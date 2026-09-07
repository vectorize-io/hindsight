"""Tests for the startup banner.

The banner is the first thing a user sees, and the tagline in it is the
project's public one-liner — it should not drift from the description on the
repository by accident. The gradient wraps every character in an ANSI escape,
so these strip the escapes and assert on the text that actually reads.
"""

import re

from hindsight_api.banner import (
    GRADIENT_END,
    GRADIENT_START,
    gradient_text,
    print_banner,
)

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
