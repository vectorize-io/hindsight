"""The pictures the corpus refers to, drawn rather than committed.

Every image is generated from a typed spec, which matters for three reasons:

- the spec *is* the ground truth. What the picture says and what the corpus
  expects to be recalled come from the same object, so they cannot drift;
- the bytes are deterministic, so re-running the benchmark re-ingests the same
  content-addressed images and measures the pipeline rather than the weather;
- nothing binary enters the repository.

The renders are deliberately plain — high contrast, large type, the visual
grammar of a screenshot or a small diagram. The benchmark asks whether
information in an image reaches memory at all; it is not an OCR stress test, and
a hard-to-read image would measure the wrong thing.
"""

from __future__ import annotations

import io

from pydantic import BaseModel, Field

_BACKGROUND = (243, 244, 246)
_CHROME = (31, 41, 55)
_ACCENT = (37, 99, 235)
_TEXT = (17, 24, 39)
_MUTED = (75, 85, 99)
_WHITE = (255, 255, 255)


class ImageSpec(BaseModel):
    """What one image shows — and therefore what memory must come to know.

    ``facts`` is the whole point: each entry is something a reader learns *only*
    by looking, which is what the text-only baseline cannot possibly recover.
    """

    name: str
    kind: str = Field(description="'screenshot' or 'diagram'.")
    title: str
    primary: str = Field(description="The largest label — a button caption or the diagram's subject.")
    lines: list[str] = Field(default_factory=list, description="Supporting labels drawn under the title.")
    facts: list[str] = Field(description="Claims true of this image and stated nowhere in the article's text.")


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 takes no size
        return ImageFont.load_default()


def _centred(draw, box: tuple[int, int, int, int], text: str, font, fill) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((left + right - (bounds[2] - bounds[0])) / 2, (top + bottom - (bounds[3] - bounds[1])) / 2),
        text,
        font=font,
        fill=fill,
    )


def render(spec: ImageSpec) -> bytes:
    """Draw ``spec`` as a PNG."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (720, 420), _BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font, body_font, primary_font = _font(26), _font(22), _font(34)

    draw.rectangle([0, 0, 720, 60], fill=_CHROME)
    draw.text((22, 16), spec.title, font=title_font, fill=_WHITE)

    for index, line in enumerate(spec.lines):
        draw.text((40, 92 + index * 34), line, font=body_font, fill=_MUTED)

    top = 92 + len(spec.lines) * 34 + 30
    if spec.kind == "screenshot":
        # A button: the label is the load-bearing detail.
        draw.rectangle([200, top, 520, top + 84], fill=_ACCENT)
        _centred(draw, (200, top, 520, top + 84), spec.primary, primary_font, _WHITE)
    else:
        # A node in a flow, with an arrow into it, so it reads as a diagram.
        draw.rectangle([60, top, 300, top + 84], outline=_MUTED, width=3)
        _centred(draw, (60, top, 300, top + 84), "Start", body_font, _TEXT)
        draw.line([310, top + 42, 400, top + 42], fill=_MUTED, width=4)
        draw.polygon([(400, top + 32), (400, top + 52), (420, top + 42)], fill=_MUTED)
        draw.rectangle([430, top, 690, top + 84], outline=_ACCENT, width=4)
        _centred(draw, (430, top, 690, top + 84), spec.primary, body_font, _ACCENT)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
