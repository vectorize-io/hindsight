"""Multimodal chunking: a file becomes passages that are embedded as they are, never OCR'd.

* PDF   -> one rendered page image per passage; its text layer is kept for display and embedded
           alongside (its units and sentences, see ``KnowledgeService._vectors_to_embed``)
* image -> the image itself
* audio -> fixed windows (m4a), with a small overlap
* video -> fixed windows (mp4, picture + soundtrack), with a small overlap
* text  -> token-sized text passages; Office/HTML files are read for their text layer first

Audio and video go through PyAV, whose wheel carries its own ffmpeg libraries. This used to shell out to
the ``ffmpeg``/``ffprobe`` binaries, which every install then had to provide on PATH (an apt package in
the Docker image, a CI step); PyAV keeps the work in-process and needs nothing on the system.
"""

from __future__ import annotations

import collections
import contextlib
import io
import math
import multiprocessing
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

PCM_RATE = 16000  # EmbeddingGemma 2 takes mono 16 kHz audio

# How a file is split (internal). The public vocabulary is MediaType below.
Kind = Literal["pdf", "image", "audio", "video", "text"]
# What a document is, as callers filter and group by it: PDF and Office/HTML files are documents.
MediaType = Literal["document", "image", "audio", "video", "text"]
MEDIA_TYPES: tuple[MediaType, ...] = ("document", "image", "audio", "video", "text")

_TEXT_TYPES = {"application/json", "application/xml", "application/x-yaml", "application/yaml", "application/x-ndjson"}
_OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".epub"}
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".rst"}
_EXT_KIND: dict[str, Kind] = {
    ".pdf": "pdf",
    **dict.fromkeys((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"), "image"),
    **dict.fromkeys((".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"), "audio"),
    **dict.fromkeys((".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"), "video"),
}

# The image formats an embedding API accepts as they are; anything else is re-encoded as PNG.
_PASSTHROUGH_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
_MAX_IMAGE_SIDE = 3072


#: A file to split: its bytes, or (for a stored upload, which may be gigabytes) the path of a local
#: copy, which the readers below open by name and read only as far as they need.
Source = bytes | str


def _readable(source: Source) -> Any:
    """What PyAV, Pillow and pypdfium2 open: the path itself, or the bytes as a file."""
    return source if isinstance(source, str) else io.BytesIO(source)


def _bytes(source: Source) -> bytes:
    return Path(source).read_bytes() if isinstance(source, str) else source


class MediaError(ValueError):
    """A file that cannot be split: unsupported type, unreadable, or over a limit."""


@dataclass(frozen=True)
class PassageDraft:
    modality: Literal["text", "image", "audio", "video"]
    text: str | None = None
    data: bytes | None = None
    mime: str | None = None
    page: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    # A PDF page's overlapping horizontal bands (top, middle, bottom halves), rendered sharper than
    # the page: one vector per page blurs a dense page, its bands keep the detail (PDF_GEMMA.md).
    bands: tuple[bytes, ...] = ()
    # A PDF page's text layer as embedded: ``text`` without the running headers and footers.
    embed_text: str | None = None
    # A text passage's window: the stretch of text around it it is also found by (``split_text``).
    window_text: str | None = None

    def __post_init__(self) -> None:
        # Postgres text cannot hold NUL, and one NUL failed every document in the same ingest
        # batch (BRIGHT-Pro economics lost 100). Every passage is built here, so strip it once.
        for name in ("text", "embed_text", "window_text"):
            value = getattr(self, name)
            if value and "\x00" in value:
                object.__setattr__(self, name, value.replace("\x00", ""))

    @property
    def extension(self) -> str:
        return {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "audio/mp4": "m4a"}.get(
            self.mime or "", (self.mime or "application/octet-stream").split("/")[-1]
        )


def classify(content_type: str | None, filename: str | None) -> Kind | None:
    """What kind of file this is, from its extension first (browsers send generic types), then its type."""
    ext = Path(filename or "").suffix.lower()
    if ext in _EXT_KIND:
        return _EXT_KIND[ext]
    if ext in _TEXT_EXTENSIONS or ext in _OFFICE_EXTENSIONS:
        return "text"
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == "application/pdf":
        return "pdf"
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("audio/"):
        return "audio"
    if ct.startswith("video/"):
        return "video"
    if ct.startswith("text/") or ct in _TEXT_TYPES:
        return "text"
    return None


def media_type_of(kind: Kind, filename: str | None) -> MediaType:
    if kind == "pdf" or Path(filename or "").suffix.lower() in _OFFICE_EXTENSIONS:
        return "document"
    return kind


# --- images ------------------------------------------------------------------------------------


def open_image(data: bytes):
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def sniff_image_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _png(image) -> bytes:
    """Lossless, at zlib's default level. ``optimize=True`` (every filter tried, then level 9) cost
    ~150-190 ms per 1536px page and was 80% of a PDF's ingest, for a file 2% smaller (468 vs 480 KB
    on ViDoRe HR); the default level is the same pixels in ~25 ms. Level 1 was barely faster (22 ms)
    and 33% bigger, which every file write and every embed request then pays for."""
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


# Small sprites (16-64px icons, pixel art) are what the embedder misses: nearest-neighbour
# upscaling to >= 256px took sprite R@1 from 10 to 16 on 5.7k game assets (GAME_ASSETS.md).
_MIN_IMAGE_SIDE = 256


def _contrasting_background(image) -> tuple[int, int, int]:
    """Dark behind light sprites, light behind dark. A fixed white made white particles vanish."""
    pixels = [(r, g, b) for r, g, b, a in image.getdata() if a > 127]
    if not pixels:
        return (128, 128, 128)
    lum = sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels) / len(pixels)
    return (32, 32, 32) if lum > 128 else (224, 224, 224)


def split_image(data: Source, *, max_megapixels: int = 178) -> list[PassageDraft]:
    """``max_megapixels`` is checked from the header, before anything is decoded: an image costs memory by
    its pixels, and a few-MB PNG can declare 30000 x 30000 of them."""
    from PIL import Image

    # Pillow's own bomb guard at ours: it only warns up to twice that, so the check below speaks first.
    Image.MAX_IMAGE_PIXELS = max_megapixels * 1_000_000
    try:
        image = Image.open(_readable(data))
        _check_pixels(image.width, image.height, max_megapixels)
        image.load()
    except MediaError:
        raise
    except Image.DecompressionBombError as exc:  # not a reason to try another decoder
        raise MediaError(
            f"image is too large ({exc}); the limit is {max_megapixels} MP (HINDSIGHT_API_KB_MAX_IMAGE_MEGAPIXELS)"
        ) from exc
    except Exception as exc:
        # Pillow has no HEIC/HEIF decoder (every iPhone photo); ffmpeg has one, and PyAV carries it.
        image = _first_frame(data, max_megapixels)
        if image is None:
            raise MediaError(f"unreadable image: {exc}") from exc
    mime = Image.MIME.get(image.format or "", "")
    transparent = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    small = min(image.size) < _MIN_IMAGE_SIDE
    if transparent or small:
        image = image.convert("RGBA")
        if small:
            factor = -(-_MIN_IMAGE_SIDE // max(1, min(image.size)))
            image = image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)
        background = Image.new("RGBA", image.size, (*_contrasting_background(image), 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
        image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
        return [PassageDraft(modality="image", data=_png(image), mime="image/png", page=1)]
    if mime in _PASSTHROUGH_IMAGE_TYPES and max(image.size) <= _MAX_IMAGE_SIDE:
        return [PassageDraft(modality="image", data=_bytes(data), mime=mime, page=1)]
    image = image.convert("RGB")
    image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    return [PassageDraft(modality="image", data=_png(image), mime="image/png", page=1)]


def _check_pixels(width: int, height: int, max_megapixels: int) -> None:
    if width * height > max_megapixels * 1_000_000:
        raise MediaError(
            f"image is {width}x{height} ({width * height / 1e6:.0f} MP); the limit is {max_megapixels} MP "
            "(HINDSIGHT_API_KB_MAX_IMAGE_MEGAPIXELS)"
        )


def _first_frame(data: Source, max_megapixels: int) -> Any:
    """The first picture ffmpeg decodes from ``data`` as a Pillow image, or None if it decodes none.
    Raises MediaError for one over ``max_megapixels``: this is not a way around the limit."""
    try:
        import av

        with av.open(_readable(data), metadata_errors="replace") as container:
            stream = container.streams.video[0]
            _check_pixels(stream.codec_context.width, stream.codec_context.height, max_megapixels)
            return next(container.decode(stream)).to_image()
    except MediaError:
        raise
    except Exception:
        return None


# --- PDF ---------------------------------------------------------------------------------------


# pdfium is not thread-safe: concurrent ingests on worker threads crashed the process (SIGABRT,
# exit 134) as soon as several PDFs were claimed at once. Every pdfium call goes through this lock;
# it is released between pages (never held across a yield), so PDFs still interleave page by page.
# Reentrant: an ingest that fails mid-file abandons its page generator, whose ``finally`` (closing the
# PDF, under this lock) then runs whenever the garbage collector gets to it — possibly on a thread
# already inside this lock. With a plain Lock that thread waited on itself forever and every ingest
# behind it hung (seen after embedder outages). The service also closes the generator itself.
_PDFIUM_LOCK = threading.RLock()

# PDF page recipe (PDF_GEMMA.md, EmbeddingGemma 2 on ViDoRe V3, +7 nDCG@10 over one blended vector).
PDF_PAGE_IMAGE_TOKENS = 560
PDF_BAND_IMAGE_TOKENS = 280
PDF_BAND_PIXELS = 2048
PDF_TEXT_UNIT_TOKENS = 128
_PDF_BANDS = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))


_PDF_LOOKAHEAD = 4
_PNG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb-png")


def _page_png(sharp: Any, page_pixels: int, bands: bool) -> bytes:
    image = sharp.copy() if bands else sharp  # with bands the full-size render is still being cropped
    image.thumbnail((page_pixels, page_pixels))
    return _png(image)


def _band_pngs(sharp: Any) -> tuple[bytes, ...]:
    return tuple(
        _png(sharp.crop((0, int(a * sharp.height), sharp.width, int(b * sharp.height)))) for a, b in _PDF_BANDS
    )


def split_pdf(data: Source, *, page_pixels: int, max_pages: int, bands: bool = False) -> Iterator[PassageDraft]:
    """One PNG per page, the longest side rendered at ``page_pixels``."""
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as exc:
            raise MediaError(f"unreadable PDF: {exc}") from exc
        page_count = len(pdf)
    try:
        if page_count > max_pages:
            raise MediaError(f"PDF has {page_count} pages; the limit is {max_pages}")
        # The pages' own text layers, when they have one (not OCR: a scan has none). A page is
        # scored on it too (whole page and 128-token units), and it is what a reader gets back.
        texts = [_page_text(pdf, number) for number in range(page_count)]
        embed_texts = strip_repeated_lines(texts)
        # Rendering goes through pdfium, one page at a time under its lock; encoding the PNGs does not
        # (Pillow lets go of the GIL), so it runs on a few threads while the next pages render. Page by
        # page in order it was half of a PDF's ingest; up to _PDF_LOOKAHEAD pages are held at once.
        pending: collections.deque[tuple[int, Future[bytes], Future[tuple[bytes, ...]] | None]] = collections.deque()
        number, rendered = 0, 0
        while number < page_count or pending:
            while number < page_count and len(pending) < _PDF_LOOKAHEAD:
                with _PDFIUM_LOCK:
                    try:
                        page = pdf[number]
                        width, height = page.get_size()
                        scale = (max(page_pixels, PDF_BAND_PIXELS) if bands else page_pixels) / max(width, height, 1)
                        sharp = page.render(scale=scale).to_pil().convert("RGB")
                        page.close()
                    except Exception:  # noqa: BLE001 - see below
                        # A damaged page (a broken page tree: "Failed to load page") is skipped, so the rest
                        # of the PDF stays searchable; raised, it was retried three times and lost them all.
                        number += 1
                        continue
                # Bands only when asked: they used to be cut and encoded for every page and then dropped,
                # three PNGs per page for nothing (half of a PDF's ingest time with bands off).
                crops = _PNG_POOL.submit(_band_pngs, sharp) if bands else None
                pending.append((number, _PNG_POOL.submit(_page_png, sharp, page_pixels, bands), crops))
                number += 1
            if not pending:
                break
            done, page_png, crops = pending.popleft()
            rendered += 1
            yield PassageDraft(
                modality="image",
                text=texts[done],
                data=page_png.result(),
                mime="image/png",
                page=done + 1,
                bands=crops.result() if crops is not None else (),
                embed_text=embed_texts[done],
            )
        if not rendered:
            raise MediaError("no page of the PDF could be rendered")
    finally:
        with _PDFIUM_LOCK:
            pdf.close()


def _page_text(pdf: Any, number: int) -> str | None:
    with _PDFIUM_LOCK:
        try:
            page = pdf[number]
        except Exception:  # noqa: BLE001 - a page PDFium cannot load has no text; split_pdf skips it
            return None
        textpage = page.get_textpage()
        # PDFium returns a soft hyphen as U+FFFE inside the word ("employ\ufffement"): 40% of ViDoRe hr
        # pages had one, and every such word was lost to keyword search.
        text = textpage.get_text_range().replace("\ufffe", "").strip() or None
        textpage.close()
        page.close()
    return text


def _line_key(line: str) -> str:
    return re.sub(r"[\d\W_]+", " ", line).strip().lower()


def strip_repeated_lines(pages: list[str | None]) -> list[str | None]:
    """Each page without its running headers and footers: lines (digits ignored, so page numbers
    match) found on at least 3 pages and 30% of the document. A deck's title on every slide
    otherwise dominates every page's text vector (PDF_GEMMA.md round 5: physics +0.6)."""
    seen = Counter(key for text in pages if text for key in {_line_key(line) for line in text.splitlines()})
    limit = max(3, 0.3 * len(pages))
    drop = {key for key, count in seen.items() if count >= limit}
    out: list[str | None] = []
    for text in pages:
        kept = "\n".join(line for line in (text or "").splitlines() if _line_key(line) not in drop).strip()
        out.append(kept or None)
    return out


# --- text --------------------------------------------------------------------------------------


#: Office files convert in a worker process, one at a time. They take about 100x their size in memory (a 27 MB
#: .xls peaked at 3.2 GB: the whole sheet as a table, then as one Markdown string), so concurrent conversions ran
#: a 6 GB server out of memory; and the converter is pure Python, so in a thread it held the GIL for minutes and
#: the API answered nobody. A worker the kernel kills for its memory fails that file, not the server.
_OFFICE_POOL: ProcessPoolExecutor | None = None
_OFFICE_POOL_LOCK = threading.Lock()


def _first_to_kill() -> None:
    """In the worker: when memory runs out, the kernel kills this process, not the server it works for (it
    picked the server once, a 2.5 GB process over a 3 GB worker). Linux only; anywhere else nothing to do."""
    with contextlib.suppress(OSError):
        Path("/proc/self/oom_score_adj").write_text("1000")


def _convert_office(data: bytes, ext: str) -> str:
    from markitdown import MarkItDown

    try:
        return MarkItDown(enable_plugins=False).convert_stream(io.BytesIO(data), file_extension=ext).text_content
    except Exception as exc:
        # Back to the server as text: markitdown's exceptions carry tracebacks, which do not pickle, and the
        # reason reached the document as "cannot pickle 'traceback' object".
        raise ValueError(str(exc)) from None


def _office_text(data: bytes, ext: str) -> str:
    global _OFFICE_POOL
    with _OFFICE_POOL_LOCK:
        if _OFFICE_POOL is None:
            # spawn, not fork: the server has threads, and a forked child inherits their locks held.
            _OFFICE_POOL = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_first_to_kill
            )
        pool = _OFFICE_POOL
    try:
        return pool.submit(_convert_office, data, ext).result()
    except BrokenProcessPool as exc:
        with _OFFICE_POOL_LOCK:
            if _OFFICE_POOL is pool:
                _OFFICE_POOL = None  # the next file gets a fresh worker
        raise MediaError(f"the {ext} converter died on this file (out of memory?)") from exc


def read_text(source: Source, filename: str | None) -> str:
    """The text of a text-like file. Office and HTML files are read for their text layer (no OCR)."""
    # ponytail: a text file is read whole (its passages are all held anyway); the size limit bounds it.
    data = _bytes(source)
    ext = Path(filename or "").suffix.lower()
    if ext in _OFFICE_EXTENSIONS:
        try:
            return _office_text(data, ext)
        except Exception as exc:
            raise MediaError(f"unreadable {ext} file: {exc}") from exc
    return data.decode("utf-8", errors="replace")


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\u201c\"])")
#: A sentence shorter than this joins the next one: a heading or a figure label alone matches everything.
_MIN_SENTENCE_WORDS = 8


def sentences(text: str) -> list[str]:
    """A page's text layer as sentences: line breaks of print joined (and words hyphenated across them),
    split after ., ! or ? before a capital or a digit, short pieces merged forward."""
    flat = re.sub(r"\s*\n\s*", " ", re.sub(r"-\n(?=[a-z])", "", text))
    out: list[str] = []
    current = ""
    for part in _SENTENCE_END.split(flat):
        current = f"{current} {part}".strip() if current else part.strip()
        if len(current.split()) >= _MIN_SENTENCE_WORDS:
            out.append(current)
            current = ""
    if current:
        if out:
            out[-1] = f"{out[-1]} {current}"
        else:
            out.append(current)
    return out


def pack_lines(text: str, *, tokens: int) -> list[str]:
    """A PDF page's text layer as pieces of whole lines, packed greedily up to ``tokens``, no overlap;
    a line longer than that is cut by words. What PDF_GEMMA.md benchmarked: a page's text is one line
    per row of print, so the paragraph splitter of text files would cut it differently (-0.7 nDCG@10)."""
    from ..engine.token_encoding import count_tokens

    pieces: list[str] = []
    for line in (raw.strip() for raw in text.replace("\r", "").split("\n")):
        if not line:
            continue
        if count_tokens(line) <= tokens:
            pieces.append(line)
            continue
        words, current = line.split(), []
        for word in words:
            if current and count_tokens(" ".join([*current, word])) > tokens:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
    out: list[str] = []
    current_lines: list[str] = []
    for piece in pieces:
        if current_lines and count_tokens("\n".join([*current_lines, piece])) > tokens:
            out.append("\n".join(current_lines))
            current_lines = []
        current_lines.append(piece)
    if current_lines:
        out.append("\n".join(current_lines))
    return out


#: A text passage is also found by its window: the consecutive passages around it, up to this many tokens,
#: embedded once as one more vector of each. A small passage alone lost what its page says about it (a
#: table's rows without the table's title, or the company named only in the footer); the window keeps it,
#: and results stay the small passages. ViDoRe V3 markdown (own language, EmbeddingGemma 2), nDCG@10 on
#: hr / finance_en / industrial / pharmaceuticals: no window 60.03 / 54.12 / 50.73 / 61.71, 1024 tokens
#: 61.12 / 59.74 / 51.64 / 62.18, 2048 tokens 61.08 / 60.95 / 51.64 / 62.24 (a dense table page outgrew 1024).
TEXT_WINDOW_TOKENS = 2048


def split_text(text: str, *, tokens: int) -> list[PassageDraft]:
    from ..engine.token_encoding import count_tokens
    from .text_passages import split_into_passages

    chunks = split_into_passages(text, passage_size=tokens)
    drafts: list[PassageDraft] = []
    window: list[str] = []
    window_tokens = 0

    def close() -> None:
        # A window of one passage is that passage: no second vector.
        joined = "\n\n".join(window) if len(window) > 1 else None
        drafts.extend(PassageDraft(modality="text", text=chunk, window_text=joined) for chunk in window)
        window.clear()

    for chunk in chunks:
        n = count_tokens(chunk)
        if window and window_tokens + n > TEXT_WINDOW_TOKENS:
            close()
            window_tokens = 0
        window.append(chunk)
        window_tokens += n
    close()
    return drafts


# --- audio / video -----------------------------------------------------------------------------
# PyAV carries its own ffmpeg libraries inside the wheel: nothing runs as a separate program and
# nothing is needed on the system.


def _open(data: Source) -> Any:
    try:
        import av
    except ImportError as exc:
        raise MediaError("audio and video need the local-ml extra (PyAV)") from exc
    try:
        return av.open(_readable(data), metadata_errors="replace")
    except av.FFmpegError as exc:
        raise MediaError(f"unreadable media: {exc}") from exc


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool


@dataclass(frozen=True)
class Window:
    start: float
    end: float


def probe(data: Source) -> MediaInfo:
    """Duration and tracks of a media file."""
    with _open(data) as container:
        duration = container.duration / 1_000_000 if container.duration else 0.0
        if not duration:
            duration = max(
                (float(s.duration * s.time_base) for s in container.streams if s.duration and s.time_base),
                default=0.0,
            )
        return MediaInfo(
            duration=duration, has_video=bool(container.streams.video), has_audio=bool(container.streams.audio)
        )


def windows(duration: float, window: float, overlap: float) -> list[Window]:
    """[start, end) windows covering ``duration``; a short tail is folded into the last window."""
    if duration <= 0:
        return []
    if duration <= window:
        return [Window(0.0, duration)]
    step = max(window - overlap, 1.0)
    starts = [i * step for i in range(math.ceil((duration - overlap) / step))]
    out = [Window(s, min(s + window, duration)) for s in starts if s < duration]
    # A tail shorter than a third of a window carries too little to stand alone.
    if len(out) > 1 and out[-1].end - out[-1].start < window / 3:
        out[-2] = Window(out[-2].start, out[-1].end)
        out.pop()
    return out


def split_media(
    data: Source,
    kind: Literal["audio", "video"],
    *,
    window_seconds: int,
    overlap_seconds: int,
    max_seconds: int,
) -> Iterator[PassageDraft]:
    """Cut audio into m4a windows and video into mp4 windows (480p, with sound)."""
    info = probe(data)
    if info.duration <= 0:
        raise MediaError("could not read the media duration")
    if info.duration > max_seconds:
        raise MediaError(f"media is {info.duration:.0f}s long; the limit is {max_seconds}s")
    if kind == "video" and not info.has_video:
        kind = "audio"
    if kind == "audio" and not info.has_audio:
        raise MediaError("the file has no audio track")
    import av

    # A damaged file still gives what decodes, as ffmpeg itself does: a window with nothing in it, or
    # one the codec refuses, is skipped. Only a file with no window left fails, and for good (a
    # MediaError): an ffmpeg error escaping here was retried as if it were transient, three times.
    cut_any, last_error = False, None
    for w in windows(info.duration, window_seconds, min(overlap_seconds, window_seconds - 1)):
        try:
            clip = _cut(data, w, video=kind == "video", audio=info.has_audio)
        except av.FFmpegError as exc:
            clip, last_error = None, exc
        except Exception as exc:  # noqa: BLE001 - a codec PyAV only half supports (MPEG-H audio, uncompressed
            # video) has no codec context, and fails with an AttributeError on every attempt: cutting is local
            # and deterministic, so a retry only repeats it.
            clip, last_error = None, exc
        if clip is None:
            continue
        cut_any = True
        yield PassageDraft(
            modality=kind,
            data=clip,
            mime="video/mp4" if kind == "video" else "audio/mp4",
            start_ms=int(w.start * 1000),
            end_ms=int(w.end * 1000),
        )
    if not cut_any:
        raise MediaError(f"no decodable {kind} in the file" + (f": {last_error}" if last_error else ""))


def _cut(data: Source, w: Window, *, video: bool, audio: bool) -> bytes | None:
    """One window re-encoded as mp4: H.264 at most 480p high, and/or mono 64 kb/s AAC; None if it is empty."""
    # A temp file, not a buffer: "+faststart" moves the index to the front by re-reading the file by name.
    with tempfile.TemporaryDirectory(prefix="hs-kb-") as tmp:
        out = str(Path(tmp) / "clip.mp4")
        if not _encode_window(data, w, out, video=video, audio=audio):
            return None
        return Path(out).read_bytes()


def _decoded(src: Any, streams: list[Any]) -> Iterator[Any]:
    """The frames of ``streams``, past what is damaged: a packet the decoder refuses is dropped, and a
    container that breaks off ends the stream, as ffmpeg's own command line does with them."""
    import av

    try:
        for packet in src.demux(*streams):
            try:
                frames = packet.decode()
            except av.FFmpegError:
                continue
            yield from frames
    except av.FFmpegError:
        return


@contextlib.contextmanager
def _seeked(data: Source, start: float) -> Iterator[Any]:
    """The file opened at ``start`` seconds in.

    A file without an index (an AVI, an mp4 starting on a non-key frame) refuses to seek (EPERM), and
    the container it refused gives no more frames; it is opened again and read from the beginning:
    slower, the same result, since frames before the window are skipped.
    """
    import av

    src = _open(data)
    try:
        try:
            src.seek(int(start * 1_000_000), any_frame=False)
        except av.FFmpegError:
            src.close()
            src = _open(data)
        yield src
    finally:
        src.close()


def _encode_window(data: Source, w: Window, out: str, *, video: bool, audio: bool) -> int:
    """Encode window ``w`` of ``data`` into ``out``; how many frames it wrote (0: nothing decoded there)."""
    from fractions import Fraction

    import av

    with (
        _seeked(data, w.start) as src,
        av.open(out, "w", format="mp4", options={"movflags": "+faststart"}) as dst,
    ):
        streams: list[Any] = []
        v_out = a_out = None
        if video:
            v_in = src.streams.video[0]
            # Even, as the width below: H.264 in yuv420p refuses an odd size ("avcodec_open2 libx264"), and a
            # source 359 pixels high failed every window of its video.
            height = min(480, v_in.codec_context.height) // 2 * 2
            rate = Fraction(v_in.average_rate or 25)
            v_out = dst.add_stream("libx264", rate=rate)
            v_out.codec_context.time_base = 1 / rate
            v_out.height = height
            v_out.width = round(v_in.codec_context.width * height / v_in.codec_context.height / 2) * 2
            v_out.pix_fmt = "yuv420p"
            v_out.options = {"preset": "veryfast", "crf": "28"}
            streams.append(v_in)
        if audio:
            a_in = src.streams.audio[0]
            a_out = dst.add_stream("aac", rate=a_in.codec_context.sample_rate or 44100, layout="mono")
            a_out.bit_rate = 64_000
            streams.append(a_in)
        resampler = (
            av.AudioResampler(format="fltp", layout="mono", rate=a_out.rate, frame_size=1024)  # AAC frame size
            if a_out
            else None
        )
        audio_samples, video_frames, finished = 0, 0, set()
        for frame in _decoded(src, streams):
            t = frame.time
            if t is None or t < w.start:
                continue
            if t >= w.end:
                finished.add(type(frame))
                if len(finished) == len(streams):
                    break
                continue
            if isinstance(frame, av.VideoFrame) and v_out is not None:
                scaled = frame.reformat(width=v_out.width, height=v_out.height, format="yuv420p")
                # Numbered at the output frame rate: source timestamps can repeat or jump.
                scaled.pts, scaled.time_base, video_frames = video_frames, 1 / rate, video_frames + 1
                dst.mux(v_out.encode(scaled))
            elif isinstance(frame, av.AudioFrame) and a_out is not None and resampler is not None:
                frame.pts = None
                for chunk in resampler.resample(frame):
                    chunk.pts, audio_samples = audio_samples, audio_samples + chunk.samples
                    dst.mux(a_out.encode(chunk))
        if a_out is not None and resampler is not None:
            for chunk in resampler.resample(None):
                chunk.pts, audio_samples = audio_samples, audio_samples + chunk.samples
                dst.mux(a_out.encode(chunk))
        for s in (v_out, a_out):
            if s is not None:
                dst.mux(s.encode(None))
        return video_frames + audio_samples


def decode_audio_pcm(data: bytes, *, required: bool = True) -> np.ndarray | None:
    """Mono 16 kHz float32 samples of a clip's soundtrack."""
    import av

    with _open(data) as container:
        if not container.streams.audio:
            if required:
                raise MediaError("the file has no audio track")
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=PCM_RATE)
        chunks = [
            out.to_ndarray().reshape(-1) for frame in container.decode(audio=0) for out in resampler.resample(frame)
        ]
        chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def decode_video_frames(data: bytes, *, fps: int = 1, side: int = 448) -> np.ndarray:
    """(T, H, W, 3) uint8 frames at ``fps``, scaled to a ``side`` x ``side`` square."""
    frames, next_time = [], 0.0
    with _open(data) as container:
        if not container.streams.video:
            raise MediaError("the video has no frames")
        for frame in container.decode(video=0):
            if frame.time is not None and frame.time + 1e-6 < next_time:
                continue
            frames.append(frame.reformat(width=side, height=side, format="rgb24").to_ndarray())
            next_time = (frame.time or next_time) + 1 / fps
    if not frames:
        raise MediaError("the video has no frames")
    return np.stack(frames)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


#: Longest side of the picture a decision model judges a hit by: the size ViDoRe V3 was measured at.
#: System One takes up to 1568 (``JEV_IMAGE_SIDE``); a smaller picture is also a cheaper request.
JUDGE_SIDE = 1024


def query_picture(data: bytes, low: int, high: int) -> bytes | None:
    """:func:`judge_picture` of a search's query image, enlarged (aspect kept) so its short side is ``low``
    when smaller, as one picture costs nothing to enlarge; None when its long side would pass ``high``."""
    from PIL import Image

    image = Image.open(io.BytesIO(judge_picture(data)))
    short = min(image.size)
    if short < low:
        scale = low / short
        size = (max(low, round(image.width * scale)), max(low, round(image.height * scale)))
        if max(size) > high:
            return None
        image = image.resize(size, Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()


def judge_picture(data: bytes) -> bytes:
    """A JPEG a decision model judges an image or PDF page hit by, at most ``JUDGE_SIDE`` a side."""
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((JUDGE_SIDE, JUDGE_SIDE))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()
