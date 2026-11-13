"""Multimodal chunking: a file becomes passages that are embedded as they are, never OCR'd.

* PDF   -> one rendered page image per passage; its text layer is kept for display and embedded
           alongside (its units and sentences, see ``KnowledgeService._vectors_to_embed``)
* image -> the image itself
* audio -> fixed windows (m4a), with a small overlap
* video -> fixed windows (mp4, picture + soundtrack), with a small overlap
* text  -> token-sized text passages; Office/HTML files are read for their text layer first

Audio and video go through the ``ffmpeg`` binary, which must be on PATH.
"""

from __future__ import annotations

import collections
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
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

    def __post_init__(self) -> None:
        # Postgres text cannot hold NUL, and one NUL failed every document in the same ingest
        # batch (BRIGHT-Pro economics lost 100). Every passage is built here, so strip it once.
        for name in ("text", "embed_text"):
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


def split_image(data: bytes) -> list[PassageDraft]:
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
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
        return [PassageDraft(modality="image", data=data, mime=mime, page=1)]
    image = image.convert("RGB")
    image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    return [PassageDraft(modality="image", data=_png(image), mime="image/png", page=1)]


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


def split_pdf(data: bytes, *, page_pixels: int, max_pages: int, bands: bool = False) -> Iterator[PassageDraft]:
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
        number = 0
        while number < page_count or pending:
            while number < page_count and len(pending) < _PDF_LOOKAHEAD:
                with _PDFIUM_LOCK:
                    page = pdf[number]
                    width, height = page.get_size()
                    scale = (max(page_pixels, PDF_BAND_PIXELS) if bands else page_pixels) / max(width, height, 1)
                    sharp = page.render(scale=scale).to_pil().convert("RGB")
                    page.close()
                # Bands only when asked: they used to be cut and encoded for every page and then dropped,
                # three PNGs per page for nothing (half of a PDF's ingest time with bands off).
                crops = _PNG_POOL.submit(_band_pngs, sharp) if bands else None
                pending.append((number, _PNG_POOL.submit(_page_png, sharp, page_pixels, bands), crops))
                number += 1
            done, page_png, crops = pending.popleft()
            yield PassageDraft(
                modality="image",
                text=texts[done],
                data=page_png.result(),
                mime="image/png",
                page=done + 1,
                bands=crops.result() if crops is not None else (),
                embed_text=embed_texts[done],
            )
    finally:
        with _PDFIUM_LOCK:
            pdf.close()


def _page_text(pdf: Any, number: int) -> str | None:
    with _PDFIUM_LOCK:
        page = pdf[number]
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


def read_text(data: bytes, filename: str | None) -> str:
    """The text of a text-like file. Office and HTML files are read for their text layer (no OCR)."""
    ext = Path(filename or "").suffix.lower()
    if ext in _OFFICE_EXTENSIONS:
        from markitdown import MarkItDown

        try:
            return MarkItDown(enable_plugins=False).convert_stream(io.BytesIO(data), file_extension=ext).text_content
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


def split_text(text: str, *, tokens: int) -> list[PassageDraft]:
    from .text_passages import split_into_passages

    return [PassageDraft(modality="text", text=chunk) for chunk in split_into_passages(text, passage_size=tokens)]


# --- audio / video -----------------------------------------------------------------------------


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MediaError("audio and video need ffmpeg on the server's PATH")
    return path


def _run(args: list[str], *, stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(args, input=stdin, capture_output=True, check=False)
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-400:]}")
    return proc.stdout


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool


@dataclass(frozen=True)
class Window:
    start: float
    end: float


def probe(path: str) -> MediaInfo:
    """Duration and tracks of a media file."""
    ffprobe = shutil.which("ffprobe") or _ffmpeg().replace("ffmpeg", "ffprobe")
    out = _run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", path])
    info = json.loads(out or b"{}")
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    try:
        duration = float(info.get("format", {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    return MediaInfo(duration=duration, has_video="video" in kinds, has_audio="audio" in kinds)


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
    data: bytes,
    kind: Literal["audio", "video"],
    *,
    filename: str | None,
    window_seconds: int,
    overlap_seconds: int,
    max_seconds: int,
) -> Iterator[PassageDraft]:
    """Cut audio into m4a windows and video into mp4 windows (480p, with sound)."""
    ffmpeg = _ffmpeg()
    with tempfile.TemporaryDirectory(prefix="hs-kb-") as tmp:
        source = str(Path(tmp) / f"source{Path(filename or '').suffix.lower() or '.bin'}")
        Path(source).write_bytes(data)
        info = probe(source)
        duration, has_video, has_audio = info.duration, info.has_video, info.has_audio
        if duration <= 0:
            raise MediaError("could not read the media duration")
        if duration > max_seconds:
            raise MediaError(f"media is {duration:.0f}s long; the limit is {max_seconds}s")
        if kind == "video" and not has_video:
            kind = "audio"
        if kind == "audio" and not has_audio:
            raise MediaError("the file has no audio track")
        for w in windows(duration, window_seconds, min(overlap_seconds, window_seconds - 1)):
            start, end = w.start, w.end
            out = str(Path(tmp) / f"w.{'mp4' if kind == 'video' else 'm4a'}")
            args = [ffmpeg, "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", source]
            if kind == "video":
                args += ["-vf", "scale=-2:'min(480,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]
                args += ["-pix_fmt", "yuv420p"]
                args += ["-c:a", "aac", "-b:a", "64k", "-ac", "1"] if has_audio else ["-an"]
            else:
                args += ["-vn", "-c:a", "aac", "-b:a", "64k", "-ac", "1"]
            _run([*args, "-movflags", "+faststart", out])
            yield PassageDraft(
                modality=kind,
                data=Path(out).read_bytes(),
                mime="video/mp4" if kind == "video" else "audio/mp4",
                start_ms=int(start * 1000),
                end_ms=int(end * 1000),
            )


def _with_temp_file(data: bytes, suffix: str, args_for: list[str]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        return _run([_ffmpeg(), "-loglevel", "error", "-i", f.name, *args_for, "-"])


def decode_audio_pcm(data: bytes, *, required: bool = True) -> np.ndarray | None:
    """Mono 16 kHz float32 samples of a clip's soundtrack."""
    try:
        raw = _with_temp_file(data, ".mp4", ["-vn", "-ac", "1", "-ar", str(PCM_RATE), "-f", "f32le"])
    except MediaError:
        if required:
            raise
        return None
    return np.frombuffer(raw, np.float32).copy()


def decode_video_frames(data: bytes, *, fps: int = 1, side: int = 448) -> np.ndarray:
    """(T, H, W, 3) uint8 frames at ``fps``, scaled to a ``side`` x ``side`` square."""
    raw = _with_temp_file(
        data, ".mp4", ["-vf", f"fps={fps},scale={side}:{side}", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    )
    frames = np.frombuffer(raw, np.uint8)
    if frames.size == 0:
        raise MediaError("the video has no frames")
    return frames.reshape(-1, side, side, 3)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
