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

import contextlib
import functools
import io
import logging
import math
import multiprocessing
import os
import re
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

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


class MediaWorkError(MediaError):
    """A video that took more work to cut than its limit allows: the whole file fails, not one window."""


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


def _page_png(sharp: Any, page_pixels: int, bands: bool) -> bytes:
    image = sharp.copy() if bands else sharp  # with bands the full-size render is still being cropped
    image.thumbnail((page_pixels, page_pixels))
    return _png(image)


def _band_pngs(sharp: Any) -> tuple[bytes, ...]:
    return tuple(
        _png(sharp.crop((0, int(a * sharp.height), sharp.width, int(b * sharp.height)))) for a, b in _PDF_BANDS
    )


@dataclass(frozen=True)
class _PdfText:
    page_count: int
    #: Each page's own text layer, or None (a scan, or a page that cannot load).
    texts: list[str | None]


@dataclass(frozen=True)
class _RenderedPage:
    number: int
    png: bytes
    bands: tuple[bytes, ...]


#: Pages rendered per call to a PDF worker: few enough to hold, enough to keep the call overhead small.
_PDF_CHUNK = 8
#: Processes rendering PDFs at once. Each is single-threaded (pdfium is not thread-safe), so PDFs of
#: different ingests render in parallel instead of page by page under one process-wide lock.
_PDF_WORKERS = 2
_PDF_POOL: ProcessPoolExecutor | None = None
_PDF_POOL_LOCK = threading.Lock()


#: The longest one call to a PDF worker may take (a PDF's page text, or 8 of its pages rendered). pdf.js's
#: tiling-pattern-tiny-steps*.pdf kept both workers at 100% CPU for over 10 minutes, and every PDF of every
#: ingest waited behind them.
_PDF_CALL_TIMEOUT = 180.0


def _kill(pool: ProcessPoolExecutor) -> None:
    """Stop ``pool`` now, its running calls too: ProcessPoolExecutor has no way to cancel a call that has
    started, so its processes are killed (``_processes``, which it keeps privately)."""
    for process in list((getattr(pool, "_processes", None) or {}).values()):
        process.kill()
    pool.shutdown(wait=False, cancel_futures=True)


def _in_pdf_worker(fn: Callable[..., _T], *args: Any) -> _T:
    """``fn(*args)`` in a PDF worker process. pdfium aborts the process on some files (a JPEG 2000 image in
    pdf.js's jpx_numcomps.pdf: SIGTRAP), and in the server that took the API down for everyone; in a worker
    it fails that file, and the next one gets a fresh worker. A call past ``_PDF_CALL_TIMEOUT`` fails its
    file too, and its worker is killed."""
    global _PDF_POOL
    with _PDF_POOL_LOCK:
        if _PDF_POOL is None:
            _PDF_POOL = _worker_pool(_PDF_WORKERS)
        pool = _PDF_POOL
    try:
        return pool.submit(fn, *args).result(timeout=_PDF_CALL_TIMEOUT)
    except TimeoutError:
        with _PDF_POOL_LOCK:
            if _PDF_POOL is pool:
                _PDF_POOL = None
        _kill(pool)  # the calls beside it break and are retried alone, below
        raise MediaError(f"rendering this PDF took over {_PDF_CALL_TIMEOUT:.0f}s") from None
    except BrokenProcessPool:
        with _PDF_POOL_LOCK:
            if _PDF_POOL is pool:
                _PDF_POOL = None
    # A crash (or a kill) breaks every call in flight on the pool, not only the file's own: once again, alone
    # in a worker of its own, so only the file that crashes it fails (a PDF rendering beside jpx_numcomps.pdf
    # failed with it).
    try:
        return _alone(fn, *args, timeout=_PDF_CALL_TIMEOUT)
    except TimeoutError:
        raise MediaError(f"rendering this PDF took over {_PDF_CALL_TIMEOUT:.0f}s") from None
    except BrokenProcessPool as exc:
        raise MediaError("the PDF renderer crashed on this file") from exc


def _alone(fn: Callable[..., _T], *args: Any, timeout: float | None = None) -> _T:
    """``fn(*args)`` once more, in a worker process of its own, killed after: the retry of a call whose
    shared worker died, so a crash fails the file that caused it and not the files queued beside it.
    Raises ``BrokenProcessPool`` if it dies again, ``TimeoutError`` past ``timeout``."""
    alone = _worker_pool(1)
    try:
        return alone.submit(fn, *args).result(timeout=timeout)
    finally:
        _kill(alone)


def _pdf_text(data: Source, max_pages: int) -> _PdfText:
    """In a worker: the page count and every page's text layer."""
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise MediaError(f"unreadable PDF: {exc}") from None  # pdfium's exceptions carry ctypes state
    try:
        page_count = len(pdf)
        if page_count > max_pages:
            raise MediaError(f"PDF has {page_count} pages; the limit is {max_pages}")
        return _PdfText(page_count, [_page_text(pdf, number) for number in range(page_count)])
    finally:
        pdf.close()


def _pdf_render(data: Source, numbers: list[int], page_pixels: int, bands: bool) -> list[_RenderedPage]:
    """In a worker: pages ``numbers`` rendered and encoded. A damaged page (a broken page tree: "Failed to
    load page") is left out, so the rest of the PDF stays searchable; raised, it was retried three times and
    lost them all."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        out = []
        for number in numbers:
            try:
                page = pdf[number]
                width, height = page.get_size()
                scale = (max(page_pixels, PDF_BAND_PIXELS) if bands else page_pixels) / max(width, height, 1)
                sharp = page.render(scale=scale).to_pil().convert("RGB")
                page.close()
            except Exception:  # noqa: BLE001 - see the docstring
                continue
            # Bands only when asked: they used to be cut and encoded for every page and then dropped,
            # three PNGs per page for nothing (half of a PDF's ingest time with bands off).
            out.append(_RenderedPage(number, _page_png(sharp, page_pixels, bands), _band_pngs(sharp) if bands else ()))
        return out
    finally:
        pdf.close()


def split_pdf(data: Source, *, page_pixels: int, max_pages: int, bands: bool = False) -> Iterator[PassageDraft]:
    """One PNG per page, the longest side rendered at ``page_pixels``. pdfium runs in worker processes
    (``_in_pdf_worker``), a few pages per call."""
    info = _in_pdf_worker(_pdf_text, data, max_pages)
    # The pages' own text layers, when they have one (not OCR: a scan has none). A page is scored on it
    # too (whole page and 128-token units), and it is what a reader gets back.
    embed_texts = strip_repeated_lines(info.texts)
    rendered = 0
    for start in range(0, info.page_count, _PDF_CHUNK):
        numbers = list(range(start, min(start + _PDF_CHUNK, info.page_count)))
        for page in _in_pdf_worker(_pdf_render, data, numbers, page_pixels, bands):
            rendered += 1
            yield PassageDraft(
                modality="image",
                text=info.texts[page.number],
                data=page.png,
                mime="image/png",
                page=page.number + 1,
                bands=page.bands,
                embed_text=embed_texts[page.number],
            )
    if not rendered:
        raise MediaError("no page of the PDF could be rendered")


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


def _container_limit() -> int:
    """The container's memory limit (cgroup v2, then v1); 0 when it has none."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        with contextlib.suppress(OSError, ValueError):
            limit = int(Path(path).read_text().strip())  # "max": no limit, a ValueError
            if limit < 1 << 50:  # cgroup v1 writes "no limit" as a huge number
                return limit
    return 0


def _memory_limit() -> int:
    """The container's memory limit, else the machine's RAM; 0 if unknown."""
    if limit := _container_limit():
        return limit
    with contextlib.suppress(OSError, ValueError, AttributeError):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return 0


def _worker_pool(workers: int) -> ProcessPoolExecutor:
    """A pool of worker processes for files (PDF rendering, Office conversion), memory-capped and guarded."""
    _start_memory_guard()
    # spawn, not fork: the server has threads, and a forked child inherits their locks held.
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_first_to_kill,
        initargs=(_worker_address_space(),),
    )


# --- the memory guard ------------------------------------------------------------------------
# Each worker's own cap is not enough: on Kubernetes (cgroup v2, memory.oom.group = 1) a container that runs
# out of memory has every process killed together, the server and its database too, and the caps of
# several workers add up past the container next to a 2 GB server. On the cluster sweep, 51535.xls (16.7M
# declared cells) grew its converter to its 5 GB cap beside a 3 GB server, and the whole pod restarted.
# So the server watches the container itself, and kills its biggest worker before the kernel kills them
# all; the call it was running is retried alone (_alone), and a file that grows that far again fails.

#: The share of the container's memory in use past which the biggest worker is killed.
_GUARD_SHARE = 0.85
#: How often the guard looks: a runaway conversion grew 14 MB a second, so 0.25 s is 4 MB.
_GUARD_INTERVAL = 0.25
_GUARD_STARTED = False
_GUARD_LOCK = threading.Lock()


def _container_used() -> int:
    """The memory the container holds that the kernel cannot drop: anonymous, shared and kernel memory
    (cgroup v2's memory.stat). Page cache is left out, as it is reclaimed before anything is killed."""
    try:
        stat = Path("/sys/fs/cgroup/memory.stat").read_text()
    except OSError:
        return 0
    fields = dict(line.split(" ", 1) for line in stat.splitlines() if " " in line)
    return sum(int(fields.get(k, 0)) for k in ("anon", "shmem", "kernel"))


def _worker_rss() -> dict[int, int]:
    """Each of this process's worker children (multiprocessing's spawned processes, not its resource
    tracker) by its resident memory in bytes."""
    out: dict[int, int] = {}
    me = os.getpid()
    page = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError, ValueError, IndexError):
            # The parent pid is the 2nd field after the parenthesised command name, which may hold spaces.
            if int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1]) != me:
                continue
            cmdline = (entry / "cmdline").read_bytes()
            if b"multiprocessing.spawn" not in cmdline or b"resource_tracker" in cmdline:
                continue
            out[int(entry.name)] = int((entry / "statm").read_text().split()[1]) * page
    return out


def _guard_once(limit: int) -> int | None:
    """Kill the biggest worker if the container is past ``_GUARD_SHARE`` of ``limit``; the pid killed."""
    used = _container_used()
    if used <= _GUARD_SHARE * limit:
        return None
    workers = _worker_rss()
    if not workers:
        return None
    pid = max(workers, key=workers.__getitem__)
    with contextlib.suppress(OSError):
        os.kill(pid, 9)
        _logger.warning(
            f"Knowledge worker {pid} killed: the container holds {used >> 20} MB of {limit >> 20} MB, "
            f"and the worker {workers[pid] >> 20} MB"
        )
        return pid
    return None


def _start_memory_guard() -> None:
    """Start the guard thread once, in a container with a memory limit (Linux); elsewhere nothing."""
    global _GUARD_STARTED
    with _GUARD_LOCK:
        if _GUARD_STARTED or not (limit := _container_limit()) or not Path("/proc/self/stat").exists():
            return
        _GUARD_STARTED = True

    def watch() -> None:
        while True:
            with contextlib.suppress(Exception):  # the guard must never stop: a read that fails is skipped
                _guard_once(limit)
            time.sleep(_GUARD_INTERVAL)

    threading.Thread(target=watch, name="kb-memory-guard", daemon=True).start()


def _worker_address_space() -> int:
    """The most address space a worker process (Office conversion, PDF rendering) may take, 0 for no cap
    (``HINDSIGHT_API_KB_WORKER_MEMORY_MB``; by default half the container's memory plus 1 GB, a worker's
    own 0.9 GB at rest). POI's 51535.xls, 42 KB, declares 65,536 rows of 256 columns, and markitdown grew by
    14 MB a second turning them into a table until the 8 GB container ran out; on Kubernetes (cgroup v2) the
    kernel then kills every process of the container, the server too, whatever their oom_score_adj. Past
    the cap the worker gets a MemoryError and that file fails alone. A flat 2 GB was tried first: it also
    failed real spreadsheets (Govdocs1's 28 MB 000032.xls peaks at 3.3 GB), so the cap follows the machine."""
    from ..config import get_config

    if (mb := get_config().kb_worker_memory_mb) > 0:
        return mb << 20
    total = _memory_limit()
    return total // 2 + (1 << 30) if total else 0


def _first_to_kill(address_space: int) -> None:
    """In the worker: its address space is capped (``_worker_address_space``), and when memory runs out
    anyway the kernel kills this process, not the server it works for. Linux only; elsewhere a no-op."""
    import resource

    if address_space > 0:
        with contextlib.suppress(OSError, ValueError):
            resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    with contextlib.suppress(OSError):
        Path("/proc/self/oom_score_adj").write_text("1000")


#: The parts of an Office file that hold its text, read when markitdown cannot: the body with its headers,
#: footers and notes (docx), the slides (pptx), the cell strings (xlsx). Any case, either separator: POI's
#: 49609.xlsx names them "xl\\sharedstrings.xml".
_OOXML_TEXT_PARTS = {
    ".docx": re.compile(r"word/(document|header\d*|footer\d*|footnotes|endnotes)\.xml", re.IGNORECASE),
    ".pptx": re.compile(r"ppt/slides/slide\d+\.xml", re.IGNORECASE),
    ".xlsx": re.compile(r"xl/sharedStrings\.xml|xl/worksheets/sheet\d+\.xml", re.IGNORECASE),
}


def _ooxml_text(data: bytes, ext: str) -> str:
    """An Office file's text runs, read from its XML a paragraph per line, without markitdown. A streaming
    parse, so a table nested 15,000 deep (POI's deep-table-cell.docx: RecursionError in markitdown) is read
    like any other; entities stay unresolved (the standard parser fetches nothing)."""
    import xml.etree.ElementTree as ET
    import zipfile

    parts = _OOXML_TEXT_PARTS.get(ext)
    if parts is None:
        return ""
    lines: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted(
            (n for n in z.namelist() if parts.fullmatch(n.replace("\\", "/"))),
            key=lambda n: [int(d) for d in re.findall(r"\d+", n)],
        )
        for name in names:
            line: list[str] = []
            with z.open(name) as part:
                for _event, element in ET.iterparse(part):
                    tag = element.tag.rsplit("}", 1)[-1]
                    if tag == "t" and element.text:
                        line.append(element.text)
                    elif tag in ("p", "si", "row") and line:
                        lines.append("".join(line))
                        line = []
                    element.clear()
            if line:
                lines.append("".join(line))
    return "\n".join(lines)


@functools.cache
def _markitdown() -> Any:
    """The worker's one converter. Each MarkItDown loads its own file-type model (magika, on ONNX Runtime) and
    never gives it back: one per file grew a worker's address space by 90 MB a file, and after ~55 files
    every conversion past the worker's memory cap failed ("ONNXRuntimeError ... std::bad_alloc", or "failed to
    create a thread"): 71 of the cluster sweep's Office files. Reused, the worker stays at 0.77 GB."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False)


#: A legacy .xls declaring more cells than this is read cell by cell (``_xls_text``), not by markitdown: it
#: builds a table of every declared cell, empty ones too, at about 1.7 KB each (Govdocs1's 000032.xls, 1.9M
#: cells, peaked at 3.3 GB). A sheet with whole columns formatted declares 65,536 rows of them: POI's 31 KB
#: SharedFormulaTest.xls (4.3M cells) and 42 KB 51535.xls (16.7M) ran a worker out of 5 GB, and read cell by
#: cell they take 0.1 s and 0.5 s, under 200 MB. Below it, markitdown's tables are kept as they were.
_MARKITDOWN_MAX_CELLS = 2_000_000


def _xls_sheets(data: bytes) -> Iterator[Any]:
    """Each sheet of a legacy .xls, loaded one at a time."""
    import xlrd

    book = xlrd.open_workbook(file_contents=data, on_demand=True)
    try:
        for index in range(book.nsheets):
            yield book.sheet_by_index(index)
            book.unload_sheet(index)
    finally:
        book.release_resources()


def _xls_text(data: bytes) -> str:
    """A legacy .xls's non-empty cells, a row per line as a table row (``| a | b |``, as markitdown writes
    them, so the passage splitter cuts them the same way), a heading per sheet."""
    lines: list[str] = []
    for sheet in _xls_sheets(data):
        lines.append(f"## {sheet.name}")
        for row in range(sheet.nrows):
            values = [str(v).strip() for v in sheet.row_values(row)]
            if any(values):
                lines.append("| " + " | ".join(v for v in values if v) + " |")
    return "\n".join(lines)


def _xls_cells(data: bytes) -> int:
    """How many cells a legacy .xls declares, every sheet's rows times columns; 0 if xlrd cannot open it."""
    try:
        return sum(sheet.nrows * sheet.ncols for sheet in _xls_sheets(data))
    except Exception:  # noqa: BLE001 - unreadable here is markitdown's to report
        return 0


def _fallback_text(data: bytes, ext: str) -> str:
    """The text of an Office file read without markitdown, when it fails or comes back empty."""
    return _xls_text(data) if ext == ".xls" else _ooxml_text(data, ext)


def _convert_office(data: bytes, ext: str) -> str:
    if ext == ".xls" and _xls_cells(data) > _MARKITDOWN_MAX_CELLS:
        return _xls_text(data)
    try:
        text = _markitdown().convert_stream(io.BytesIO(data), file_extension=ext).text_content
    except Exception as exc:
        # markitdown fails on files Office opens: a KeyError on a field it did not expect (POI's 60329.docx),
        # recursion on deep tables, 974 words of slides (2411-Performance_Up.pptx). Their text is still read.
        try:
            if text := _fallback_text(data, ext):
                return text
        except Exception:
            pass
        # Back to the server as text: markitdown's exceptions carry tracebacks, which do not pickle, and the
        # reason reached the document as "cannot pickle 'traceback' object".
        raise ValueError(str(exc)) from None
    if text.strip():
        return text
    # markitdown reads a Word body only: a document whose text is all in its header (POI's Headers.docx)
    # came out empty and failed "nothing to search".
    try:
        return _fallback_text(data, ext)
    except Exception:
        return text


#: One Office conversion at a time in the server, its retry included. A big spreadsheet's conversion holds
#: memory by the 100x its size (Govdocs1's 28 MB 000032.xls peaks at 3.3 GB), so two side by side do not
#: fit where one does: on the cluster, the retries of two broken calls ran at once, 000032.xls beside the
#: runaway 51535.xls, and the memory guard had to kill both. Waiting here, a file is never queued on the
#: worker either, so a worker's death breaks only the file it was converting.
_OFFICE_CALL_LOCK = threading.Lock()


def _office_text(data: bytes, ext: str) -> str:
    with _OFFICE_CALL_LOCK:
        return _office_text_now(data, ext)


def _office_text_now(data: bytes, ext: str) -> str:
    global _OFFICE_POOL
    with _OFFICE_POOL_LOCK:
        if _OFFICE_POOL is None:
            _OFFICE_POOL = _worker_pool(1)
        pool = _OFFICE_POOL
    try:
        return pool.submit(_convert_office, data, ext).result()
    except BrokenProcessPool:
        with _OFFICE_POOL_LOCK:
            if _OFFICE_POOL is pool:
                _OFFICE_POOL = None  # the next file gets a fresh worker
    # Once again in a fresh worker: the memory guard kills the biggest worker when the container runs low,
    # which can be this one for memory the PDF workers hold. A file that kills it twice fails. (Files used
    # to queue on the worker, and a 9 KB spreadsheet behind the runaway 51535.xls failed with it; they now
    # wait on _OFFICE_CALL_LOCK instead.)
    try:
        return _alone(_convert_office, data, ext)
    except BrokenProcessPool as exc:
        raise MediaError(f"the {ext} converter died on this file (out of memory?)") from exc


def read_text(source: Source, filename: str | None) -> str:
    """The text of a text-like file. Office and HTML files are read for their text layer (no OCR)."""
    # ponytail: a text file is read whole (its passages are all held anyway); the size limit bounds it.
    data = _bytes(source)
    ext = Path(filename or "").suffix.lower()
    if ext in _OFFICE_EXTENSIONS:
        try:
            text = _office_text(data, ext)
        except Exception as exc:
            raise MediaError(f"unreadable {ext} file: {exc}") from exc
        # markitdown converts the first <body> only: a Govdocs1 export opens <body> twice, the first time
        # still inside <head> and empty, so a page of 1,500 words came out empty and failed "nothing to
        # search". Only then is its text read without markup.
        if ext in (".html", ".htm") and not text.strip():
            return _html_text(data.decode("utf-8", errors="replace"))
        return text
    return data.decode("utf-8", errors="replace")


def _html_text(html: str) -> str:
    """The visible text of an HTML page, without scripts and styles (the stdlib parser: no tree to misplace)."""
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.chunks: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag in ("script", "style"):
                self._skip += 1

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style") and self._skip:
                self._skip -= 1

        def handle_data(self, data: str) -> None:
            if not self._skip and data.strip():
                self.chunks.append(data.strip())

    parser = _Text()
    parser.feed(html)
    parser.close()
    return "\n".join(parser.chunks)


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
    # A page with no sentence ends (pdf.js's perf_text_labels.pdf: thousands of labels) was one "sentence"
    # of 14k tokens, which the embedder refused (EmbeddingGemma 2 reads 8,192): cut at words.
    return [piece for sentence in out for piece in _cut_words(sentence, _MAX_SENTENCE_TOKENS)]


#: The longest a sentence is embedded as: past it, it is cut at words. A real sentence is far shorter.
_MAX_SENTENCE_TOKENS = 256


def _cut_words(text: str, tokens: int) -> list[str]:
    """``text`` as pieces of whole words of about ``tokens`` at most, counting each word once (linear: a
    piece re-counted per word was quadratic on a 14k-token line)."""
    from ..engine.token_encoding import count_tokens

    if count_tokens(text) <= tokens:
        return [text]
    pieces, current, size = [], [], 0
    for word in text.split():
        n = count_tokens(" " + word)
        if current and size + n > tokens:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += n
    if current:
        pieces.append(" ".join(current))
    return pieces


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
    #: The first video track's frame size and rate as its header declares them (0 when unknown or no video).
    width: int = 0
    height: int = 0
    fps: float = 0.0
    #: False for a file without a seek index: each window is then decoded from the start of the file.
    seekable: bool = True


@dataclass(frozen=True)
class Window:
    start: float
    end: float


def probe(data: Source) -> MediaInfo:
    """Duration, tracks and (for video) frame size, rate and seekability of a media file, from its header."""
    import av

    with _open(data) as container:
        duration = container.duration / 1_000_000 if container.duration else 0.0
        if not duration:
            duration = max(
                (float(s.duration * s.time_base) for s in container.streams if s.duration and s.time_base),
                default=0.0,
            )
        width = height = 0
        fps = 0.0
        if container.streams.video:
            v = container.streams.video[0]
            # No codec context for a codec PyAV does not know (FATE's uncv-rgb24.mp4): reading its width raised
            # an AttributeError, which the worker retried for three minutes before the file failed anyway.
            ctx = v.codec_context
            width, height = (ctx.width or 0, ctx.height or 0) if ctx is not None else (0, 0)
            rate = v.average_rate or v.guessed_rate
            fps = float(rate) if rate else 0.0
        seekable = True
        if duration > 0 and container.streams.video:
            # A file without an index refuses the seek (EPERM), or takes it and goes back to its start (an
            # AVI with no idx1): either way each window is then decoded from the start. The keyframe a seek
            # lands on can be well before the target, but not before half of it.
            target = duration / 2
            try:
                container.seek(int(target * 1_000_000), any_frame=False)
                landed = next(
                    (p.pts * p.time_base for p in container.demux(video=0) if p.pts is not None and p.time_base),
                    None,
                )
                seekable = landed is not None and float(landed) >= target / 2
            except av.FFmpegError:
                seekable = False
        return MediaInfo(
            duration=duration,
            has_video=bool(container.streams.video),
            has_audio=bool(container.streams.audio),
            width=width,
            height=height,
            fps=fps,
            seekable=seekable,
        )


# --- the work of cutting a video ---------------------------------------------------------------
# Cutting costs CPU by frames and their pixels: each frame is decoded at its own size, then scaled and
# re-encoded at 480p. Measured with the cut below (one core, libx264 veryfast): 1080p at 30 fps took
# 13 CPU-seconds per minute of video, 4K at 60 fps 40, so a frame costs a fixed part (the re-encode)
# plus a part by its pixels (the decode), the fixed part as much as decoding ~9.5 megapixels. Neither
# length nor file size says which: a 500 MB file holds two hours of 4K at 60 fps as easily as of 1080p.

#: The per-frame part of the work, in megapixels decoded.
_FRAME_WORK_MEGAPIXELS = 9.5
#: The unit a limit is given in: one second of 1080p at 30 fps.
REFERENCE_SECOND_WORK = 30 * (1 + 1920 * 1080 / 1e6 / _FRAME_WORK_MEGAPIXELS)
#: How far decoding may run past the estimate before the cut stops: a seek lands on the keyframe before a
#: window, and the frames from there to the window's start are decoded too.
_WORK_HEADROOM = 1.5


def _span(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def _frame_work(width: int, height: int) -> float:
    return 1 + width * height / 1e6 / _FRAME_WORK_MEGAPIXELS


def video_work(info: MediaInfo, spans: list[Window]) -> float:
    """The work of cutting ``spans`` of a video, in seconds of 1080p 30 fps video. Each window decodes its
    own span, or everything from the start of the file when the file cannot seek, and overlapping
    windows decode their overlap twice."""
    fps = info.fps or 30.0  # A header without a rate: the decode below counts what is really there.
    decoded = sum(w.end - (w.start if info.seekable else 0.0) for w in spans)
    return decoded * fps * _frame_work(info.width, info.height) / REFERENCE_SECOND_WORK


class _WorkMeter:
    """Counts the frames a cut really decodes, so a header that understates the size or rate cannot
    take more than the limit allows."""

    def __init__(self, limit_seconds: float) -> None:
        self.limit_seconds = limit_seconds
        self._left = limit_seconds * REFERENCE_SECOND_WORK * _WORK_HEADROOM

    def spend(self, width: int, height: int) -> None:
        self._left -= _frame_work(width, height)
        if self._left < 0:
            raise MediaWorkError(
                "the video took more work to cut than its header showed (more frames than it declares, or a "
                f"file that does not seek), past the limit of "
                f"{_span(self.limit_seconds)} of 1080p 30 fps video (HINDSIGHT_API_KB_MAX_VIDEO_WORK_SECONDS)"
            )


def _check_video_work(info: MediaInfo, spans: list[Window], limit_seconds: float) -> None:
    # Before decoding, each window counts its own span, even in a file that does not seek. A header can claim
    # far more than the file holds: a cut-off upload (FATE's 1242-small.mkv, 100 KB) declares 22 minutes, and
    # counted as decoded from its start for every window it was refused as 7.2 h of work, where its few real
    # frames take a second. The meter (_WorkMeter) counts what is really decoded, from the start too, so a
    # long file without an index is still stopped at the limit.
    work = video_work(replace(info, seekable=True), spans)
    if work <= limit_seconds:
        return
    shape = f"{info.width}x{info.height} at {info.fps or 30:.0f} fps for {_span(info.duration)}"
    if not info.seekable:
        hint = "the file has no seek index, so every window is decoded from its start; saving it again as MP4 adds one"
    else:
        fits = info.duration * limit_seconds / work
        hint = f"at this size and frame rate about {_span(fits)} fits; a lower resolution or frame rate fits more"
    raise MediaWorkError(
        f"video is {shape}: cutting it is the work of {_span(work)} of 1080p 30 fps video, and the limit "
        f"is {_span(limit_seconds)} (HINDSIGHT_API_KB_MAX_VIDEO_WORK_SECONDS); {hint}"
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
    max_video_work_seconds: float = 0,
) -> Iterator[PassageDraft]:
    """Cut audio into m4a windows and video into mp4 windows (480p, with sound).

    ``max_video_work_seconds`` bounds the work of cutting a video (``video_work``), checked from the header
    before anything is decoded and counted again while decoding; 0 leaves it unbounded."""
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
    # At most half a window of overlap, so a moment of the file is cut at most twice: with a one-second
    # step, a 1800 s window over two hours of media was cut 5,401 times, 2,700 hours of video.
    spans = windows(info.duration, window_seconds, min(overlap_seconds, window_seconds // 2))
    meter = None
    if kind == "video" and max_video_work_seconds > 0:
        _check_video_work(info, spans, max_video_work_seconds)
        meter = _WorkMeter(max_video_work_seconds)
    for w in spans:
        try:
            clip = _cut(data, w, video=kind == "video", audio=info.has_audio, meter=meter)
        except MediaWorkError:
            raise
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


def _cut(data: Source, w: Window, *, video: bool, audio: bool, meter: _WorkMeter | None = None) -> bytes | None:
    """One window re-encoded as mp4: H.264 at most 480p high, and/or mono 64 kb/s AAC; None if it is empty."""
    # A temp file, not a buffer: "+faststart" moves the index to the front by re-reading the file by name.
    with tempfile.TemporaryDirectory(prefix="hs-kb-") as tmp:
        out = str(Path(tmp) / "clip.mp4")
        if not _encode_window(data, w, out, video=video, audio=audio, meter=meter):
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


def _encode_window(
    data: Source, w: Window, out: str, *, video: bool, audio: bool, meter: _WorkMeter | None = None
) -> int:
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
            if meter is not None and isinstance(frame, av.VideoFrame):
                meter.spend(frame.width, frame.height)  # every frame decoded, before the window too
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
