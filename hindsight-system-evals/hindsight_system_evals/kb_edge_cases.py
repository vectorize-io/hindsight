"""Knowledge-bank edge cases: odd, broken and big files, and what ingesting them costs.

Not a quality eval. The question per file is only: does the server process it, without a 5xx, a
hang or a crash, and can it then be found? A file the server refuses is fine when it says why
(a 4xx, or a failed document with a reason); a file that is ready and comes back from a search
is fine too. Everything else is a bug.

Two ways in, one code path:

* ``harvest`` runs any folder of files through a server it starts (or ``--api-url``) and writes
  one JSON line per file, so a public corpus (pdf.js's test PDFs, Apache POI's test documents,
  FFmpeg's FATE samples, ...) can be swept for what breaks::

      uv run python -m hindsight_system_evals.kb_edge_cases harvest ~/corpus/pdfjs --out pdfjs.jsonl

* the eval (``evals/knowledge``) runs the sample: one file per issue found that way, with what the
  server must do with it, kept in vectorize-io/hindsight-datasets (``broken-media/``) and downloaded
  at a pinned commit (``sample_dir``).

Each file is ingested alone and the server's process tree is sampled meanwhile (peak memory,
CPU seconds), so a file that costs too much shows up as a number, not a guess.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

KB = "/v1/default/knowledge-banks"
#: The sample's files and manifest: ``broken-media/`` of vectorize-io/hindsight-datasets at this commit.
#: They used to be committed here (``fixtures/kb-edge-cases``); binaries live in the datasets repo now.
BROKEN_MEDIA_REV = "b0cb5a34edc763de1150bac2d6f9dd587e9d9f43"
_BROKEN_MEDIA_URL = "https://raw.githubusercontent.com/vectorize-io/hindsight-datasets/{rev}/broken-media/{path}"

#: The extensions a knowledge bank takes (``knowledge/media.py``). Others are rejected with a 415,
#: which is correct and so not worth a harvest's time.
SUPPORTED = {
    ".pdf",
    *(".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"),
    *(".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"),
    *(".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"),
    *(".txt", ".md", ".markdown", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".rst"),
    *(".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".epub"),
}


@dataclass
class Usage:
    start_rss_mb: float
    peak_rss_mb: float
    cpu_seconds: float
    rss_after_mb: float

    @property
    def growth_mb(self) -> float:
        """How far above where it started the server's memory went: what this file cost it."""
        return round(self.peak_rss_mb - self.start_rss_mb, 1)


#: Where the server runs: its process id (one this run started), the name of the docker container it
#: runs in (``--container``: memory capped, so a runaway kills the container, not the machine), a pod
#: container as ``k8s:<namespace>/<pod>/<container>`` (same, on a cluster), or None.
Server = int | str | None


def _exec(server: str) -> list[str]:
    """The command prefix that runs something inside ``server`` (a docker container or a pod's container)."""
    if server.startswith("k8s:"):
        namespace, pod, container = server.removeprefix("k8s:").split("/")
        return ["kubectl", "exec", "-n", namespace, pod, "-c", container, "--"]
    return ["docker", "exec", server]


@dataclass(frozen=True)
class _Sample:
    rss_mb: float
    cpu_seconds: float


def _container_rss_cpu(name: str) -> _Sample:
    """The container's anonymous memory (what its processes hold; not the page cache its temp files fill)
    and CPU seconds, from its cgroup."""
    import subprocess

    try:
        out = subprocess.run(
            [
                *_exec(name),
                "sh",
                "-c",
                "grep '^anon ' /sys/fs/cgroup/memory.stat; grep usage_usec /sys/fs/cgroup/cpu.stat",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.split()
        return _Sample(int(out[1]) / 1048576, int(out[3]) / 1e6)
    except (IndexError, ValueError, OSError, subprocess.SubprocessError):
        return _Sample(0.0, 0.0)


class ResourceMeter:
    """Samples the server (a process and its children, or a container) on a thread."""

    def __init__(self, pid: Server, interval: float = 0.2) -> None:
        self._pid = pid
        self._interval = interval if not isinstance(pid, str) else 1.0  # a docker exec per sample
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_start = 0.0
        self._start = 0.0

    def _tree(self):
        import psutil

        root = psutil.Process(self._pid)
        return [root, *root.children(recursive=True)]

    def _rss_cpu(self) -> _Sample:
        if isinstance(self._pid, str):
            return _container_rss_cpu(self._pid)
        import psutil

        rss = cpu = 0.0
        for p in self._tree():
            try:
                rss += p.memory_info().rss
                t = p.cpu_times()
                cpu += t.user + t.system
            except psutil.Error:
                continue
        return _Sample(rss / 1048576, cpu)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._rss_cpu().rss_mb)
            self._stop.wait(self._interval)

    def __enter__(self) -> ResourceMeter:
        if self._pid is not None:
            first = self._rss_cpu()
            self._start, self._cpu_start = first.rss_mb, first.cpu_seconds
            self._peak = self._start
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def usage(self) -> Usage | None:
        if self._pid is None:
            return None
        now = self._rss_cpu()
        rss, cpu = now.rss_mb, now.cpu_seconds
        return Usage(
            start_rss_mb=round(self._start, 1),
            peak_rss_mb=round(max(self._peak, rss), 1),
            cpu_seconds=round(cpu - self._cpu_start, 2),
            rss_after_mb=round(rss, 1),
        )


@dataclass
class FileOutcome:
    name: str
    document_id: str
    size_bytes: int
    #: ``ready`` | ``failed`` (the document says why) | ``rejected`` (a 4xx at upload) |
    #: ``error`` (a 5xx at upload) | ``timeout`` (never left pending/processing)
    status: str
    http_status: int
    error: str | None
    passage_count: int
    media_type: str | None
    seconds: float
    usage: Usage | None

    @property
    def handled(self) -> bool:
        """Processed, or refused with a reason. ``error`` and ``timeout`` are bugs."""
        return self.status in ("ready", "failed", "rejected")


@dataclass(frozen=True)
class _DocStatus:
    """The fields of a knowledge document this module reads (``GET .../documents/{id}``)."""

    status: str | None = None
    error: str | None = None
    passage_count: int = 0
    media_type: str | None = None

    @classmethod
    def of(cls, body: dict[str, Any]) -> _DocStatus:
        return cls(body.get("status"), body.get("error"), body.get("passage_count") or 0, body.get("media_type"))


@dataclass(frozen=True)
class EdgeCase:
    """One sample file and what the server must do with it (``broken-media/manifest.json``, ``sample_dir``)."""

    id: str
    category: str
    file: str
    source: str
    issue: str
    #: ``ready``, ``failed`` or ``rejected``.
    expect: str
    #: A ready one must come back in the top 10 for this search.
    query: str | None = None
    #: A ready one's passages must hold this text.
    text_contains: str | None = None
    #: A refused one's reason must hold this text.
    error_contains: str | None = None
    #: The file's, checked when it is downloaded (``sample_dir``).
    sha256: str = ""


def sample_dir() -> Path:
    """The sample, downloaded once per ``BROKEN_MEDIA_REV`` into ``~/.cache/hindsight-evals/broken-media``,
    each file checked against its manifest's sha256. ``HINDSIGHT_EVAL_BROKEN_MEDIA`` names a local copy
    of ``broken-media/`` instead (a checkout of hindsight-datasets, to try a new case before it lands)."""
    if local := os.getenv("HINDSIGHT_EVAL_BROKEN_MEDIA"):
        return Path(local)
    root = Path.home() / ".cache" / "hindsight-evals" / "broken-media" / BROKEN_MEDIA_REV
    manifest = root / "manifest.json"
    with httpx.Client(timeout=60, follow_redirects=True) as http:

        def fetch(path: str) -> bytes:
            r = http.get(_BROKEN_MEDIA_URL.format(rev=BROKEN_MEDIA_REV, path=path))
            r.raise_for_status()
            return r.content

        if not manifest.exists():
            root.mkdir(parents=True, exist_ok=True)
            manifest.write_bytes(fetch("manifest.json"))
        for case in json.loads(manifest.read_text())["cases"]:
            path = root / case["file"]
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]:
                continue
            data = fetch(case["file"])
            if hashlib.sha256(data).hexdigest() != case["sha256"]:
                raise ValueError(f"broken-media/{case['file']} at {BROKEN_MEDIA_REV} does not match its sha256")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return root


def load_manifest() -> list[EdgeCase]:
    return [EdgeCase(**case) for case in json.loads((sample_dir() / "manifest.json").read_text())["cases"]]


async def create_bank(http: httpx.AsyncClient, bank_id: str | None = None) -> str:
    bank_id = bank_id or f"syseval-kb-{uuid.uuid4().hex[:10]}"
    r = await http.post(KB, json={"bank_id": bank_id})
    r.raise_for_status()
    return bank_id


async def ingest_file(
    http: httpx.AsyncClient,
    bank_id: str,
    path: Path,
    *,
    document_id: str,
    pid: Server,
    timeout: float = 900.0,
) -> FileOutcome:
    size = path.stat().st_size
    started = time.monotonic()
    with ResourceMeter(pid) as meter:
        r = await _upload(http, bank_id, path, document_id, timeout)
        doc = _DocStatus()
        if r.status_code < 400:
            deadline = time.monotonic() + timeout
            checked = time.monotonic()
            while True:
                d = await http.get(
                    f"{KB}/{bank_id}/documents/{quote(document_id, safe='')}"
                )  # an id from a file name may hold "#"
                doc = _DocStatus.of(d.json()) if d.status_code == 200 else _DocStatus()
                if doc.status in ("ready", "failed") or time.monotonic() > deadline:
                    break
                # A server that died and came back has lost the document (and, on a pod, its database): a
                # sweep once polled it for an hour. Checked every 10 s, so its death is noticed meanwhile.
                if time.monotonic() - checked > 10:
                    checked = time.monotonic()
                    if not server_alive(pid):
                        break
                await asyncio.sleep(0.25)
    if r.status_code >= 500:
        status, error = "error", r.text[:500]
    elif r.status_code >= 400:
        status, error = "rejected", _detail(r)
    else:
        status = doc.status if doc.status in ("ready", "failed") else "timeout"
        error = doc.error
    return FileOutcome(
        name=path.name,
        document_id=document_id,
        size_bytes=size,
        status=status,
        http_status=r.status_code,
        error=error,
        passage_count=doc.passage_count,
        media_type=doc.media_type,
        seconds=round(time.monotonic() - started, 2),
        usage=meter.usage(),
    )


async def _upload(
    http: httpx.AsyncClient, bank_id: str, path: Path, document_id: str, timeout: float
) -> httpx.Response:
    """The upload, sent again once if the connection drops before any answer: a kept-alive connection the
    server closed as idle just as it was reused (a client race, not the file). Re-sending is an upsert."""
    for attempt in (1, 2):
        # From the open file, not read first: a heavy case is gigabytes, and this process is not under test.
        with path.open("rb") as upload:
            try:
                return await http.post(
                    f"{KB}/{bank_id}/files/ingest",
                    files={"files": (path.name, upload)},
                    data={"request": json.dumps({"files": [{"document_id": document_id}]})},
                    # Only the upload and its queueing, never the processing: given the whole file's timeout,
                    # a request in flight when the server died waited the hour out.
                    timeout=httpx.Timeout(min(timeout, 600), connect=30),
                )
            except (httpx.ReadError, httpx.RemoteProtocolError):
                if attempt == 2:
                    raise
    raise AssertionError("unreachable")


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail"))
    except ValueError:
        return r.text[:500]


async def rank_of(http: httpx.AsyncClient, bank_id: str, query: str, document_id: str, k: int = 10) -> int | None:
    """Where ``document_id`` first comes back for ``query`` (1-based), or None if not in the top ``k``."""
    r = await http.post(f"{KB}/{bank_id}/search", json={"query": query})
    r.raise_for_status()
    seen: list[str] = []
    for hit in r.json()["results"]:
        if hit["document_id"] not in seen:
            seen.append(hit["document_id"])
    return seen.index(document_id) + 1 if document_id in seen[:k] else None


def server_alive(pid: Server) -> bool:
    if pid is None:
        return True
    if isinstance(pid, str) and pid.startswith("k8s:"):
        return _pod_container_alive(pid)
    if isinstance(pid, str):
        import subprocess

        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", pid], capture_output=True, text=True, timeout=30
        )
        return state.stdout.strip() == "true"
    import psutil

    return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE


#: Each pod container's restart count when first seen: a restart (OOMKilled, a crash) since is a death,
#: even though the container is running again by the time anyone looks.
_FIRST_RESTARTS: dict[str, int] = {}


def _pod_container_alive(server: str) -> bool:
    import subprocess

    namespace, pod, container = server.removeprefix("k8s:").split("/")
    jsonpath = f'{{.status.containerStatuses[?(@.name=="{container}")].restartCount}}'
    out = subprocess.run(
        ["kubectl", "get", "pod", "-n", namespace, pod, "-o", f"jsonpath={jsonpath}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0 or not out.stdout.strip().isdigit():
        # kubectl cannot tell (expired cluster credentials made every file of a sweep "died"): unknown is not
        # dead. A pod that is really gone stops answering, and the requests' own retry waits for it.
        return True
    restarts = int(out.stdout.strip())
    return restarts == _FIRST_RESTARTS.setdefault(server, restarts)


# --- heavy files: common, just big -------------------------------------------------------------


@dataclass(frozen=True)
class HeavyCase:
    """A file a real user uploads every day, only large: built at run time, never committed."""

    id: str
    filename: str
    #: ``ffmpeg`` arguments after the inputs; None for a file this module writes itself.
    ffmpeg: tuple[str, ...] | None
    query: str | None
    #: ``ready``, or ``rejected`` for a file over the server's size limit.
    expect: str = "ready"
    #: The most the server's memory may grow ingesting it: set where the answer must not scale with the file.
    max_growth_mb: float | None = None


#: Over the server's default 500 MB upload limit, so it must be refused, and without being read into memory.
OVER_LIMIT_MB = 3 * 1024

HEAVY_CASES = (
    # Phone footage: a minute of 4K at 60 fps and 50 Mb/s (what a phone records), with sound: ~375 MB.
    HeavyCase(
        "4k-60fps-video",
        "phone-4k-1min.mp4",
        (
            *("-f", "lavfi", "-i", "testsrc2=s=3840x2160:r=60", "-f", "lavfi", "-i", "sine=f=440:r=48000"),
            *("-t", "60", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
            *("-b:v", "50M", "-maxrate", "50M", "-bufsize", "100M", "-c:a", "aac"),
        ),
        "a test pattern with a moving colour gradient and a counter",
    ),
    # Three minutes of the same is 3 GB: refused for its size, at upload, before anything reads it.
    HeavyCase("over-limit-video", "phone-4k-long.mp4", None, None, expect="rejected", max_growth_mb=300),
    # A camera's full-resolution photo: 108 megapixels, past Pillow's 89 MP warning.
    HeavyCase(
        "108mp-photo",
        "camera-108mp.jpg",
        ("-f", "lavfi", "-i", "testsrc2=s=12000x9000", "-frames:v", "1", "-q:v", "3"),
        "a colourful test pattern image",
    ),
    # An hour-long podcast.
    HeavyCase(
        "1h-podcast",
        "podcast-1h.m4a",
        ("-f", "lavfi", "-i", "sine=f=220:r=44100", "-t", "3600", "-c:a", "aac", "-b:a", "96k"),
        "a steady tone",
    ),
    HeavyCase("1500-page-pdf", "manual-1500p.pdf", None, "section 1234 torque specification for the rear axle"),
    HeavyCase("50mb-csv", "export-50mb.csv", None, "order 777777 shipped to Reykjavik"),
)


def build_heavy(case: HeavyCase, folder: Path) -> Path:
    """Write ``case``'s file into ``folder`` (once: an existing one is reused). Needs ``ffmpeg`` for media."""
    import shutil
    import subprocess

    path = folder / case.filename
    if path.exists():
        return path
    if case.ffmpeg is not None:
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("ffmpeg is not on PATH")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *case.ffmpeg, str(path)], check=True)
    elif case.expect == "rejected":
        # Its content is never read, so it need not be a video: a sparse file costs no disk to make.
        with path.open("wb") as f:
            f.truncate(OVER_LIMIT_MB * 1048576)
    elif path.suffix == ".pdf":
        path.write_bytes(
            text_pdf([f"Section {n}. Torque specification for the rear axle: {n % 90 + 10} Nm." for n in range(1500)])
        )
    else:
        with path.open("w") as f:
            f.write("order_id,customer,city,status\n")
            n = 0
            while f.tell() < 50 * 1048576:
                city = "Reykjavik" if n == 777777 else "Lisbon"
                f.write(f"{n},customer-{n},{city},shipped\n")
                n += 1
    return path


def text_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF, one line of Helvetica per page: no PDF library needed to make one."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 14 Tf 72 720 Td ({text.replace('(', '[').replace(')', ']')}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objects)} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


# --- harvest -----------------------------------------------------------------------------------


def _files(roots: list[Path], limit: int | None) -> list[Path]:
    found = sorted(p for root in roots for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    return found[:limit] if limit else found


async def _reachable(http: httpx.AsyncClient, wait: float = 900.0) -> bool:
    """Whether the server answers ``/health`` again within ``wait`` seconds."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            if (await http.get("/health", timeout=10)).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(5)
    return False


async def _harvest(
    files: list[Path], out: Path, url: str, pid: Server, timeout: float, concurrency: int, bank_config: dict
) -> int:
    """Ingest ``files`` into one bank, ``concurrency`` at a time (then usage is the mix's, not one file's).

    Resumes: a file already in ``out`` is skipped, so a stopped sweep picks up where it was.
    """
    done = {json.loads(line)["path"] for line in out.open()} if out.exists() else set()
    todo = [p for p in files if str(p) not in done]
    bugs = 0
    gate = asyncio.Semaphore(concurrency)
    in_flight: dict[Path, str] = {}
    async with httpx.AsyncClient(base_url=url, timeout=60) as http:
        bank_id = await create_bank(http)
        if bank_config:
            (await http.patch(f"{KB}/{bank_id}/config", json={"updates": bank_config})).raise_for_status()
        print(f"bank {bank_id}, {len(todo)} of {len(files)} files to go -> {out}")
        with out.open("a") as sink:

            async def one(n: int, path: Path) -> None:
                nonlocal bugs
                doc_id = f"f{n}-{path.stem[:60]}"
                async with gate:
                    if not server_alive(pid):
                        return
                    in_flight[path] = doc_id
                    for attempt in range(3):
                        try:
                            outcome = await ingest_file(
                                http, bank_id, path, document_id=doc_id, pid=pid, timeout=timeout
                            )
                            break
                        except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
                            # The way to the server broke (a port-forward to a pod lost its connection: ~300
                            # files were "errors" in seconds once), not the file: wait for it, then again.
                            outcome = FileOutcome(
                                path.name, doc_id, path.stat().st_size, "error", 0, repr(exc), 0, None, 0, None
                            )
                            if not server_alive(pid) or not await _reachable(http) or attempt == 2:
                                break
                        except httpx.HTTPError as exc:
                            outcome = FileOutcome(
                                path.name, doc_id, path.stat().st_size, "error", 0, repr(exc), 0, None, 0, None
                            )
                            break
                    if not server_alive(pid):
                        return  # still in flight: recorded below, with the others when it died
                    in_flight.pop(path, None)
                sink.write(json.dumps(asdict(outcome) | {"path": str(path)}) + "\n")
                sink.flush()
                mark = "ok " if outcome.handled else "BUG"
                usage = (
                    f"{outcome.usage.peak_rss_mb:.0f}MB {outcome.usage.cpu_seconds:.0f}cpu-s" if outcome.usage else ""
                )
                print(
                    f"{mark} {n}/{len(todo)} {outcome.status:8} {outcome.seconds:6.1f}s {usage:18} {path.name}  {(outcome.error or '')[:120]}"
                )
                bugs += not outcome.handled

            await asyncio.gather(*(one(n, p) for n, p in enumerate(todo, 1)))
    if not server_alive(pid):
        # Each file in flight when it died is recorded as such, so a resumed sweep skips them; one of them
        # is the culprit, and is ingested alone to find out which.
        died = [p for p in files if str(p) not in done and str(p) not in {json.loads(x)["path"] for x in out.open()}]
        suspects = [p for p in died if p in in_flight] or died[:concurrency]
        with out.open("a") as sink:
            for p in suspects:
                line = FileOutcome(
                    p.name, in_flight.get(p, ""), p.stat().st_size, "died", 0, "the server died", 0, None, 0, None
                )
                sink.write(json.dumps(asdict(line) | {"path": str(p)}) + "\n")
        print(f"the server died with in flight: {', '.join(p.name for p in suspects)}")
        return bugs + 1
    return bugs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("harvest", help="ingest every supported file under the given folders, one JSON line each")
    h.add_argument("roots", nargs="+", type=Path)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--limit", type=int, default=None, help="only the first N files (sorted by path)")
    h.add_argument("--timeout", type=float, default=900.0, help="seconds a file may take")
    h.add_argument("--concurrency", type=int, default=1, help="files in flight at once")
    h.add_argument("--api-url", default=None, help="a server already running; else one is started")
    h.add_argument(
        "--bank-config",
        type=json.loads,
        default={},
        help="the sweep bank's own settings, e.g. '{\"image_tokens\": 70}': cheaper pictures on a CPU embedder",
    )
    h.add_argument(
        "--container",
        default=None,
        help="with --api-url: the docker container it runs in, or k8s:<namespace>/<pod>/<container>, to measure",
    )
    args = parser.parse_args(argv)

    from hindsight_system_evals.target import eval_target

    files = _files(args.roots, args.limit)
    # The server's log beside the results: a file that broke it is read about there.
    with eval_target(args.api_url, log_path=args.out.with_suffix(".server.log")) as target:
        server = args.container or target.pid
        return (
            1
            if asyncio.run(
                _harvest(files, args.out, target.url, server, args.timeout, args.concurrency, args.bank_config)
            )
            else 0
        )


if __name__ == "__main__":
    sys.exit(main())
