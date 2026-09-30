"""Streaming ZIP archive writer using PKZIP Data Descriptors, so no entry is ever held whole.

Design Rationale:
-----------------
1. Why not Python's standard `zipfile.ZipFile`?
   - `zipfile.ZipFile(file, mode="w")` fundamentally requires `file` to support `seek()`.
     When writing file headers, it writes placeholder zeros for CRC32, compressed_size,
     and uncompressed_size, and then seeks backwards upon file completion to backfill
     these fields.
   - Streaming HTTP responses (`StreamingResponse`), S3 upload sockets, and database
     pipelines are non-seekable. Supporting `zipfile` forces either:
     a) Full in-memory accumulation (`io.BytesIO`), which caused severe memory spikes
        (multi-gigabyte RAM spikes triggering container OOM kills on large banks);
     b) Local disk staging (`tempfile`), which causes 404 Node-Drift in multi-pod
        Kubernetes clusters when a download request routes to a different pod, and risks
        root filesystem disk exhaustion.

2. Why not external third-party libraries (e.g. `stream-zip`, `zipstream`)?
   - Third-party packages rely almost exclusively on synchronous generator interfaces
     (`def ... yield`), requiring threadpool dispatch (`anyio.to_thread`) and channel
     queues to bridge into asyncio. This introduces thread-switching overhead and memory
     buffering that erodes the bounded-memory guarantee.
   - `hindsight-api-slim` enforces a strictly audited, minimal supply-chain footprint.
     The PKZIP Bit 3 protocol is compact and stable (~270 LOC) and can be fully
     implemented using Python's built-in `struct` and `zlib` without adding external
     dependencies or CVE attack surface.

3. Key Features and Design Trade-offs:
   - Async-native generator pipeline (`AsyncIterator[bytes]`), seamlessly hooking into
     FastAPI `StreamingResponse` and `FileStorage.store_stream`.
   - PKZIP General Purpose Bit 3 (0x0008) Data Descriptors, emitting CRC32 and sizes
     immediately after each file's stream without ever seeking backwards.
   - Zip64 Archive Support: Transparently emits Zip64 End of Central Directory and
     Zip64 Locator records when the overall archive size, member count (>= 65,535), or
     central directory offset exceeds 32-bit limits (4 GB). Individual member streams
     are bounded to 4 GB, avoiding invalid 64-bit data descriptors with 32-bit local headers.
   - Deflate Level 0 (Method 8) with Data Descriptors: Text and JSON documents use standard
     Deflate compression (level 6), while binary multimodal attachments bypass CPU compression
     overhead by wrapping into uncompressed Deflate blocks (level 0 / Method 8).
     Emitting a Bit 3 Data Descriptor with Deflate framing provides self-terminating streams
     guaranteeing compatibility with sequential streaming ZIP readers (Java ZipInputStream,
     funzip) without buffering the full blob in memory for CRC/size pre-computation.
   - Event-loop friendly: periodically yields control via `asyncio.sleep(0)` during chunk
     streaming across entries to avoid blocking concurrent requests.
   - Bounded pieces: whatever size the source yields, the compressor is fed at most
     `CHUNK_SIZE` at a time, so its output never grows with a blob. The central
     directory (one small record per entry) is the only state that grows with the archive.
"""

from __future__ import annotations

import asyncio
import struct
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

# PKZIP magic signatures
SIG_LOCAL_FILE_HEADER = b"PK\x03\x04"
SIG_DATA_DESCRIPTOR = b"PK\x07\x08"
SIG_CENTRAL_DIRECTORY = b"PK\x01\x02"
SIG_END_OF_CENTRAL_DIRECTORY = b"PK\x05\x06"
SIG_ZIP64_EOCD = b"PK\x06\x06"
SIG_ZIP64_LOCATOR = b"PK\x06\x07"

CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunk size
_YIELD_EVERY_CHUNKS = 16  # Yield event loop every ~1 MB


@dataclass(frozen=True)
class _DosDateTime:
    """DOS-encoded time and date."""

    time: int
    date: int


@dataclass
class _ZipEntry:
    """Metadata recorded for each completed entry to build the Central Directory."""

    fn_bytes: bytes
    method: int
    flags: int
    dos_time: int
    dos_date: int
    crc: int
    comp_size: int
    uncomp_size: int
    offset: int


async def _bounded(chunk_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Re-slice a stream so no piece handed to the compressor exceeds ``CHUNK_SIZE``.

    A source is free to yield its data in one piece (a storage backend that reads a
    whole row). Compressing that piece in one call would make the compressor emit
    an output buffer as large as the input, doubling the peak for the largest blob.
    """
    async for chunk in chunk_stream:
        view = memoryview(chunk)
        for start in range(0, len(view), CHUNK_SIZE):
            yield view[start : start + CHUNK_SIZE]


class ZipStreamer:
    """Async writer for streaming PKZIP archives without holding any entry in memory.

    Employs PKZIP General Purpose Bit 3 (0x0008) so that files can be compressed
    and emitted on the fly without knowing the compressed or uncompressed size or CRC
    in advance. Supports Zip64 transparently when exceeding 32-bit limits (4 GB).
    """

    def __init__(self, compression_level: int = 6) -> None:
        self.entries: list[_ZipEntry] = []
        self.offset = 0
        self.compression_level = compression_level
        self._total_chunks_streamed = 0

    def _dos_datetime(self, dt: datetime | None = None) -> _DosDateTime:
        now = dt or datetime.now()
        # DOS timestamps only support years 1980..2107
        year = max(1980, min(2107, now.year))
        dos_time = (now.hour << 11) | (now.minute << 5) | (now.second // 2)
        dos_date = ((year - 1980) << 9) | (now.month << 5) | now.day
        return _DosDateTime(time=dos_time, date=dos_date)

    async def write_file_chunks(
        self,
        filename: str,
        chunk_stream: AsyncIterator[bytes],
        compress: bool = True,
        compression_level: int | None = None,
        dt: datetime | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream an entry into the ZIP from an async iterator of chunks.

        When compress=True and compression_level=0 (Z_NO_COMPRESSION), chunks are
        wrapped into standard Deflate blocks (Method 8) without compression overhead.
        This provides self-terminating streams required by sequential/streaming ZIP
        readers (e.g. Java ZipInputStream, funzip) while avoiding redundant CPU usage
        on pre-compressed multimodal binary blobs.
        """
        fn_bytes = filename.encode("utf-8")
        level = self.compression_level if compression_level is None else compression_level
        method = 8 if compress else 0  # 8 = Deflate, 0 = Stored
        flags = 0x0808  # Bit 3 (Data descriptor) + Bit 11 (UTF-8 filename)
        dos_dt = self._dos_datetime(dt)
        local_offset = self.offset
        version_needed = 20

        # Local file header (sizes & crc zeroed out because of Bit 3)
        local_header = (
            struct.pack(
                "<4sHHHHHIIIHH",
                SIG_LOCAL_FILE_HEADER,
                version_needed,
                flags,
                method,
                dos_dt.time,
                dos_dt.date,
                0,
                0,
                0,
                len(fn_bytes),
                0,  # extra field length
            )
            + fn_bytes
        )
        self.offset += len(local_header)
        yield local_header

        crc = 0
        uncomp_size = 0
        comp_size = 0
        compressor = zlib.compressobj(level=level, method=zlib.DEFLATED, wbits=-15) if compress else None

        async for chunk in _bounded(chunk_stream):
            crc = zlib.crc32(chunk, crc)
            uncomp_size += len(chunk)
            out = compressor.compress(chunk) if compressor else bytes(chunk)
            if out:
                comp_size += len(out)
                self.offset += len(out)
                yield out
            self._total_chunks_streamed += 1
            if self._total_chunks_streamed % _YIELD_EVERY_CHUNKS == 0:
                await asyncio.sleep(0)

        if compressor:
            tail = compressor.flush()
            if tail:
                comp_size += len(tail)
                self.offset += len(tail)
                yield tail

        # Data descriptor (emitted immediately following the entry data)
        if comp_size >= 0xFFFFFFFF or uncomp_size >= 0xFFFFFFFF:
            raise OverflowError(
                f"Archive entry '{filename}' exceeds 4 GB limit ({max(comp_size, uncomp_size)} bytes). "
                "Streaming single entries exceeding 4 GB requires pre-allocated Zip64 extra headers."
            )

        dd = struct.pack(
            "<4sIII",
            SIG_DATA_DESCRIPTOR,
            crc,
            comp_size,
            uncomp_size,
        )
        self.offset += len(dd)
        yield dd

        self.entries.append(
            _ZipEntry(
                fn_bytes=fn_bytes,
                method=method,
                flags=flags,
                dos_time=dos_dt.time,
                dos_date=dos_dt.date,
                crc=crc,
                comp_size=comp_size,
                uncomp_size=uncomp_size,
                offset=local_offset,
            )
        )

    async def write_file_bytes(
        self,
        filename: str,
        data: bytes,
        compress: bool = True,
        compression_level: int | None = None,
        dt: datetime | None = None,
    ) -> AsyncIterator[bytes]:
        """Convenience method to write an in-memory byte buffer as a single entry."""

        async def _gen() -> AsyncIterator[bytes]:
            # Slice into CHUNK_SIZE chunks to allow interleaving and avoid huge single-buffer copies
            if not data:
                return
            for i in range(0, len(data), CHUNK_SIZE):
                yield data[i : i + CHUNK_SIZE]

        async for chunk in self.write_file_chunks(
            filename, _gen(), compress=compress, compression_level=compression_level, dt=dt
        ):
            yield chunk

    def finish(self) -> bytes:
        """Construct Central Directory and End of Central Directory records.

        Returns the trailing bytes that complete the ZIP archive.
        """
        cd_start = self.offset
        out = bytearray()
        needs_zip64 = (
            len(self.entries) >= 0xFFFF or cd_start >= 0xFFFFFFFF or any(e.offset >= 0xFFFFFFFF for e in self.entries)
        )

        for e in self.entries:
            use_e_zip64 = e.offset >= 0xFFFFFFFF
            rec_offset = 0xFFFFFFFF if use_e_zip64 else e.offset
            extra = struct.pack("<HHQ", 0x0001, 8, e.offset) if use_e_zip64 else b""

            cd_record = (
                struct.pack(
                    "<4sHHHHHHIIIHHHHHII",
                    SIG_CENTRAL_DIRECTORY,
                    45 if use_e_zip64 else 20,  # version made by
                    45 if use_e_zip64 else 20,  # version needed
                    e.flags,
                    e.method,
                    e.dos_time,
                    e.dos_date,
                    e.crc,
                    e.comp_size,
                    e.uncomp_size,
                    len(e.fn_bytes),
                    len(extra),
                    0,  # comment len
                    0,  # disk num start
                    0,  # int attr
                    0,  # ext attr
                    rec_offset,
                )
                + e.fn_bytes
                + extra
            )
            out.extend(cd_record)

        cd_size = len(out)

        if needs_zip64:
            zip64_eocd_offset = cd_start + cd_size
            zip64_eocd = struct.pack(
                "<4sQHHIIQQQQ",
                SIG_ZIP64_EOCD,
                44,  # size of remaining record
                45,  # version made by
                45,  # version needed
                0,  # disk number
                0,  # disk with CD
                len(self.entries),
                len(self.entries),
                cd_size,
                cd_start,
            )
            out.extend(zip64_eocd)

            zip64_locator = struct.pack(
                "<4sIQI",
                SIG_ZIP64_LOCATOR,
                0,  # disk with Zip64 EOCD
                zip64_eocd_offset,
                1,  # total number of disks
            )
            out.extend(zip64_locator)

            eocd = struct.pack(
                "<4sHHHHIIH",
                SIG_END_OF_CENTRAL_DIRECTORY,
                0,
                0,
                0xFFFF,
                0xFFFF,
                0xFFFFFFFF,
                0xFFFFFFFF,
                0,
            )
            out.extend(eocd)
        else:
            eocd = struct.pack(
                "<4sHHHHIIH",
                SIG_END_OF_CENTRAL_DIRECTORY,
                0,
                0,
                len(self.entries),
                len(self.entries),
                cd_size,
                cd_start,
                0,
            )
            out.extend(eocd)

        self.offset += len(out)
        return bytes(out)
