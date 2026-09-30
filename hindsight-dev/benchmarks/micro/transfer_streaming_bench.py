"""Microbenchmark for bank export archive streaming with multimodal attachments.

Bank transfer export produces a PKZIP archive containing JSON metadata (manifest,
per-document payloads, observations, and mental models) alongside binary multimodal
attachments (images, PDFs, audio/video clips, and document files).

Measures wall time, CPU time, peak memory (tracemalloc), and throughput across:
- ``baseline_legacy``:
    In-memory ZIP archive accumulation (``zipfile.ZipFile(io.BytesIO())``).
    Loads all document objects, facts, observations, and entire attachment binary payloads
    into memory simultaneously, causing severe memory spikes on media-rich banks.
- ``streaming (prod)``:
    Near-constant memory streaming export pipeline via ``ZipStreamer`` (PKZIP Bit 3 Data
    Descriptors), batched document JSON generation, and single-item chunked
    attachment pass-through streaming with Deflate Level 0.

Scope Note:
This microbenchmark specifically benchmarks the memory watermark and CPU throughput
of the archive creation / serialization pipeline (ZipStreamer vs legacy BytesIO + zipfile)
in isolation from external database connections and network I/O.

Every variant's output is verified using Python's standard ``zipfile.ZipFile`` to
confirm archive integrity, CRC32 correctness, and complete member extraction.

Usage (from the repo root):
    ./scripts/benchmarks/run-transfer-streaming-bench.sh
    ./scripts/benchmarks/run-transfer-streaming-bench.sh --repeats 5
    ./scripts/benchmarks/run-transfer-streaming-bench.sh --workload multimodal_medium
    ./scripts/benchmarks/run-transfer-streaming-bench.sh --json /tmp/transfer_bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
import json
import os
import random
import shutil
import tempfile
import time
import tracemalloc
import zipfile
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import BinaryIO

import pydantic_core
from hindsight_api.engine.transfer.schema import (
    SCHEMA_VERSION,
    TransferAttachment,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferManifest,
    TransferObservation,
    TransferObservationSource,
)
from hindsight_api.engine.transfer.stream_archive import ZipStreamer
from rich.console import Console
from rich.table import Table

_term_cols = shutil.get_terminal_size().columns
console = Console(width=max(_term_cols, 112), height=25) if _term_cols <= 80 else Console()


@dataclass(frozen=True)
class AttachmentSpec:
    """Specification of a multimodal binary attachment."""

    short_id: str
    media_type: str
    size_bytes: int
    kind: str = "image"
    filename: str = "attachment.bin"


@dataclass(frozen=True)
class Workload:
    """A realistic workload reflecting different bank scales and multimodal mixtures."""

    name: str
    description: str
    num_docs: int
    facts_per_doc: int
    attachments: list[AttachmentSpec]
    batch_size: int = 100
    num_observations: int = 0
    num_mental_models: int = 0

    @property
    def total_facts(self) -> int:
        return self.num_docs * self.facts_per_doc

    @property
    def total_blob_bytes(self) -> int:
        return sum(a.size_bytes for a in self.attachments)

    @property
    def total_attachments(self) -> int:
        return len(self.attachments)


@dataclass
class VariantResult:
    """One variant measured against one workload."""

    workload: str
    variant: str
    wall_ms: float
    cpu_ms: float
    ttfb_ms: float
    peak_mib: float
    total_docs: int
    total_facts: int
    total_attachments: int
    total_blob_mib: float
    archive_size_mib: float
    throughput_mb_s: float
    valid_archive: bool
    total_observations: int = 0
    total_mental_models: int = 0


# --- Synthetic Realistic Data Generators ---


_WORDS = (
    "architecture memory streaming telemetry latency throughput optimization "
    "retention pipeline Postgres asynchronous chunking transaction execution "
    "serialization descriptor payload verification enterprise cluster indexing"
).split()


def _generate_doc(doc_idx: int, facts_per_doc: int, rng: random.Random) -> TransferDocument:
    """Generates a realistic TransferDocument with structured facts and chunks."""
    facts: list[TransferFact] = []
    for f_idx in range(facts_per_doc):
        phrase = " ".join(rng.choices(_WORDS, k=12))
        facts.append(
            TransferFact(
                text=f"Fact {f_idx} for document {doc_idx:06d}: {phrase}.",
                fact_type="fact",
                context=f"Context for fact {f_idx} in document {doc_idx:06d} regarding system operation.",
                created_at=datetime.now(UTC),
                entities=[f"Entity_{f_idx % 8}_Alpha", f"Entity_{f_idx % 12}_Beta"],
                tags=["multimodal", "production", "benchmark"],
                metadata={"source": "benchmark", "doc_idx": str(doc_idx), "fact_idx": str(f_idx)},
            )
        )

    chunks: list[TransferChunk] = [
        TransferChunk(
            chunk_index=c_idx,
            chunk_text=(
                f"Document {doc_idx:06d} chunk {c_idx} containing multi-sentence contextual description "
                "with technical details, architecture references, and memory storage profiles."
            )
            * 3,
        )
        for c_idx in range(3)
    ]

    return TransferDocument(
        id=f"doc_{doc_idx:06d}",
        original_text=(
            f"Original text for document {doc_idx:06d}. Detailed discussion on distributed indexing, "
            "streaming transfer protocols, database connection pooling, and multi-tenant memory banking."
        )
        * 6,
        facts=facts,
        chunks=chunks,
    )


def _generate_observations(
    num_docs: int, facts_per_doc: int, count: int, rng: random.Random
) -> list[TransferObservation]:
    """Generates realistic consolidated TransferObservations referencing source facts."""
    if count <= 0:
        return []
    observations: list[TransferObservation] = []
    for o_idx in range(count):
        phrase = " ".join(rng.choices(_WORDS, k=14))
        src_docs = rng.sample(range(num_docs), k=min(3, num_docs))
        sources = [
            TransferObservationSource(
                document_id=f"doc_{d_idx:06d}",
                fact_index=rng.randint(0, facts_per_doc - 1),
            )
            for d_idx in src_docs
        ]
        observations.append(
            TransferObservation(
                source_id=f"obs_src_{o_idx:06d}",
                text=f"Consolidated observation {o_idx:06d}: {phrase}.",
                created_at=datetime.now(UTC),
                tags=["synthesis", "benchmark", "observation"],
                proof_count=len(sources),
                sources=sources,
            )
        )
    return observations


def _generate_mental_models(count: int, rng: random.Random) -> list[dict]:
    """Generates synthetic mental models matching bank export structure."""
    if count <= 0:
        return []
    models: list[dict] = []
    for i in range(count):
        phrase = " ".join(rng.choices(_WORDS, k=10))
        models.append(
            {
                "id": f"mm_{i:04d}",
                "bank_id": "bench_bank",
                "name": f"Mental Model {i:04d}",
                "description": f"Synthesized understanding for topic {i:04d}: {phrase}.",
                "content": (
                    f"Detailed mental model content for cluster {i:04d} synthesizing observed facts and documents. "
                    f"Domain concepts: {phrase}."
                ),
                "source_type": "observation",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
    return models


def _generate_blob_pattern(size_bytes: int) -> bytes:
    """Generates deterministic pseudo-random binary data of the requested size."""
    block_size = min(64 * 1024, size_bytes)
    pattern = bytearray((i * 37) % 256 for i in range(block_size))
    repeats = size_bytes // block_size
    remainder = size_bytes % block_size
    return bytes(pattern * repeats + pattern[:remainder])


async def _stream_blob_chunks(size_bytes: int, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    """Asynchronously yields synthetic chunks without keeping full bytes in memory."""
    pattern_block = bytearray((i * 37) % 256 for i in range(min(chunk_size, size_bytes)))
    bytes_left = size_bytes
    while bytes_left > 0:
        take = min(bytes_left, chunk_size)
        if take == len(pattern_block):
            yield bytes(pattern_block)
        else:
            yield bytes(pattern_block[:take])
        bytes_left -= take


# --- Workload Definitions ---


def build_workloads(seed: int = 1234) -> list[Workload]:
    """Builds realistic benchmark workloads mirroring production multimodal environments."""
    return [
        Workload(
            name="text_only_small",
            description="Pure text sanity check (100 docs, 1,500 facts, 100 obs, 0 blobs)",
            num_docs=100,
            facts_per_doc=15,
            num_observations=100,
            num_mental_models=5,
            attachments=[],
            batch_size=100,
        ),
        Workload(
            name="multimodal_light",
            description="Light multimodal bank (300 docs, 4,500 facts, 300 obs, 10 blobs ~25 MiB)",
            num_docs=300,
            facts_per_doc=15,
            num_observations=300,
            num_mental_models=15,
            attachments=[
                # 6 images (500 KB ~ 1.5 MB)
                *[
                    AttachmentSpec(
                        short_id=f"img_{i:02d}",
                        media_type="image/png",
                        size_bytes=(800 + i * 150) * 1024,
                        kind="image",
                        filename=f"screenshot_{i:02d}.png",
                    )
                    for i in range(6)
                ],
                # 3 PDFs (4 MB ~ 5 MB)
                *[
                    AttachmentSpec(
                        short_id=f"doc_{i:02d}",
                        media_type="application/pdf",
                        size_bytes=(4000 + i * 500) * 1024,
                        kind="document",
                        filename=f"report_{i:02d}.pdf",
                    )
                    for i in range(3)
                ],
                # 1 media blob (6.5 MB)
                AttachmentSpec(
                    short_id="media_01",
                    media_type="audio/mp4",
                    size_bytes=6500 * 1024,
                    kind="audio",
                    filename="audio_notes.mp4",
                ),
            ],
            batch_size=100,
        ),
        Workload(
            name="multimodal_medium",
            description="Active multimodal bank (800 docs, 12,000 facts, 800 obs, 25 blobs ~100 MiB)",
            num_docs=800,
            facts_per_doc=15,
            num_observations=800,
            num_mental_models=30,
            attachments=[
                # 16 images (600 KB ~ 2 MB)
                *[
                    AttachmentSpec(
                        short_id=f"img_{i:02d}",
                        media_type="image/jpeg",
                        size_bytes=(600 + (i % 5) * 300) * 1024,
                        kind="image",
                        filename=f"camera_capture_{i:02d}.jpg",
                    )
                    for i in range(16)
                ],
                # 7 PDFs (6 MB ~ 9 MB)
                *[
                    AttachmentSpec(
                        short_id=f"doc_{i:02d}",
                        media_type="application/pdf",
                        size_bytes=(6000 + (i % 4) * 1000) * 1024,
                        kind="document",
                        filename=f"specification_doc_{i:02d}.pdf",
                    )
                    for i in range(7)
                ],
                # 2 large media blobs (18 MB ~ 20 MB)
                *[
                    AttachmentSpec(
                        short_id=f"media_{i:02d}",
                        media_type="video/mp4",
                        size_bytes=(18000 + i * 2000) * 1024,
                        kind="video",
                        filename=f"screencast_{i:02d}.mp4",
                    )
                    for i in range(2)
                ],
            ],
            batch_size=100,
        ),
        Workload(
            name="multimodal_heavy",
            description="Heavy multimedia bank (2,000 docs, 30,000 facts, 2,000 obs, 50 blobs ~250 MiB)",
            num_docs=2000,
            facts_per_doc=15,
            num_observations=2000,
            num_mental_models=60,
            attachments=[
                # 32 images (700 KB ~ 2.2 MB)
                *[
                    AttachmentSpec(
                        short_id=f"img_{i:02d}",
                        media_type="image/png",
                        size_bytes=(700 + (i % 6) * 300) * 1024,
                        kind="image",
                        filename=f"schematic_{i:02d}.png",
                    )
                    for i in range(32)
                ],
                # 14 documents / PDFs (7 MB ~ 12 MB)
                *[
                    AttachmentSpec(
                        short_id=f"doc_{i:02d}",
                        media_type="application/pdf",
                        size_bytes=(7000 + (i % 5) * 1200) * 1024,
                        kind="document",
                        filename=f"research_paper_{i:02d}.pdf",
                    )
                    for i in range(14)
                ],
                # 4 large archive / media blobs (25 MB ~ 30 MB)
                *[
                    AttachmentSpec(
                        short_id=f"blob_{i:02d}",
                        media_type="application/octet-stream",
                        size_bytes=(25000 + i * 1500) * 1024,
                        kind="binary",
                        filename=f"dataset_partition_{i:02d}.bin",
                    )
                    for i in range(4)
                ],
            ],
            batch_size=100,
        ),
    ]


# --- Export Pipeline Implementations ---


def _verify_zip_file(
    file_obj: BinaryIO,
    expected_docs: int,
    expected_attachments: int,
    expected_observations: int = 0,
    expected_mental_models: int = 0,
) -> bool:
    """Verifies archive structural validity and CRC checksums from a file object."""
    try:
        file_obj.seek(0)
        with zipfile.ZipFile(file_obj, "r") as zf:
            if zf.testzip() is not None:
                return False

            names = set(zf.namelist())
            if "manifest.json" not in names:
                return False

            if expected_observations > 0 and "observations.json" not in names:
                return False

            if expected_mental_models > 0 and "mental_models.json" not in names:
                return False

            if expected_attachments > 0:
                if "attachments.json" not in names:
                    return False
                for i in range(expected_attachments):
                    if f"blobs/{i:06d}.bin" not in names:
                        return False

            doc_entries = [n for n in names if n.startswith("documents/") and n.endswith(".json")]
            if len(doc_entries) != expected_docs:
                return False

            return True
    except Exception:
        return False


@dataclass(frozen=True)
class StreamRunOutput:
    archive_size: int
    is_valid: bool
    ttfb_ms: float


def _run_legacy_variant(
    workload: Workload,
    seed: int = 1234,
    verify: bool = True,
) -> StreamRunOutput:
    """Legacy export: loads all docs, blobs, obs into memory, builds ZIP via BytesIO."""
    t_start = time.perf_counter()
    rng = random.Random(seed)

    # 1. Load all documents in memory
    all_docs = [_generate_doc(i, workload.facts_per_doc, rng) for i in range(workload.num_docs)]

    # 2. Observations and mental models
    observations = _generate_observations(workload.num_docs, workload.facts_per_doc, workload.num_observations, rng)
    mental_models = _generate_mental_models(workload.num_mental_models, rng)

    # 3. Load all blobs into memory dict
    blobs: dict[str, bytes] = {}
    transfer_attachments: list[TransferAttachment] = []
    for idx, att in enumerate(workload.attachments):
        entry = f"blobs/{idx:06d}.bin"
        data = _generate_blob_pattern(att.size_bytes)
        blobs[entry] = data
        transfer_attachments.append(
            TransferAttachment(
                bank_id="bench_bank",
                document_id=f"doc_{idx % workload.num_docs:06d}",
                attachment_hash=f"hash_{idx:08x}",
                short_id=att.short_id,
                media_type=att.media_type,
                byte_size=att.size_bytes,
                kind=att.kind,
                filename=att.filename,
                created_at=datetime.now(UTC),
                entry=entry,
            )
        )

    # 4. Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, doc in enumerate(all_docs):
            zf.writestr(
                f"documents/{idx:06d}.json",
                doc.model_dump_json(indent=2, exclude_none=False).encode("utf-8"),
            )

        if observations:
            obs_json = ("[\n" + ",\n".join(o.model_dump_json(indent=2) for o in observations) + "\n]\n").encode("utf-8")
            zf.writestr("observations.json", obs_json)

        if mental_models:
            zf.writestr("mental_models.json", json.dumps(mental_models, indent=2).encode("utf-8"))

        if transfer_attachments:
            att_json = ("[\n" + ",\n".join(a.model_dump_json(indent=2) for a in transfer_attachments) + "\n]\n").encode(
                "utf-8"
            )
            zf.writestr("attachments.json", att_json)
            for entry, data in blobs.items():
                zf.writestr(zipfile.ZipInfo(entry), data, compress_type=zipfile.ZIP_STORED)

        manifest = TransferManifest(
            schema_version=SCHEMA_VERSION,
            source_bank_id="bench_bank",
            document_count=len(all_docs),
            fact_count=workload.total_facts,
            observation_count=len(observations),
            mental_model_count=len(mental_models),
            attachment_count=len(transfer_attachments),
            exported_at=datetime.now(UTC),
        )
        zf.writestr("manifest.json", manifest.model_dump_json(indent=2).encode("utf-8"))

    archive_bytes = buf.getvalue()
    ttfb_ms = (time.perf_counter() - t_start) * 1000
    buf.close()

    is_valid = True
    if verify:
        is_valid = _verify_zip_file(
            io.BytesIO(archive_bytes),
            workload.num_docs,
            workload.total_attachments,
            workload.num_observations,
            workload.num_mental_models,
        )
    return StreamRunOutput(archive_size=len(archive_bytes), is_valid=is_valid, ttfb_ms=ttfb_ms)


async def _run_streaming_variant(
    workload: Workload,
    seed: int = 1234,
    capture_file: BinaryIO | None = None,
) -> StreamRunOutput:
    """Streaming export: ZipStreamer with batched docs, obs, mental models, and chunked blobs."""
    t_start = time.perf_counter()
    ttfb_ms = 0.0
    first_chunk = True
    rng = random.Random(seed)
    streamer = ZipStreamer()

    observations = _generate_observations(workload.num_docs, workload.facts_per_doc, workload.num_observations, rng)
    mental_models = _generate_mental_models(workload.num_mental_models, rng)

    transfer_attachments: list[TransferAttachment] = []
    for idx, att in enumerate(workload.attachments):
        entry = f"blobs/{idx:06d}.bin"
        transfer_attachments.append(
            TransferAttachment(
                bank_id="bench_bank",
                document_id=f"doc_{idx % workload.num_docs:06d}",
                attachment_hash=f"hash_{idx:08x}",
                short_id=att.short_id,
                media_type=att.media_type,
                byte_size=att.size_bytes,
                kind=att.kind,
                filename=att.filename,
                created_at=datetime.now(UTC),
                entry=entry,
            )
        )

    # 4MB virtual chunk buffer simulating FileStorage.store_stream or HTTP socket
    CHUNK_STORAGE_FLUSH_SIZE = 4 * 1024 * 1024
    chunk_buffer = bytearray()
    total_archive_bytes = 0

    async def _emit_chunk(chunk: bytes) -> None:
        nonlocal total_archive_bytes, first_chunk, ttfb_ms
        if first_chunk and len(chunk) > 0:
            ttfb_ms = (time.perf_counter() - t_start) * 1000
            first_chunk = False
        total_archive_bytes += len(chunk)
        if capture_file is not None:
            capture_file.write(chunk)
        else:
            chunk_buffer.extend(chunk)
            if len(chunk_buffer) >= CHUNK_STORAGE_FLUSH_SIZE:
                chunk_buffer.clear()

    # 1. Stream documents in batches
    for batch_start in range(0, workload.num_docs, workload.batch_size):
        batch_end = min(batch_start + workload.batch_size, workload.num_docs)
        for doc_idx in range(batch_start, batch_end):
            doc = _generate_doc(doc_idx, workload.facts_per_doc, rng)
            doc_bytes = pydantic_core.to_json(doc, exclude_none=True)
            async for chunk in streamer.write_file_bytes(f"documents/{doc_idx:06d}.json", doc_bytes, compress=True):
                await _emit_chunk(chunk)

    # 2. Stream observations
    if observations:
        obs_bytes = pydantic_core.to_json(observations, exclude_none=True)
        async for chunk in streamer.write_file_bytes("observations.json", obs_bytes, compress=True):
            await _emit_chunk(chunk)

    # 3. Stream attachments one by one (chunked pass-through)
    for idx, att in enumerate(workload.attachments):
        entry = f"blobs/{idx:06d}.bin"
        blob_chunks = _stream_blob_chunks(att.size_bytes)
        async for chunk in streamer.write_file_chunks(entry, blob_chunks, compress=True, compression_level=0):
            await _emit_chunk(chunk)

    # 4. Stream attachments metadata
    if transfer_attachments:
        att_bytes = pydantic_core.to_json(transfer_attachments, exclude_none=True)
        async for chunk in streamer.write_file_bytes("attachments.json", att_bytes, compress=True):
            await _emit_chunk(chunk)

    # 5. Stream mental models
    if mental_models:
        mm_bytes = json.dumps(mental_models).encode("utf-8")
        async for chunk in streamer.write_file_bytes("mental_models.json", mm_bytes, compress=True):
            await _emit_chunk(chunk)

    # 6. Stream manifest
    manifest = TransferManifest(
        schema_version=SCHEMA_VERSION,
        source_bank_id="bench_bank",
        document_count=workload.num_docs,
        fact_count=workload.total_facts,
        observation_count=len(observations),
        mental_model_count=len(mental_models),
        attachment_count=len(transfer_attachments),
        exported_at=datetime.now(UTC),
    )
    manifest_bytes = pydantic_core.to_json(manifest, indent=2)
    async for chunk in streamer.write_file_bytes("manifest.json", manifest_bytes, compress=True):
        await _emit_chunk(chunk)

    # 7. Finish streamer
    await _emit_chunk(streamer.finish())
    chunk_buffer.clear()

    is_valid = True
    if capture_file is not None:
        is_valid = _verify_zip_file(
            capture_file,
            workload.num_docs,
            workload.total_attachments,
            workload.num_observations,
            workload.num_mental_models,
        )
    return StreamRunOutput(archive_size=total_archive_bytes, is_valid=is_valid, ttfb_ms=ttfb_ms)


# --- Timing and Profiling ---


@dataclass(frozen=True)
class Timing:
    wall_ms: float
    cpu_ms: float
    ttfb_ms: float
    archive_size_bytes: int
    is_valid: bool


def _measure_timing(
    fn: Callable[[], StreamRunOutput],
    repeats: int,
    warmup: int = 1,
) -> Timing:
    for _ in range(warmup):
        fn()

    best_wall = float("inf")
    best_cpu = 0.0
    best_ttfb = float("inf")
    archive_size = 0
    is_valid = False

    for _ in range(repeats):
        gc.collect()
        t0, c0 = time.perf_counter(), time.process_time()
        output = fn()
        archive_size, is_valid, ttfb_ms = output.archive_size, output.is_valid, output.ttfb_ms
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        if wall < best_wall:
            best_wall, best_cpu, best_ttfb = wall, cpu, ttfb_ms

    return Timing(
        wall_ms=best_wall * 1000,
        cpu_ms=best_cpu * 1000,
        ttfb_ms=best_ttfb,
        archive_size_bytes=archive_size,
        is_valid=is_valid,
    )


def _measure_peak_mib(fn: Callable[[], StreamRunOutput]) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 * 1024)


# --- Benchmark Orchestration ---


def run(
    workloads: Sequence[Workload],
    repeats: int,
    warmup: int = 1,
    seed: int = 1234,
) -> list[VariantResult]:
    results: list[VariantResult] = []

    for wl in workloads:
        # 1. Verify archive validity once per variant outside the timed loop
        legacy_valid = _run_legacy_variant(wl, seed, verify=True).is_valid
        with tempfile.TemporaryFile() as tmp:
            streaming_valid = asyncio.run(_run_streaming_variant(wl, seed, capture_file=tmp)).is_valid

        # 2. Benchmark pure export generation timing and memory watermark
        def legacy_time_fn() -> StreamRunOutput:
            return _run_legacy_variant(wl, seed, verify=False)

        def legacy_mem_fn() -> StreamRunOutput:
            return _run_legacy_variant(wl, seed, verify=False)

        def streaming_time_fn() -> StreamRunOutput:
            return asyncio.run(_run_streaming_variant(wl, seed, capture_file=None))

        def streaming_mem_fn() -> StreamRunOutput:
            return asyncio.run(_run_streaming_variant(wl, seed, capture_file=None))

        variants = [
            ("baseline_legacy", legacy_time_fn, legacy_mem_fn, legacy_valid),
            ("streaming (prod)", streaming_time_fn, streaming_mem_fn, streaming_valid),
        ]

        for name, time_fn, mem_fn, is_valid in variants:
            timing = _measure_timing(time_fn, repeats, warmup=warmup)
            peak_mib = _measure_peak_mib(mem_fn)

            archive_size_mib = timing.archive_size_bytes / (1024 * 1024)
            wall_sec = timing.wall_ms / 1000.0
            throughput = archive_size_mib / wall_sec if wall_sec > 0 else 0.0

            results.append(
                VariantResult(
                    workload=wl.name,
                    variant=name,
                    wall_ms=timing.wall_ms,
                    cpu_ms=timing.cpu_ms,
                    ttfb_ms=timing.ttfb_ms,
                    peak_mib=peak_mib,
                    total_docs=wl.num_docs,
                    total_facts=wl.total_facts,
                    total_attachments=wl.total_attachments,
                    total_blob_mib=wl.total_blob_bytes / (1024 * 1024),
                    archive_size_mib=archive_size_mib,
                    throughput_mb_s=throughput,
                    valid_archive=is_valid,
                    total_observations=wl.num_observations,
                    total_mental_models=wl.num_mental_models,
                )
            )

    return results


def _render(workloads: Sequence[Workload], results: list[VariantResult]) -> None:
    by_wl: dict[str, list[VariantResult]] = {}
    for r in results:
        by_wl.setdefault(r.workload, []).append(r)

    for wl in workloads:
        rows = by_wl.get(wl.name, [])
        if not rows:
            continue

        baseline = next((r for r in rows if "baseline" in r.variant), rows[0])

        table = Table(
            title=(
                f"[bold]{wl.name}[/bold] — {wl.description}\n"
                f"[dim]Docs: {wl.num_docs:,} | Facts: {wl.total_facts:,} | "
                f"Obs: {wl.num_observations:,} | Models: {wl.num_mental_models:,} | "
                f"Attachments: {wl.total_attachments} ({wl.total_blob_bytes / (1024 * 1024):.1f} MiB blobs)[/dim]"
            ),
            title_justify="left",
        )
        table.add_column("variant", style="cyan", no_wrap=True)
        table.add_column("wall ms", justify="right")
        table.add_column("speedup", justify="right", style="green")
        table.add_column("cpu ms", justify="right")
        table.add_column("TTFB ms", justify="right", style="magenta")
        table.add_column("peak RAM", justify="right")
        table.add_column("RAM Δ", justify="right", style="bold green")
        table.add_column("archive", justify="right")
        table.add_column("size Δ", justify="right", style="green")
        table.add_column("MB/s", justify="right")
        table.add_column("status", justify="center")

        for r in rows:
            speedup = baseline.wall_ms / r.wall_ms if r.wall_ms > 0 else float("inf")
            if r.variant == baseline.variant:
                mem_str = "base"
                size_str = "base"
            else:
                pct = ((baseline.peak_mib - r.peak_mib) / baseline.peak_mib) * 100
                mem_str = f"-{pct:.1f}%" if pct >= 0 else f"+{-pct:.1f}%"
                size_delta = ((r.archive_size_mib - baseline.archive_size_mib) / baseline.archive_size_mib) * 100
                size_str = f"{size_delta:+.2f}%" if abs(size_delta) >= 0.005 else "0.00%"

            table.add_row(
                r.variant,
                f"{r.wall_ms:,.1f}",
                f"{speedup:.2f}x" if r.variant != baseline.variant else "—",
                f"{r.cpu_ms:,.1f}",
                f"{r.ttfb_ms:,.2f}" if r.ttfb_ms < 10 else f"{r.ttfb_ms:,.1f}",
                f"{r.peak_mib:,.2f} MiB",
                mem_str,
                f"{r.archive_size_mib:,.2f} MiB",
                size_str,
                f"{r.throughput_mb_s:,.1f}",
                "[green]ok[/green]" if r.valid_archive else "[red]FAIL[/red]",
            )
        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed repeats per variant (best-of); default 3",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="unmeasured warmup runs per variant before timing; default 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="seed for synthetic generation; default 1234",
    )
    parser.add_argument(
        "--workload",
        action="append",
        help="run only this workload (repeatable); default all",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="also write raw results to this path",
    )
    args = parser.parse_args()

    workloads = build_workloads(args.seed)
    if args.workload:
        wanted = set(args.workload)
        unknown = wanted - {w.name for w in workloads}
        if unknown:
            parser.error(f"unknown workload(s): {', '.join(sorted(unknown))}")
        workloads = [w for w in workloads if w.name in wanted]

    console.print(
        f"[dim]Running Transfer Streaming microbenchmarks | cpu_count={os.cpu_count()} | repeats={args.repeats} | warmup={args.warmup}[/dim]\n"
    )
    results = run(workloads, repeats=args.repeats, warmup=args.warmup, seed=args.seed)
    _render(workloads, results)

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        console.print(f"[dim]wrote {args.json_path}[/dim]")


if __name__ == "__main__":
    main()
