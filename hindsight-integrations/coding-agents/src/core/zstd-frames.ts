/**
 * Read a CONCATENATION of Zstandard frames — the container DeepSeek Harness writes its session logs
 * in (one checksummed frame for the header, one per appended batch).
 *
 * Node's `zstdDecompressSync` and its streaming decompressor both stop after the FIRST frame, so a
 * whole-buffer decode silently returns only the header line and the session reads as empty. The fix
 * is the one dsh's own backend uses: walk the frame structure (RFC 8878 §3.1 — magic, frame header
 * descriptor, block headers, optional checksum) WITHOUT decompressing, then one-shot decode each
 * frame. Scanning is required rather than splitting on the magic number, whose bytes can legally
 * occur inside compressed data.
 *
 * A torn final frame (a crash mid-append) ends the scan instead of failing: its complete
 * predecessors are still real history.
 */
import * as zlib from "node:zlib";

const ZSTD_MAGIC = 0xfd2fb528;

/** Byte ranges of the complete frames in `buffer`, in order. Stops at the first incomplete one. */
export function scanZstdFrames(buffer: Buffer): { start: number; end: number }[] {
  const frames: { start: number; end: number }[] = [];
  let offset = 0;
  while (offset < buffer.length) {
    const start = offset;
    if (buffer.length - offset < 5) return frames; // torn tail
    if (buffer.readUInt32LE(offset) !== ZSTD_MAGIC) return frames; // not a frame boundary: stop
    offset += 4;
    const descriptor = buffer.readUInt8(offset);
    offset += 1;
    if ((descriptor & 0b0001_1000) !== 0) return frames; // reserved bits set: not ours to read
    const contentSizeFlag = descriptor >>> 6;
    const singleSegment = (descriptor & 0b0010_0000) !== 0;
    const hasChecksum = (descriptor & 0b0000_0100) !== 0;
    const dictionaryFlag = descriptor & 0b11;
    const dictionaryBytes = dictionaryFlag === 3 ? 4 : dictionaryFlag;
    const contentSizeBytes = contentSizeFlag === 0 ? (singleSegment ? 1 : 0) : 1 << contentSizeFlag;
    offset += (singleSegment ? 0 : 1) + dictionaryBytes + contentSizeBytes;
    if (offset > buffer.length) return frames;
    // Blocks: a 3-byte header each, the last one flagged, then the optional 4-byte checksum.
    for (;;) {
      if (buffer.length - offset < 3) return frames;
      const blockHeader = buffer.readUIntLE(offset, 3);
      offset += 3;
      const lastBlock = (blockHeader & 1) !== 0;
      const blockType = (blockHeader >>> 1) & 0b11;
      if (blockType === 3) return frames; // reserved block type
      // An RLE block's payload is a single byte repeated `blockSize` times.
      const payloadBytes = blockType === 1 ? 1 : blockHeader >>> 3;
      offset += payloadBytes;
      if (offset > buffer.length) return frames;
      if (lastBlock) break;
    }
    if (hasChecksum) {
      if (buffer.length - offset < 4) return frames;
      offset += 4;
    }
    frames.push({ start, end: offset });
  }
  return frames;
}

/** Decode every complete frame in a concatenated-frame artifact and join the results. */
export function zstdDecompressFrames(buffer: Buffer): string {
  return scanZstdFrames(buffer)
    .map((frame) =>
      zlib.zstdDecompressSync(buffer.subarray(frame.start, frame.end)).toString("utf8")
    )
    .join("");
}
