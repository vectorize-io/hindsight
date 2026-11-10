/** A single satisfiable byte range, inclusive on both ends (HTTP semantics). */
interface ByteRange {
  start: number;
  end: number;
}

/**
 * Parses a `Range: bytes=...` header against a body of `size` bytes.
 *
 * Returns the range, `"unsatisfiable"` (answer 416), or null when the header is
 * absent, malformed or asks for several ranges — those get the whole body (200),
 * which RFC 9110 allows a server to do.
 */
export function parseByteRange(
  header: string | null,
  size: number
): ByteRange | "unsatisfiable" | null {
  const match = header?.trim().match(/^bytes=(\d*)-(\d*)$/);
  if (!match) return null;
  const [, rawStart, rawEnd] = match;
  if (!rawStart && !rawEnd) return null;
  if (!rawStart) {
    // Suffix form: the last N bytes.
    const suffix = Number(rawEnd);
    if (suffix === 0) return "unsatisfiable";
    return { start: Math.max(0, size - suffix), end: size - 1 };
  }
  const start = Number(rawStart);
  if (start >= size) return "unsatisfiable";
  const end = rawEnd ? Math.min(Number(rawEnd), size - 1) : size - 1;
  if (end < start) return null;
  return { start, end };
}
