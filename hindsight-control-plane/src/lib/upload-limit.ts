// The largest request body the control plane accepts (HINDSIGHT_CP_MAX_UPLOAD_SIZE: bytes, or
// '100mb' / '1gb'; default 100 MB, matching the dataplane's
// HINDSIGHT_API_FILE_CONVERSION_MAX_BATCH_SIZE_MB). next.config hands it to Next.js, which
// buffers request bodies that pass through middleware and TRUNCATES anything longer without an
// error, so a too-big upload arrived corrupted. The middleware refuses those with a 413 instead,
// from Content-Length, before Next.js cuts anything. Imported by next.config: no `@/` paths.

const UNITS: Record<string, number> = { b: 1, kb: 1024, mb: 1024 ** 2, gb: 1024 ** 3 };

export function maxUploadBytes(value = process.env.HINDSIGHT_CP_MAX_UPLOAD_SIZE): number {
  const m = /^\s*(\d+(?:\.\d+)?)\s*(b|kb|mb|gb)?\s*$/i.exec(value ?? "");
  if (!m) return 100 * UNITS.mb;
  return Math.floor(Number(m[1]) * UNITS[(m[2] ?? "b").toLowerCase()]);
}
