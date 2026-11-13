import { NextRequest, NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";
import { parseByteRange } from "@/lib/byte-range";

/**
 * Forwards everything under /api/knowledge-banks to the dataplane's
 * /v1/default/knowledge-banks. The generated SDK does not cover knowledge banks, and
 * the endpoints are a thin tree (banks, documents, passages, search), so one
 * catch-all route forwards them instead of one file per endpoint.
 *
 * - Request bodies are streamed through untouched with the browser's Content-Type, so
 *   a multipart upload keeps its boundary and is never buffered here.
 * - Binary responses (original files, passage media) are streamed back with their
 *   Content-Type. The dataplane does not answer Range requests, which <video> and
 *   Safari's <audio> need to seek; passage media are bounded clips (one page, one
 *   30-60s window), so for those this route buffers the clip and serves the range
 *   itself. Original files can be hundreds of MB and are only ever downloaded, so they
 *   are never buffered.
 */

// Response headers worth keeping from the dataplane. Not content-length or
// content-encoding: fetch hands back the body already decompressed, so the upstream
// length would describe different bytes and truncate the response.
const PASSTHROUGH_HEADERS = [
  "content-type",
  "content-disposition",
  "cache-control",
  "content-range",
  "accept-ranges",
  "etag",
  "last-modified",
];

const MEDIA_PATH = /^[^/]+\/documents\/[^/]+\/passages\/\d+\/media$/;
const FILE_PATH = /^[^/]+\/documents\/[^/]+\/file$/;

// An original stored as application/octet-stream (a client that sent no type) would
// download instead of previewing; its file name says what it is.
const TYPE_BY_EXTENSION: Record<string, string> = {
  pdf: "application/pdf",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  mp3: "audio/mpeg",
  m4a: "audio/mp4",
  wav: "audio/wav",
  ogg: "audio/ogg",
  aiff: "audio/aiff",
  mp4: "video/mp4",
  webm: "video/webm",
  mov: "video/quicktime",
  md: "text/markdown; charset=utf-8",
  txt: "text/plain; charset=utf-8",
  csv: "text/csv; charset=utf-8",
  json: "application/json",
  html: "text/html; charset=utf-8",
};

/** The file name in a `Content-Disposition` header (RFC 5987 `filename*` or plain `filename`). */
function dispositionFilename(header: string | null): string | null {
  const star = header?.match(/filename\*=UTF-8''([^;]+)/i);
  if (star) return decodeURIComponent(star[1]);
  return header?.match(/filename="?([^";]+)"?/i)?.[1] ?? null;
}

function upstreamUrl(path: string[], search: string): string | null {
  // An encoded ".." still normalises as a dot segment in the URL parser, which would
  // let a caller step out of the knowledge-banks prefix.
  if (path.some((segment) => segment === "." || segment === ".." || segment === "")) return null;
  const suffix = path.map(encodeURIComponent).join("/");
  return `${DATAPLANE_URL}/v1/default/knowledge-banks${suffix ? `/${suffix}` : ""}${search}`;
}

function copyHeaders(from: Headers): Headers {
  const headers = new Headers();
  for (const name of PASSTHROUGH_HEADERS) {
    const value = from.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

async function forward(request: NextRequest, { params }: { params: Promise<{ path?: string[] }> }) {
  const { path = [] } = await params;
  const url = upstreamUrl(path, request.nextUrl.search);
  if (!url) {
    return NextResponse.json({ error: "Invalid path" }, { status: 400 });
  }

  const hasBody = !["GET", "HEAD", "DELETE"].includes(request.method);
  const extra: Record<string, string> = {};
  const contentType = request.headers.get("content-type");
  if (hasBody && contentType) extra["Content-Type"] = contentType;
  const range = request.headers.get("range");
  if (range) extra["Range"] = range;

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: request.method,
      headers: getDataplaneHeaders(extra),
      body: hasBody ? request.body : undefined,
      // Required by Node's fetch to send a streamed request body.
      ...(hasBody ? { duplex: "half" } : {}),
      cache: "no-store",
    } as RequestInit);
  } catch (error) {
    console.error("Knowledge bank proxy error:", error);
    return NextResponse.json(
      { error: "Failed to reach the Hindsight API", details: (error as Error).message },
      { status: 502 }
    );
  }

  const headers = copyHeaders(upstream.headers);
  if (
    FILE_PATH.test(path.join("/")) &&
    /^application\/octet-stream/.test(headers.get("content-type") || "")
  ) {
    const ext = dispositionFilename(headers.get("content-disposition"))?.split(".").pop();
    const inferred = ext && TYPE_BY_EXTENSION[ext.toLowerCase()];
    if (inferred) headers.set("content-type", inferred);
  }
  const upstreamType = headers.get("content-type") || "";
  if (!upstreamType.startsWith("application/json")) {
    // Bytes from an upload are served on the control plane's origin: never let the
    // browser sniff them into HTML, and never let an uploaded page run script. Not for a
    // PDF: Chrome will not start its PDF viewer in a sandboxed document, so the original
    // showed blank, and the viewer keeps a PDF's own script out of this origin anyway.
    headers.set("X-Content-Type-Options", "nosniff");
    if (!upstreamType.startsWith("application/pdf")) {
      headers.set("Content-Security-Policy", "sandbox");
    }
  }

  const isMedia = request.method === "GET" && MEDIA_PATH.test(path.join("/"));
  if (isMedia && upstream.ok && upstream.status !== 206) {
    const bytes = new Uint8Array(await upstream.arrayBuffer());
    headers.set("Accept-Ranges", "bytes");
    const parsed = parseByteRange(range, bytes.length);
    if (parsed === "unsatisfiable") {
      headers.set("Content-Range", `bytes */${bytes.length}`);
      return new NextResponse(null, { status: 416, headers });
    }
    if (parsed) {
      headers.set("Content-Range", `bytes ${parsed.start}-${parsed.end}/${bytes.length}`);
      return new NextResponse(bytes.slice(parsed.start, parsed.end + 1), { status: 206, headers });
    }
    return new NextResponse(bytes, { status: 200, headers });
  }

  return new NextResponse(upstream.body, { status: upstream.status, headers });
}

export { forward as GET, forward as POST, forward as PUT, forward as PATCH, forward as DELETE };
