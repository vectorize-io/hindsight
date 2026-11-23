import { NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";
import { parseByteRange } from "@/lib/byte-range";

/**
 * The three knowledge-bank routes the SDK cannot serve: a file upload, which must stream
 * through without being buffered here, and the two binary downloads (a document's original
 * file and a passage's media), which need their headers kept and, for media, byte ranges.
 *
 * - The upload body is streamed with the browser's Content-Type, so the multipart
 *   boundary survives and a large upload is never held in memory.
 * - The dataplane does not answer Range requests, which <video> and Safari's <audio>
 *   need to seek. Passage media are bounded clips (one page, one 30-60s window), so the
 *   clip is buffered and the range served here. Original files can be hundreds of MB and
 *   are only ever downloaded, so they are never buffered.
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

function upstreamUrl(segments: string[]): string | null {
  // An encoded ".." still normalises as a dot segment in the URL parser, which would
  // let a caller step out of the knowledge-banks prefix.
  if (segments.some((s) => s === "." || s === ".." || s === "")) return null;
  return `${DATAPLANE_URL}/v1/default/knowledge-banks/${segments.map(encodeURIComponent).join("/")}`;
}

function copyHeaders(from: Headers): Headers {
  const headers = new Headers();
  for (const name of PASSTHROUGH_HEADERS) {
    const value = from.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

async function send(
  segments: string[],
  init: RequestInit & { duplex?: "half" }
): Promise<Response | NextResponse> {
  const url = upstreamUrl(segments);
  if (!url) return NextResponse.json({ error: "Invalid path" }, { status: 400 });
  try {
    return await fetch(url, { ...init, cache: "no-store" } as RequestInit);
  } catch (error) {
    console.error("Knowledge bank stream error:", error);
    return NextResponse.json(
      { error: "Failed to reach the Hindsight API", details: (error as Error).message },
      { status: 502 }
    );
  }
}

/** `POST /{bank}/files/ingest`: the multipart body streamed through, the JSON answer passed back. */
export async function streamUpload(request: Request, segments: string[]): Promise<Response> {
  const contentType = request.headers.get("content-type");
  const upstream = await send(segments, {
    method: "POST",
    headers: getDataplaneHeaders(contentType ? { "Content-Type": contentType } : {}),
    body: request.body,
    // Required by Node's fetch to send a streamed request body.
    duplex: "half",
  });
  if (upstream instanceof NextResponse) return upstream;
  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: copyHeaders(upstream.headers),
  });
}

/**
 * A document's original file (`kind: "file"`) or a passage's media (`kind: "media"`),
 * with the headers that keep an uploaded file from running on the control plane's origin.
 */
export async function streamBinary(
  request: Request,
  segments: string[],
  kind: "file" | "media"
): Promise<Response> {
  const range = request.headers.get("range");
  const upstream = await send(segments, {
    method: "GET",
    headers: getDataplaneHeaders(range ? { Range: range } : {}),
  });
  if (upstream instanceof NextResponse) return upstream;

  const headers = copyHeaders(upstream.headers);
  if (kind === "file" && /^application\/octet-stream/.test(headers.get("content-type") || "")) {
    const ext = dispositionFilename(headers.get("content-disposition"))?.split(".").pop();
    const inferred = ext && TYPE_BY_EXTENSION[ext.toLowerCase()];
    if (inferred) headers.set("content-type", inferred);
  }
  const type = headers.get("content-type") || "";
  if (!type.startsWith("application/json")) {
    // Bytes from an upload are served on the control plane's origin: never let the
    // browser sniff them into HTML, and never let an uploaded page run script. Not for a
    // PDF: Chrome will not start its PDF viewer in a sandboxed document, so the original
    // showed blank, and the viewer keeps a PDF's own script out of this origin anyway.
    headers.set("X-Content-Type-Options", "nosniff");
    if (!type.startsWith("application/pdf")) headers.set("Content-Security-Policy", "sandbox");
  }

  if (kind === "media" && upstream.ok && upstream.status !== 206) {
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
