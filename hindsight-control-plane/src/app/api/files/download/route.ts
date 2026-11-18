import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

/**
 * Download a stored file (e.g. an async document-export archive) by its dataplane
 * path. The path comes from an operation's result_metadata.download_url; we proxy
 * it to the dataplane with server-side auth and stream the bytes back so the
 * browser downloads it directly.
 */
export async function GET(request: NextRequest) {
  try {
    const path = request.nextUrl.searchParams.get("path");
    // Only proxy the file-download endpoint — never an arbitrary dataplane path (SSRF guard).
    if (!path || !path.startsWith("/v1/default/files/download/")) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "A valid file download path is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const response = await fetch(`${DATAPLANE_URL}${path}`, { headers: getDataplaneHeaders() });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const body = await response.arrayBuffer();
    const fallbackName = path.split("/").pop() || "download.zip";
    // Always a download, never rendered: this route answers from the UI's own
    // origin, and a stored file's type is whatever it was stored with. The
    // dataplane's filename is kept; an "inline" disposition from it is not.
    const disposition = response.headers
      .get("content-disposition")
      ?.replace(/^\s*inline\b/i, "attachment");
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": response.headers.get("content-type") || "application/zip",
        "Content-Disposition": disposition?.startsWith("attachment")
          ? disposition
          : `attachment; filename="${fallbackName}"`,
        "Content-Security-Policy": "sandbox; default-src 'none'",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    console.error("Error downloading file:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to download file",
        errorKey: "api.errors.documents.export",
      }),
      { status: 500 }
    );
  }
}
