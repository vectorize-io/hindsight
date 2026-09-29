import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

/**
 * Serve an attachment retained as inline content, by bank and short id.
 *
 * Not a page — a byte proxy. Documents retained with inline attachments keep a
 * placeholder token where each one sat, and the UI turns those back into
 * pictures and file cards by pointing at this route. It exists because the
 * browser has no dataplane credentials: the API key lives server-side in
 * `getDataplaneHeaders()`, so the bytes are fetched here and streamed on.
 *
 * The Content-Type the dataplane returns is passed through, which is the
 * caller's own declared type — see the retain docs for what that implies about
 * storing active content.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string; attachmentId: string }> }
) {
  try {
    const { bankId, attachmentId } = await params;

    // The id is a hex prefix of the attachment's sha256 and nothing else.
    // Validating it here keeps a caller from steering the proxied path anywhere
    // but the attachment endpoint.
    if (!bankId || !/^[0-9a-f]{12}$/.test(attachmentId)) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "A bank id and a valid attachment id are required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const path = `/v1/default/banks/${encodeURIComponent(bankId)}/attachments/${attachmentId}`;
    const response = await fetch(`${DATAPLANE_URL}${path}`, { headers: getDataplaneHeaders() });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const body = await response.arrayBuffer();
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": response.headers.get("content-type") || "application/octet-stream",
        // Content-addressed, so the bytes behind this URL can never change.
        "Cache-Control": "private, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    console.error("Error fetching attachment:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch attachment",
        errorKey: "api.errors.documents.export",
      }),
      { status: 500 }
    );
  }
}
