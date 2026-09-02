import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

/**
 * Serve an image retained as inline content, by bank and content hash.
 *
 * Documents retained with inline images keep a placeholder token where each
 * image sat; the UI turns those back into pictures by pointing an <img> at this
 * route. The bytes are proxied with server-side auth rather than exposed
 * directly, exactly as the export-archive download is.
 */
export async function GET(request: NextRequest) {
  try {
    const bankId = request.nextUrl.searchParams.get("bank_id");
    const imageId = request.nextUrl.searchParams.get("id");

    // The id is a hex prefix of the image's sha256 and nothing else. Validating it
    // here keeps a caller from steering the proxied path anywhere but the image
    // endpoint.
    if (!bankId || !imageId || !/^[0-9a-f]{12}$/.test(imageId)) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "A bank id and a valid image id are required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const path = `/v1/default/banks/${encodeURIComponent(bankId)}/images/${imageId}`;
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
      },
    });
  } catch (error) {
    console.error("Error fetching image:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch image",
        errorKey: "api.errors.documents.export",
      }),
      { status: 500 }
    );
  }
}
