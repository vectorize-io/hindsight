import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id") || searchParams.get("agent_id");

    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    // Build query params for the dataplane
    const params = new URLSearchParams();
    const type = searchParams.get("type") || searchParams.get("fact_type");
    if (type) params.append("type", type);
    const limit = searchParams.get("limit");
    if (limit) params.append("limit", limit);
    const q = searchParams.get("q");
    if (q) params.append("q", q);
    const tags = searchParams.getAll("tags");
    for (const tag of tags) params.append("tags", tag);
    // Forward an explicit match mode if the caller set one (e.g. "exact" for
    // observation-scope filtering, which also drives the global/untagged scope
    // when no tags are present). Otherwise default to all_strict when tags are
    // present (tag isolation), matching prior behavior.
    const tagsMatch = searchParams.get("tags_match");
    if (tagsMatch) {
      params.append("tags_match", tagsMatch);
    } else if (tags.length > 0) {
      params.append("tags_match", "all_strict");
    }
    const documentId = searchParams.get("document_id");
    if (documentId) params.append("document_id", documentId);
    const chunkId = searchParams.get("chunk_id");
    if (chunkId) params.append("chunk_id", chunkId);

    // Forward the browser's abort: when the user leaves a tab, its graph read
    // stops here and at the dataplane instead of finishing for nobody.
    const response = await fetch(`${DATAPLANE_URL}/v1/default/banks/${bankId}/graph?${params}`, {
      headers: getDataplaneHeaders(),
      signal: request.signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    // Stream the body through without parsing it: on a big bank it is several
    // MB, and parsing it only to serialize it again costs the user half a second.
    // Gzip it on the way when the browser accepts that: graph JSON shrinks
    // ~13x, which is the difference between usable and not over a VPN (#5548).
    // `fetch` already decoded the dataplane's own gzip, so this is the only one.
    const gzip = /\bgzip\b/.test(request.headers.get("accept-encoding") ?? "");
    const body =
      gzip && response.body
        ? response.body.pipeThrough(new CompressionStream("gzip"))
        : response.body;
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        ...(gzip ? { "Content-Encoding": "gzip", Vary: "Accept-Encoding" } : {}),
      },
    });
  } catch (error) {
    // The browser hung up (tab switched); nobody is waiting for an answer.
    if (request.signal.aborted) return new NextResponse(null, { status: 499 });
    console.error("Error fetching graph data:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to fetch graph data",
        errorKey: "api.errors.graph.fetch",
      }),
      { status: 500 }
    );
  }
}
