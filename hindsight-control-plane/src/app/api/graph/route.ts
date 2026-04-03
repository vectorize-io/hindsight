import { NextRequest, NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: NextRequest) {
  try {
    const tenant = request.nextUrl.searchParams.get("tenant");
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id") || searchParams.get("agent_id");

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
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
    if (tags.length > 0) params.append("tags_match", "all_strict");
    const documentId = searchParams.get("document_id");
    if (documentId) params.append("document_id", documentId);
    const chunkId = searchParams.get("chunk_id");
    if (chunkId) params.append("chunk_id", chunkId);

    const response = await fetch(`${DATAPLANE_URL}/v1/default/banks/${bankId}/graph?${params}`, {
      headers: getDataplaneHeaders(tenant),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error fetching graph data:", error);
    return NextResponse.json({ error: "Failed to fetch graph data" }, { status: 500 });
  }
}
