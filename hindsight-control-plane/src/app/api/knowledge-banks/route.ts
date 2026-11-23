import { NextRequest, NextResponse } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const response = await sdk.listKnowledgeBanks({
    client: lowLevelClient,
    query: {
      limit: sp.has("limit") ? Number(sp.get("limit")) : undefined,
      offset: sp.has("offset") ? Number(sp.get("offset")) : undefined,
      q: sp.get("q") ?? undefined,
    },
  });
  return respondWithSdk(response, "Failed to list knowledge banks", { request });
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (body === null) return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  const response = await sdk.createKnowledgeBank({ client: lowLevelClient, body });
  return respondWithSdk(response, "Failed to create knowledge bank", { request });
}
