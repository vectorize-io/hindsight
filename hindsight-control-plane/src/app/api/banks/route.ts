import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

const HTTP_CREATED = 201;

export async function GET() {
  const response = await sdk.listBanks({ client: lowLevelClient });
  return respondWithSdk(response, "Failed to fetch banks");
}

export async function POST(request: Request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  const { bank_id } = body;

  if (!bank_id) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  const response = await sdk.createOrUpdateBank({
    client: lowLevelClient,
    path: { bank_id },
    body: {},
  });
  return respondWithSdk(response, "Failed to create bank", HTTP_CREATED);
}
