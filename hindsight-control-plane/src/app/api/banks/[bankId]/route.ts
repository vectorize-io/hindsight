import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function PUT(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  const response = await sdk.createOrUpdateBank({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body: {
      name: body.name,
      mission: body.mission,
      disposition: body.disposition,
    },
  });
  return respondWithSdk(response, "Failed to update bank");
}

export async function PATCH(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  const response = await sdk.updateBank({
    client: lowLevelClient,
    path: { bank_id: bankId },
    body: {
      name: body.name,
      mission: body.mission,
      disposition: body.disposition,
    },
  });
  return respondWithSdk(response, "Failed to update bank");
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ bankId: string }> }
) {
  const { bankId } = await params;

  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }

  const response = await sdk.deleteBank({
    client: lowLevelClient,
    path: { bank_id: bankId },
  });
  return respondWithSdk(response, "Failed to delete bank");
}
