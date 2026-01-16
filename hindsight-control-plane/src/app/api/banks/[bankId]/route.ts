import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function PUT(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;
    const body = await request.json();

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

    if (response.error) {
      console.error("API error updating bank:", response.error);
      return NextResponse.json({ error: "Failed to update bank" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error updating bank:", error);
    return NextResponse.json({ error: "Failed to update bank" }, { status: 500 });
  }
}

export async function PATCH(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;
    const body = await request.json();

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

    if (response.error) {
      console.error("API error patching bank:", response.error);
      return NextResponse.json({ error: "Failed to update bank" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error patching bank:", error);
    return NextResponse.json({ error: "Failed to update bank" }, { status: 500 });
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const response = await sdk.deleteBank({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });

    if (response.error) {
      console.error("API error deleting bank:", response.error);
      return NextResponse.json({ error: "Failed to delete bank" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error deleting bank:", error);
    return NextResponse.json({ error: "Failed to delete bank" }, { status: 500 });
  }
}
