import { NextRequest, NextResponse } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;

    const response = await sdk.getBankConfig({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });

    if (!response.data) {
      console.error("[Bank Config API] No data in response", { response, error: response.error });
      throw new Error(`API returned no data: ${JSON.stringify(response.error || "Unknown error")}`);
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching bank config:", error);
    return NextResponse.json({ error: "Failed to fetch bank config" }, { status: 500 });
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;
    const body = await request.json();
    const { updates } = body;

    const response = await sdk.updateBankConfig({
      client: lowLevelClient,
      path: { bank_id: bankId },
      body: { updates },
    });

    if (!response.data) {
      console.error("[Bank Config API] No data in response", { response, error: response.error });
      throw new Error(`API returned no data: ${JSON.stringify(response.error || "Unknown error")}`);
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error updating bank config:", error);
    return NextResponse.json({ error: "Failed to update bank config" }, { status: 500 });
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;

    const response = await sdk.resetBankConfig({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });

    if (!response.data) {
      console.error("[Bank Config API] No data in response", { response, error: response.error });
      throw new Error(`API returned no data: ${JSON.stringify(response.error || "Unknown error")}`);
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error resetting bank config:", error);
    return NextResponse.json({ error: "Failed to reset bank config" }, { status: 500 });
  }
}
