import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    // Parse request body for optional subtype filter
    let body: { subtype?: "structural" | "emergent"; tags?: string[] } | undefined;
    try {
      const text = await request.text();
      if (text) {
        body = JSON.parse(text);
      }
    } catch {
      // Empty body is fine
    }

    const response = await sdk.refreshMentalModels({
      client: lowLevelClient,
      path: { bank_id: bankId },
      body: body,
    });

    if (response.error) {
      console.error("API error refreshing mental models:", response.error);
      return NextResponse.json({ error: "Failed to refresh mental models" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error refreshing mental models:", error);
    return NextResponse.json({ error: "Failed to refresh mental models" }, { status: 500 });
  }
}
