import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const response = await sdk.listMentalModels({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });

    if (response.error) {
      console.error("API error listing mental models:", response.error);
      return NextResponse.json({ error: "Failed to list mental models" }, { status: 500 });
    }

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error listing mental models:", error);
    return NextResponse.json({ error: "Failed to list mental models" }, { status: 500 });
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const body = await request.json();

    // Call the dataplane API directly since SDK may not have the new endpoint yet
    const response = await fetch(`${DATAPLANE_URL}/v1/default/banks/${bankId}/mental-models`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error creating mental model:", errorText);
      return NextResponse.json(
        { error: errorText || "Failed to create mental model" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 201 });
  } catch (error) {
    console.error("Error creating mental model:", error);
    return NextResponse.json({ error: "Failed to create mental model" }, { status: 500 });
  }
}
