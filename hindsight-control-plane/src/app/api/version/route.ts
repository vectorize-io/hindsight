import { NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function GET() {
  try {
    const response = await sdk.getVersion({
      client: lowLevelClient,
    });

    if (response.error) {
      console.error("API error getting version:", response.error);
      return NextResponse.json({ error: "Failed to get version" }, { status: 500 });
    }

    const data = response.data as Record<string, unknown>;
    const features = (data.features ?? {}) as Record<string, boolean>;
    features.access_key_auth = !!process.env.HINDSIGHT_CP_ACCESS_KEY;
    data.features = features;

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error getting version:", error);
    return NextResponse.json({ error: "Failed to get version" }, { status: 500 });
  }
}
