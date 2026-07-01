import { NextResponse } from "next/server";

export async function GET() {
  try {
    const apiUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
    const response = await fetch(`${apiUrl}/openapi.json`, {
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Failed to fetch OpenAPI spec" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to fetch OpenAPI spec:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
