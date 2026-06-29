import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id") || "default";
    const limit = searchParams.get("limit") || "20";

    const apiUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";
    const response = await fetch(`${apiUrl}/v1/default/banks/${bankId}/operations?limit=${limit}`);

    if (!response.ok) {
      return NextResponse.json(
        { error: "Failed to fetch operations" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to fetch operations:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
