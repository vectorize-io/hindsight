import { NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const url = `${DATAPLANE_URL}/v1/default/banks`;
    const response = await fetch(url, {
      headers: getDataplaneHeaders(),
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error(`Dataplane returned ${response.status}`);
    }

    const data = await response.json();
    const banks = data.banks || data || [];
    return NextResponse.json({
      banks,
      count: Array.isArray(banks) ? banks.length : 0,
    });
  } catch (error) {
    console.error("Error fetching banks:", error);
    return NextResponse.json(
      { banks: [], count: 0, error: "Failed to fetch banks" },
      { status: 500 }
    );
  }
}
