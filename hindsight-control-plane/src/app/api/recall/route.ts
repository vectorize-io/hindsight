import { NextRequest, NextResponse } from "next/server";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const bankId = body.bank_id || body.agent_id || "default";
    const {
      query,
      types,
      fact_type,
      max_tokens,
      trace,
      budget,
      include,
      query_timestamp,
      tags,
      tags_match,
    } = body;

    const response = await sdk.recallMemories({
      client: lowLevelClient,
      path: { bank_id: bankId },
      body: {
        query,
        types: types || fact_type,
        max_tokens,
        trace,
        budget: budget || "mid",
        include,
        query_timestamp,
        tags,
        tags_match,
      },
    });

    if (!response.data) {
      console.error("[Recall API] No data in response", { response, error: response.error });
      throw new Error(`API returned no data: ${JSON.stringify(response.error || "Unknown error")}`);
    }

    // Return a clean JSON object by spreading the response
    // This ensures any non-serializable properties are excluded
    const jsonResponse = {
      results: response.data.results,
      trace: response.data.trace,
      entities: response.data.entities,
      chunks: response.data.chunks,
    };

    return NextResponse.json(jsonResponse, { status: 200 });
  } catch (error) {
    console.error("Error recalling:", error);
    return NextResponse.json({ error: "Failed to recall" }, { status: 500 });
  }
}
