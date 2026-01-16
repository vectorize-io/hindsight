import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const bankId = body.bank_id || body.agent_id || "default";
    const {
      query,
      budget,
      thinking_budget,
      include_facts,
      include_tool_calls,
      tags,
      tags_match,
      max_tokens,
    } = body;

    const requestBody: any = {
      query,
      budget: budget || (thinking_budget ? "mid" : "low"),
      tags,
      tags_match,
      max_tokens: max_tokens || undefined,
    };

    // Add include options if specified
    const includeOptions: any = {};
    if (include_facts) {
      includeOptions.facts = {};
    }
    if (include_tool_calls) {
      includeOptions.tool_calls = {};
    }
    if (Object.keys(includeOptions).length > 0) {
      requestBody.include = includeOptions;
    }

    const response = await sdk.reflect({
      client: lowLevelClient,
      path: { bank_id: bankId },
      body: requestBody,
    });

    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error reflecting:", error);
    return NextResponse.json({ error: "Failed to reflect" }, { status: 500 });
  }
}
