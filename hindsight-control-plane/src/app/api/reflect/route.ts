import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const bankId = body.bank_id || body.agent_id || "default";
    const { query, context, budget, thinking_budget, include_facts, tags, tags_match } = body;

    const requestBody: any = {
      query,
      budget: budget || (thinking_budget ? "mid" : "low"),
      context: context || undefined,
      tags,
      tags_match,
    };

    // Add include options if specified
    if (include_facts) {
      requestBody.include = {
        facts: {},
      };
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
