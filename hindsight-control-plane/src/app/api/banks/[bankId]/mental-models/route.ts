import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;
    const { searchParams } = new URL(request.url);
    const tags = searchParams.getAll("tags");
    const tagsMatch = searchParams.get("tags_match");

    if (!bankId) {
      return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
    }

    const queryParams = new URLSearchParams();
    if (tags.length > 0) {
      tags.forEach((t) => queryParams.append("tags", t));
    }
    if (tagsMatch) {
      queryParams.append("tags_match", tagsMatch);
    }

    const url = dataplaneBankUrl(
      bankId,
      `/mental-models${queryParams.toString() ? `?${queryParams}` : ""}`
    );
    const response = await fetch(url, { method: "GET", headers: getDataplaneHeaders() });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error listing mental models:", errorText);
      return NextResponse.json(
        { error: "Failed to list mental models" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
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

    const response = await fetch(dataplaneBankUrl(bankId, "/mental-models"), {
      method: "POST",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
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
    // Returns operation_id - content is generated in background
    return NextResponse.json(data, { status: 202 });
  } catch (error) {
    console.error("Error creating mental model:", error);
    return NextResponse.json({ error: "Failed to create mental model" }, { status: 500 });
  }
}
