import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;
    const { searchParams } = new URL(request.url);
    const tags = searchParams.getAll("tags");
    const tagsMatch = searchParams.get("tags_match");

    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
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
      `/directives${queryParams.toString() ? `?${queryParams}` : ""}`
    );
    const response = await fetch(url, { method: "GET", headers: getDataplaneHeaders() });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error listing directives:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to list directives",
          errorKey: "api.errors.directives.list",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error listing directives:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list directives",
        errorKey: "api.errors.directives.list",
      }),
      { status: 500 }
    );
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;

    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const body = await request.json();

    const response = await fetch(dataplaneBankUrl(bankId, "/directives"), {
      method: "POST",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error creating directive:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: errorText || "Failed to create directive",
          errorKey: "api.errors.directives.create",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 201 });
  } catch (error) {
    console.error("Error creating directive:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to create directive",
        errorKey: "api.errors.directives.create",
      }),
      { status: 500 }
    );
  }
}
