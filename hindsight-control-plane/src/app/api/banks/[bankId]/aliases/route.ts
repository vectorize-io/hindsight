import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
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

    const response = await fetch(dataplaneBankUrl(bankId, "/aliases"), {
      method: "GET",
      headers: getDataplaneHeaders(),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API error listing bank aliases:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Failed to list bank aliases",
          errorKey: "api.errors.bankAliases.list",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error listing bank aliases:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to list bank aliases",
        errorKey: "api.errors.bankAliases.list",
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

    const response = await fetch(dataplaneBankUrl(bankId, "/aliases"), {
      method: "POST",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      // The name being taken (409) is the ordinary outcome of a typo or a retry,
      // not a fault, so the dataplane's own message is passed through — it says
      // which name collided, which the generic key cannot.
      const errorText = await response.text();
      console.error("API error creating bank alias:", errorText);
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: errorText || "Failed to create bank alias",
          errorKey: "api.errors.bankAliases.create",
        }),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 201 });
  } catch (error) {
    console.error("Error creating bank alias:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to create bank alias",
        errorKey: "api.errors.bankAliases.create",
      }),
      { status: 500 }
    );
  }
}
