import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

function bankId(request: NextRequest): string | null {
  return request.nextUrl.searchParams.get("bank_id");
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ nodeId: string }> }
) {
  try {
    const { nodeId } = await params;
    const bank = bankId(request);
    if (!bank) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }
    const body = await request.json();
    const response = await fetch(
      dataplaneBankUrl(
        bank,
        `/knowledge-base/nodes/${encodeURIComponent(decodeURIComponent(nodeId))}`
      ),
      {
        method: "PATCH",
        headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
      }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }
    return NextResponse.json(await response.json(), { status: 200 });
  } catch (error) {
    console.error("Failed to update node:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to update node",
        errorKey: "api.errors.knowledgeBase.updateNode",
      }),
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ nodeId: string }> }
) {
  try {
    const { nodeId } = await params;
    const bank = bankId(request);
    if (!bank) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }
    const response = await fetch(
      dataplaneBankUrl(
        bank,
        `/knowledge-base/nodes/${encodeURIComponent(decodeURIComponent(nodeId))}`
      ),
      { method: "DELETE", headers: getDataplaneHeaders() }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }
    return NextResponse.json(await response.json(), { status: 200 });
  } catch (error) {
    console.error("Failed to delete node:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to delete node",
        errorKey: "api.errors.knowledgeBase.deleteNode",
      }),
      { status: 500 }
    );
  }
}
