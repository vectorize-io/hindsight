import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function GET(request: NextRequest, { params }: { params: Promise<{ bankId: string }> }) {
  try {
    const { bankId } = await params;
    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Bank ID is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const includeHistory = request.nextUrl.searchParams.get("include_history") === "true";
    const suffix = `/backup${includeHistory ? "?include_history=true" : ""}`;
    const response = await fetch(dataplaneBankUrl(bankId, suffix), {
      headers: getDataplaneHeaders(),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const body = await response.arrayBuffer();
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": response.headers.get("content-type") || "application/zip",
        "Content-Disposition":
          response.headers.get("content-disposition") || `attachment; filename="${bankId}-bank-backup.zip"`,
      },
    });
  } catch (error) {
    console.error("Error backing up bank:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to backup bank",
        errorKey: "api.errors.bank.backup",
      }),
      { status: 500 }
    );
  }
}
