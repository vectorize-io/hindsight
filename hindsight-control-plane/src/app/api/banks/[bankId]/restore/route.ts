import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
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

    const inForm = await request.formData();
    const file = inForm.get("file");
    if (!(file instanceof Blob)) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "file is required",
          errorKey: "api.errors.validation.fileRequired",
        }),
        { status: 400 }
      );
    }

    const outForm = new FormData();
    const filename = file instanceof File ? file.name : "bank-backup.zip";
    outForm.append("file", file, filename);

    const includeHistory = request.nextUrl.searchParams.get("include_history") === "true";
    const suffix = `/restore${includeHistory ? "?include_history=true" : ""}`;
    const response = await fetch(dataplaneBankUrl(bankId, suffix), {
      method: "POST",
      headers: getDataplaneHeaders(),
      body: outForm,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    return NextResponse.json(await response.json(), { status: 200 });
  } catch (error) {
    console.error("Error restoring bank:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to restore bank",
        errorKey: "api.errors.bank.restore",
      }),
      { status: 500 }
    );
  }
}
