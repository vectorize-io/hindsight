import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

/**
 * Export documents as a transfer ZIP archive.
 * Proxies GET /v1/default/banks/{bank_id}/document-transfer and streams the
 * binary zip back to the browser.
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");
    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const qs = new URLSearchParams();
    for (const id of searchParams.getAll("document_id")) {
      qs.append("document_id", id);
    }
    if (searchParams.get("include_observations") === "true") {
      qs.set("include_observations", "true");
    }
    const suffix = `/document-transfer${qs.toString() ? `?${qs.toString()}` : ""}`;

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
        "Content-Type": "application/zip",
        "Content-Disposition":
          response.headers.get("content-disposition") ||
          `attachment; filename="${bankId}-documents.zip"`,
      },
    });
  } catch (error) {
    console.error("Error exporting documents:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to export documents",
        errorKey: "api.errors.documents.export",
      }),
      { status: 500 }
    );
  }
}

/**
 * Import a transfer ZIP archive into a bank.
 * Proxies the multipart upload to POST /v1/default/banks/{bank_id}/document-transfer.
 */
export async function POST(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");
    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }
    const onConflict = searchParams.get("on_conflict") || "skip";

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
    const filename = file instanceof File ? file.name : "transfer.zip";
    outForm.append("file", file, filename);

    const suffix = `/document-transfer?on_conflict=${encodeURIComponent(onConflict)}`;
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
    console.error("Error importing documents:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to import documents",
        errorKey: "api.errors.documents.import",
      }),
      { status: 500 }
    );
  }
}
